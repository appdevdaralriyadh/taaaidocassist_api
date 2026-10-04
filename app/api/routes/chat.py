"""
Chat + retrieval-based query routing (spec §3.4, §8 build order item 3).

Each message is embedded and checked against the shared knowledge base via
app/services/retrieval.py's SQL Server VECTOR_DISTANCE search. Clearing
MAX_COSINE_DISTANCE *is* the routing decision -- there's no separate
classifier call:
  - At or below the cutoff -> answer generated from the matched chunks,
    with a deterministic "Source: filename" citation appended to what
    gets persisted (never left to the model to cite reliably on its
    own), UsedDocuments=True.
  - Above the cutoff, or an empty knowledge base -> general knowledge,
    same behavior as Phase 1, UsedDocuments=False.

Every message logs which path was taken and the best (lowest) distance --
spec §3.4: "useful for debugging the classifier/threshold over time."
Note this is a DISTANCE, not the old similarity score -- lower is better.
"""

import logging
import uuid
from typing import Optional

from anthropic import Anthropic
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.config import settings
from app.db.models import ChatMessage, Conversation, User
from app.db.session import get_db
from app.schemas import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    ClearMyConversationsResponse,
    ConversationDeleteResponse,
    ConversationListItem,
    ConversationRenameRequest,
    DebugRetrievalChunk,
    DebugRetrievalResponse,
)
from app.services import retrieval

router = APIRouter()
logger = logging.getLogger("darai.chat")

# Sidebar conversation titles: same truncation rule the frontend used to
# apply client-side (ChatStateService.registerConversation, now retired in
# favor of this server-side title) -- kept here so the title is generated
# once, consistently, no matter which client sends the first message.
_TITLE_MAX_LEN = 40


def _make_title(first_message: str) -> str:
    text = first_message.strip()
    if len(text) > _TITLE_MAX_LEN:
        return f"{text[:_TITLE_MAX_LEN]}…"
    return text or "Untitled conversation"

_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

_GENERAL_SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in an internal document "
    "processing and chat application. Answer from your general knowledge."
)

_KB_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant embedded in an internal document "
    "processing and chat application. Answer the user's question using "
    "ONLY the context below, drawn from the shared document knowledge "
    "base. If the context doesn't actually contain the answer, say so "
    "plainly rather than guessing or falling back to outside knowledge.\n\n"
    "--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
)


def _build_context_block(chunks: list[retrieval.RetrievedChunk]) -> str:
    return "\n\n".join(f"[Source: {c.filename}]\n{c.chunk_text}" for c in chunks)


@router.post("/message", response_model=ChatMessageResponse)
def send_message(
    payload: ChatMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation_id = payload.conversation_id or uuid.uuid4()

    matches = retrieval.search(db, payload.message)
    used_documents = len(matches) > 0
    sources = sorted({m.filename for m in matches}) if used_documents else []

    best_distance = matches[0].distance if matches else None
    logger.info(
        "conversation=%s route=%s best_distance=%s max_distance=%.2f matched_chunks=%d",
        conversation_id,
        "knowledge_base" if used_documents else "general",
        f"{best_distance:.3f}" if best_distance is not None else "n/a",
        settings.MAX_COSINE_DISTANCE,
        len(matches),
    )

    system_prompt = (
        _KB_SYSTEM_PROMPT_TEMPLATE.format(context=_build_context_block(matches))
        if used_documents
        else _GENERAL_SYSTEM_PROMPT
    )

    # Recent history for this conversation gives the model continuity
    # (spec §3.5). Not filtered by UserId: either account can see either
    # account's history, so if the admin continues a user's conversation
    # the model should see the same thread they saw.
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.ConversationId == conversation_id)
        .order_by(asc(ChatMessage.CreatedAt))
        .all()
    )

    messages = [{"role": m.Role, "content": m.Message} for m in history]
    messages.append({"role": "user", "content": payload.message})

    response = _client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    reply_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    # The citation is appended deterministically from the chunks actually
    # retrieved -- never left to the model to cite correctly on its own --
    # and only into what's *persisted*, so the raw DB record stays
    # self-describing (spec §3.4: "with a 'Source: filename' citation")
    # even read outside the app. The live response keeps `reply` clean and
    # returns `sources` separately so the frontend can badge it instead.
    stored_text = reply_text
    if used_documents:
        stored_text = f"{reply_text}\n\nSource: {', '.join(sources)}"

    db.add(
        ChatMessage(
            UserId=user.Id,
            ConversationId=conversation_id,
            Role="user",
            Message=payload.message,
            UsedDocuments=False,
        )
    )
    db.add(
        ChatMessage(
            UserId=user.Id,
            ConversationId=conversation_id,
            Role="assistant",
            Message=stored_text,
            UsedDocuments=used_documents,
        )
    )

    # Sidebar conversation-list bookkeeping only -- doesn't touch anything
    # above this point (retrieval, routing, the Claude call, or what gets
    # persisted as chat history). Get-or-create rather than branching on
    # "payload.conversation_id was None", so a conversation started before
    # DarAI_Conversations existed (see 002_add_conversations_table.sql's
    # backfill) still gets picked up correctly on its next message instead
    # of erroring or silently staying untitled.
    conversation_row = (
        db.query(Conversation).filter(Conversation.ConversationId == conversation_id).one_or_none()
    )
    if conversation_row is None:
        db.add(
            Conversation(
                ConversationId=conversation_id,
                UserId=user.Id,
                Title=_make_title(payload.message),
                LastMessageAt=func.sysutcdatetime(),
            )
        )
    else:
        conversation_row.LastMessageAt = func.sysutcdatetime()

    db.commit()

    return ChatMessageResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        used_documents=used_documents,
        sources=sources,
    )


@router.get("/history/{conversation_id}", response_model=list[ChatHistoryItem])
def get_history(
    conversation_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Either account can view either account's chat history (spec §3.5) --
    # deliberately not filtered by UserId, only by conversation.
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.ConversationId == conversation_id)
        .order_by(asc(ChatMessage.CreatedAt))
        .all()
    )
    return [
        ChatHistoryItem(
            role=r.Role,
            message=r.Message,
            used_documents=r.UsedDocuments,
            created_at=r.CreatedAt,
        )
        for r in rows
    ]


@router.get("/conversations", response_model=list[ConversationListItem])
def list_conversations(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Backs the Angular sidebar's conversation list. Shared visibility, same
    as get_history() above -- every conversation is returned regardless of
    which account started it, not just the caller's own.
    """
    rows = db.query(Conversation).order_by(desc(Conversation.LastMessageAt)).all()
    return [
        ConversationListItem(id=r.ConversationId, title=r.Title, last_message_at=r.LastMessageAt)
        for r in rows
    ]


@router.patch("/conversations/{conversation_id}", response_model=ConversationListItem)
def rename_conversation(
    conversation_id: uuid.UUID,
    payload: ConversationRenameRequest,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.query(Conversation).filter(Conversation.ConversationId == conversation_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title cannot be empty.")

    row.Title = title[:200]
    db.commit()
    db.refresh(row)
    return ConversationListItem(id=row.ConversationId, title=row.Title, last_message_at=row.LastMessageAt)


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
def delete_conversation(
    conversation_id: uuid.UUID,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Permanently removes the conversation and every message in it (matches
    the sidebar's "will be permanently removed" delete-confirmation copy).
    Not restricted to the caller's own conversation -- consistent with
    get_history()/list_conversations()'s shared-visibility design.
    """
    row = db.query(Conversation).filter(Conversation.ConversationId == conversation_id).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    messages_deleted = (
        db.query(ChatMessage)
        .filter(ChatMessage.ConversationId == conversation_id)
        .delete(synchronize_session=False)
    )
    db.delete(row)
    db.commit()
    return ConversationDeleteResponse(conversation_id=conversation_id, messages_deleted=messages_deleted)


@router.delete("/conversations", response_model=ClearMyConversationsResponse)
def clear_my_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Backs the sidebar's "Clear All" -- deliberately scoped to the caller's
    OWN conversations only (Conversation.UserId, i.e. whoever's first
    message created each one), unlike the shared-visibility endpoints
    above. A bulk destructive action from a single click shouldn't also be
    able to wipe the other account's conversations; that's what Data
    Management's typed-confirmation "clear chat history" flow is for.
    """
    conversation_ids = [
        row.ConversationId
        for row in db.query(Conversation.ConversationId).filter(Conversation.UserId == user.Id).all()
    ]

    messages_deleted = 0
    if conversation_ids:
        messages_deleted = (
            db.query(ChatMessage)
            .filter(ChatMessage.ConversationId.in_(conversation_ids))
            .delete(synchronize_session=False)
        )
        db.query(Conversation).filter(Conversation.UserId == user.Id).delete(synchronize_session=False)

    db.commit()
    return ClearMyConversationsResponse(
        conversations_deleted=len(conversation_ids),
        messages_deleted=messages_deleted,
    )


@router.get("/debug-retrieval", response_model=DebugRetrievalResponse)
def debug_retrieval(
    q: str = Query(..., description="Test query to run through the retrieval search"),
    max_distance: Optional[float] = Query(
        None,
        description=(
            "Override MAX_COSINE_DISTANCE for this call only -- doesn't touch the "
            "configured value or live chat routing. This is a DISTANCE (lower is "
            "better: 0 = identical, 2 = completely opposite), the inverse of the old "
            "similarity score this param used to be named `threshold` for. "
            "retrieval.search() sorts matches ascending and stops after the first one "
            "above the cutoff, so the *real* configured value can never show you what "
            "an above-cutoff distance actually is. Pass 2 here (the maximum possible "
            "cosine distance) to see every chunk's true raw distance -- this is the "
            "only way to see the actual gap between a genuine match and background "
            "noise when recalibrating MAX_COSINE_DISTANCE after an embedding model "
            "change."
        ),
    ),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Runs the same embed + vector-distance search used to route live chat
    messages, without calling Claude -- for tuning MAX_COSINE_DISTANCE /
    TOP_K_CHUNKS against your real documents without burning API calls.
    """
    effective_max_distance = (
        settings.MAX_COSINE_DISTANCE if max_distance is None else max_distance
    )
    matches = retrieval.search(db, q, max_distance=effective_max_distance)
    return DebugRetrievalResponse(
        query=q,
        max_distance=effective_max_distance,
        matches=[
            DebugRetrievalChunk(filename=m.filename, distance=m.distance, chunk_text=m.chunk_text)
            for m in matches
        ],
    )
