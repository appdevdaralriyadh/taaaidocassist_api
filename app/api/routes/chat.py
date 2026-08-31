"""
Chat + retrieval-based query routing (spec §3.4, §8 build order item 3).

Each message is embedded and checked against the shared knowledge base via
app/services/retrieval.py's numpy cosine-similarity search. Clearing
SIMILARITY_THRESHOLD *is* the routing decision -- there's no separate
classifier call:
  - Above threshold -> answer generated from the matched chunks, with a
    deterministic "Source: filename" citation appended to what gets
    persisted (never left to the model to cite reliably on its own),
    UsedDocuments=True.
  - Below threshold, or an empty knowledge base -> general knowledge,
    same behavior as Phase 1, UsedDocuments=False.

Every message logs which path was taken and the top similarity score --
spec §3.4: "useful for debugging the classifier/threshold over time."
"""

import logging
import uuid

from anthropic import Anthropic
from fastapi import APIRouter, Depends, Query
from sqlalchemy import asc
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.config import settings
from app.db.models import ChatMessage, User
from app.db.session import get_db
from app.schemas import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    DebugRetrievalChunk,
    DebugRetrievalResponse,
)
from app.services import retrieval

router = APIRouter()
logger = logging.getLogger("darai.chat")

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

    top_score = matches[0].score if matches else None
    logger.info(
        "conversation=%s route=%s top_score=%s threshold=%.2f matched_chunks=%d",
        conversation_id,
        "knowledge_base" if used_documents else "general",
        f"{top_score:.3f}" if top_score is not None else "n/a",
        settings.SIMILARITY_THRESHOLD,
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


@router.get("/debug-retrieval", response_model=DebugRetrievalResponse)
def debug_retrieval(
    q: str = Query(..., description="Test query to run through the retrieval search"),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Runs the same embed + similarity search used to route live chat
    messages, without calling Claude -- for tuning SIMILARITY_THRESHOLD /
    TOP_K_CHUNKS against your real documents without burning API calls.
    """
    matches = retrieval.search(db, q)
    return DebugRetrievalResponse(
        query=q,
        threshold=settings.SIMILARITY_THRESHOLD,
        matches=[
            DebugRetrievalChunk(filename=m.filename, score=m.score, chunk_text=m.chunk_text)
            for m in matches
        ],
    )
