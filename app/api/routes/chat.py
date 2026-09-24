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
