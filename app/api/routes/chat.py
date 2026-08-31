"""
Basic chat -- Phase 1 (spec §8 build order, item 1).

No retrieval yet: every message goes straight to Claude and every reply is
general knowledge, logged with UsedDocuments=False. Retrieval and the
knowledge-base-vs-general-knowledge query routing (spec §3.4) are added in
Phase 3, once documents can actually be ingested (Phase 2).
"""

import uuid

from anthropic import Anthropic
from fastapi import APIRouter, Depends
from sqlalchemy import asc
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.config import settings
from app.db.models import ChatMessage, User
from app.db.session import get_db
from app.schemas import ChatHistoryItem, ChatMessageRequest, ChatMessageResponse

router = APIRouter()

_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

_SYSTEM_PROMPT = (
    "You are a helpful assistant embedded in an internal document "
    "processing and chat application. Answer from your general knowledge."
)


@router.post("/message", response_model=ChatMessageResponse)
def send_message(
    payload: ChatMessageRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation_id = payload.conversation_id or uuid.uuid4()

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
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    reply_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

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
            Message=reply_text,
            UsedDocuments=False,
        )
    )
    db.commit()

    return ChatMessageResponse(
        conversation_id=conversation_id,
        reply=reply_text,
        used_documents=False,
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
