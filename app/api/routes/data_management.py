"""
Data management (spec §3.6, §8 build order item 5): equal-permission,
destructive controls shared by both accounts --

  - Clear knowledge base: wipes every DarAI_DocumentChunks row (plus the
    version history, archived chunks and Deleted documents), then every
    DarAI_Documents row (child rows first -- DocumentChunks.DocumentId has
    a FOREIGN KEY REFERENCES DarAI_Documents(Id), spec §4 -- so deleting
    the parent first would violate the constraint). Requires the literal
    confirmation string "DELETE" in the request body (spec §3.3's "type
    DELETE to confirm", enforced here server-side, not only in the UI --
    a direct API call without it is rejected too).

  - Clear chat history: deletes DarAI_ChatHistory rows for one or more
    selected accounts, then removes any DarAI_Conversations row left with
    zero messages (so the Angular sidebar's conversation list actually
    empties out too, not just the chat history view). Either account can
    clear its own or the other account's history (spec §3.5) -- no role
    check, since both accounts have identical permissions (spec §3.1);
    the frontend's own Yes/No confirmation is the gate here (no
    typed-confirmation requirement on this endpoint).

Both endpoints are additive-safe in the sense that they only ever touch
the table(s) they say they touch: clearing history never touches
Documents/DocumentChunks, and vice versa.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import exists
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.models import (
    ChatMessage,
    Conversation,
    Document,
    DocumentChunk,
    DocumentChunkArchive,
    DocumentMatchExclusion,
    DocumentVersion,
    User,
)
from app.db.session import get_db
from app.schemas import (
    AccountListItem,
    ClearChatHistoryRequest,
    ClearChatHistoryResponse,
    ClearKnowledgeBaseRequest,
    ClearKnowledgeBaseResponse,
)

router = APIRouter()

_REQUIRED_CONFIRMATION = "DELETE"


@router.get("/accounts", response_model=list[AccountListItem])
def list_accounts(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Every account that can appear as a chat-history-clear target -- not
    hardcoded to "2 users" (spec §1's current reality), so this keeps
    working if a third account is ever added to the Entra app
    registration.
    """
    users = db.query(User).order_by(User.DisplayName).all()
    return [AccountListItem(id=u.Id, display_name=u.DisplayName) for u in users]


@router.post("/knowledge-base/clear", response_model=ClearKnowledgeBaseResponse)
def clear_knowledge_base(
    payload: ClearKnowledgeBaseRequest,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.confirm != _REQUIRED_CONFIRMATION:
        raise HTTPException(
            status_code=400,
            detail=f'Confirmation text must be exactly "{_REQUIRED_CONFIRMATION}".',
        )

    # Child rows first: DarAI_DocumentChunks.DocumentId is a FOREIGN KEY
    # REFERENCES DarAI_Documents(Id) (spec §4) with no ON DELETE CASCADE
    # specified, so deleting Documents first would fail the constraint.
    # Everything document-related goes, including Deleted documents and
    # version history (004_versions_archive_settings.sql): "never merge"
    # pairs first (their FKs to Documents have no cascade), then archived
    # chunks and version rows, then the documents themselves. Counts are
    # for the knowledge base proper (live chunks, and documents that were
    # Active), as before.
    db.query(DocumentMatchExclusion).delete(synchronize_session=False)
    chunks_deleted = db.query(DocumentChunk).delete(synchronize_session=False)
    db.query(DocumentChunkArchive).delete(synchronize_session=False)
    db.query(DocumentVersion).delete(synchronize_session=False)
    documents_deleted = (
        db.query(Document).filter(Document.Status == "Active").count()
    )
    db.query(Document).delete(synchronize_session=False)
    db.commit()

    return ClearKnowledgeBaseResponse(
        documents_deleted=documents_deleted,
        chunks_deleted=chunks_deleted,
    )


@router.post("/chat-history/clear", response_model=ClearChatHistoryResponse)
def clear_chat_history(
    payload: ClearChatHistoryRequest,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not payload.user_ids:
        raise HTTPException(status_code=400, detail="user_ids must not be empty.")

    requested_ids = set(payload.user_ids)
    existing_ids = {
        row[0] for row in db.query(User.Id).filter(User.Id.in_(requested_ids)).all()
    }
    unknown_ids = requested_ids - existing_ids
    if unknown_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown user id(s): {sorted(unknown_ids)}",
        )

    messages_deleted = (
        db.query(ChatMessage)
        .filter(ChatMessage.UserId.in_(requested_ids))
        .delete(synchronize_session=False)
    )

    # A conversation left with zero messages after this clear is dead
    # weight in the Angular sidebar's conversation list -- without this,
    # "Clear chat history" removed the messages but the (now-empty)
    # conversation kept showing up there. Scoped to "no messages remain
    # at all", not just "none from these user_ids", so a conversation the
    # OTHER account is still actively using (either account can post into
    # either conversation, spec §3.5) is left alone.
    db.query(Conversation).filter(
        ~exists().where(ChatMessage.ConversationId == Conversation.ConversationId)
    ).delete(synchronize_session=False)

    db.commit()

    return ClearChatHistoryResponse(
        messages_deleted=messages_deleted,
        cleared_user_ids=sorted(requested_ids),
    )
