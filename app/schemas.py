"""
Pydantic request/response models for the Phase 1 (auth + basic chat),
Phase 2 (document upload + library), Phase 3 (retrieval + query routing),
and Phase 5 (data management -- clear knowledge base / clear chat
history, spec §3.6) endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    entra_token: str  # the Entra ID token (idToken) from MSAL.js


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    display_name: Optional[str] = None


class ChatMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[uuid.UUID] = None  # omitted -> starts a new conversation


class ChatMessageResponse(BaseModel):
    conversation_id: uuid.UUID
    reply: str
    used_documents: bool  # True when routed to the knowledge base (spec §3.4)
    sources: list[str] = Field(default_factory=list)  # filenames used, [] if general knowledge


class ChatHistoryItem(BaseModel):
    role: str
    message: str
    used_documents: bool
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    chunk_count: int
    is_duplicate: bool  # flagged, never blocked -- see spec §7


class DocumentListItem(BaseModel):
    id: int
    filename: str
    source_type: str
    uploaded_by: Optional[str] = None
    uploaded_at: datetime
    chunk_count: int


class DebugRetrievalChunk(BaseModel):
    filename: str
    score: float
    chunk_text: str


class DebugRetrievalResponse(BaseModel):
    query: str
    threshold: float
    matches: list[DebugRetrievalChunk]


class AccountListItem(BaseModel):
    id: int
    display_name: Optional[str] = None


class ClearKnowledgeBaseRequest(BaseModel):
    # Server-side enforced, not just a UI gate (spec §3.3: "type DELETE to
    # confirm") -- a direct API call without this exact literal is
    # rejected too.
    confirm: str


class ClearKnowledgeBaseResponse(BaseModel):
    documents_deleted: int
    chunks_deleted: int


class ClearChatHistoryRequest(BaseModel):
    # Either account can clear its own or the other account's history
    # (spec §3.5); one or more ids in a single call.
    user_ids: list[int]


class ClearChatHistoryResponse(BaseModel):
    messages_deleted: int
    cleared_user_ids: list[int]
