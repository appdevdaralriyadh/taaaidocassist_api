"""
Pydantic request/response models for the Phase 1 (auth + basic chat),
Phase 2 (document upload + library), Phase 3 (retrieval + query routing),
Phase 5 (data management -- clear knowledge base / clear chat history,
spec §3.6), and the final phase (OneDrive/Google Drive connectors, spec
§3.2) endpoints.
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


class ConversationListItem(BaseModel):
    id: uuid.UUID
    # "Untitled conversation" is filled in by the endpoint, never here --
    # keeping this Optional documents that DarAI_Conversations.Title can
    # genuinely be NULL (a conversation created but not yet titled).
    title: Optional[str] = None
    last_message_at: datetime


class ConversationRenameRequest(BaseModel):
    title: str


class ConversationDeleteResponse(BaseModel):
    conversation_id: uuid.UUID
    messages_deleted: int


class ClearMyConversationsResponse(BaseModel):
    conversations_deleted: int
    messages_deleted: int


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
    # Cosine DISTANCE from VECTOR_DISTANCE('cosine', ...) -- LOWER is
    # better (0 = identical, 2 = completely opposite). This is not the
    # old similarity score; renamed from `score` specifically so it can't
    # be misread as "higher is better" the way that field used to mean.
    distance: float
    chunk_text: str


class DebugRetrievalResponse(BaseModel):
    query: str
    max_distance: float
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


class GoogleDriveConnectRequest(BaseModel):
    # Short-lived Google OAuth token, acquired in the browser via Google
    # Identity Services right before this call (see
    # googledrive.component.ts) -- never persisted server-side, same
    # pattern OneDriveConnectRequest.access_token already uses for
    # Microsoft Graph.
    access_token: str
    folder_path: str  # pasted Google Drive folder link, or a bare folder ID
    display_label: Optional[str] = None


class OneDriveConnectRequest(BaseModel):
    access_token: str  # short-lived Microsoft Graph token from the frontend's MSAL session
    shared_link: str  # pasted OneDrive/SharePoint shared-folder link
    display_label: Optional[str] = None


class SyncRequest(BaseModel):
    # Required for both onedrive and googledrive connections -- a fresh
    # token, re-acquired by the frontend right before the call, for
    # whichever provider the connection belongs to. Optional at the
    # schema level only because this one request type is shared between
    # source types; app/api/routes/sources.py enforces it's actually
    # present for both.
    access_token: Optional[str] = None


class SyncFileResult(BaseModel):
    path: str
    status: str  # 'added' | 'duplicate' | 'skipped' | 'failed'
    detail: Optional[str] = None


class SyncResponse(BaseModel):
    connection_id: int
    files_added: int
    files_duplicate: int
    files_skipped: int
    files_failed: int
    details: list[SyncFileResult] = Field(default_factory=list)


class SourceConnectionItem(BaseModel):
    id: int
    source_type: str
    display_label: str
    created_by: Optional[str] = None
    created_at: datetime
    last_synced_at: Optional[datetime] = None
    last_sync_status: Optional[str] = None
    last_sync_error: Optional[str] = None


class ConnectResponse(BaseModel):
    connection: SourceConnectionItem
    sync: SyncResponse
