"""
Pydantic request/response models for the Phase 1 (auth + basic chat) and
Phase 2 (document upload + library) endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


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
    used_documents: bool  # always False in Phase 1 -- no retrieval yet


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
