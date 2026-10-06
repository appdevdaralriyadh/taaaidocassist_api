"""
Pydantic request/response models for the Phase 1 (auth + basic chat),
Phase 2 (document upload + library), Phase 3 (retrieval + query routing),
Phase 5 (data management -- clear knowledge base / clear chat history,
spec §3.6), and the final phase (OneDrive/Google Drive connectors, spec
§3.2) endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional, Union

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
    id: int  # the stored document (for 'unchanged': the existing identical one)
    filename: str
    chunk_count: int
    # Kept for compatibility: True exactly when outcome == 'unchanged'.
    is_duplicate: bool
    # 'new' | 'updated' | 'unchanged' | 'older_version' | 'restored' (see
    # app/services/ingestion.py's ingest_local_upload)
    outcome: str
    version: int
    previous_version: Optional[int] = None
    # outcome == 'updated': the version that was replaced (its filename may
    # differ from the new one) and who/when stored it; how it was
    # recognised ('filename' | 'name_and_content' | 'content', with the
    # two-way wording similarity 0.0-1.0 for the last two); and any other
    # older copies/versions removed with it
    replaced_filename: Optional[str] = None
    replaced_at: Optional[datetime] = None
    replaced_by: Optional[str] = None
    matched_by: Optional[str] = None
    similarity: Optional[float] = None
    # the higher of the two directions, for 'content' matches/flags
    similarity_max: Optional[float] = None
    removed_copies: int = 0
    removed_filenames: list[str] = Field(default_factory=list)
    # outcome == 'unchanged': the existing document with identical content
    # outcome == 'older_version': the existing NEWER document (upload not stored)
    # outcome == 'new' + needs_review: the document it looked like a version of
    matched_filename: Optional[str] = None
    matched_source_type: Optional[str] = None
    # outcome == 'older_version': the years that decided it
    upload_year: Optional[int] = None
    existing_year: Optional[int] = None
    # outcome == 'new': True when it looked like a version of an existing
    # document but didn't meet the automatic rules, so both were kept --
    # `note` says why
    needs_review: bool = False
    note: Optional[str] = None
    job_id: Optional[int] = None
    # outcome == 'updated': days the replaced version stays restorable in
    # History, and days the other removed copies stay in Deleted (0 = kept
    # until removed by hand)
    retention_days: Optional[int] = None
    removed_retention_days: Optional[int] = None
    # outcome == 'restored': a Deleted document with identical content was
    # brought back -- under the uploaded name when renamed_from is set; and
    # the document it had been merged into, now kept separate from it
    renamed_from: Optional[str] = None
    never_merge_with: Optional[str] = None


class DocumentDeleteResponse(BaseModel):
    # Deleting is permanent (content and every version removed); only the
    # automatic version matching moves documents to Deleted.
    id: int
    filename: str
    chunks_deleted: int  # live + archived chunks removed for good
    deleted_at: Optional[datetime] = None
    permanent: bool = True
    was_in_deleted: bool = False  # deleted from the Deleted tab
    # kept for compatibility: always 0 / None now
    retention_days: int = 0
    restorable_until: Optional[datetime] = None


class DeletedDocumentItem(BaseModel):
    id: int
    filename: str
    source_type: str
    version: int
    chunk_count: int
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    reason: Optional[str] = None
    # set when the automatic matching merged it into another document
    merged_into_filename: Optional[str] = None
    restorable: bool = True
    permanent_delete_on: Optional[datetime] = None


class DeletedDocumentsResponse(BaseModel):
    retention_days: int
    items: list[DeletedDocumentItem] = Field(default_factory=list)


class DocumentRestoreRequest(BaseModel):
    # optional new name -- needed when an active document already has its name
    new_name: Optional[str] = Field(default=None, max_length=255)


class DocumentRestoreResponse(BaseModel):
    id: int
    filename: str
    version: int
    chunk_count: int
    renamed_from: Optional[str] = None
    never_merge_with: Optional[str] = None


class DocumentListItem(BaseModel):
    id: int
    filename: str
    source_type: str
    uploaded_by: Optional[str] = None
    uploaded_at: datetime
    chunk_count: int
    version: int = 1
    # 'Active' | 'Excluded' (kept in the Library, left out of the chat)
    status: str = "Active"
    # cloud documents: the file's path within the connected folder
    source_path: Optional[str] = None
    # when the content (or name) last changed, and by whom
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    # earlier versions still restorable from History
    previous_versions: int = 0


class DocumentExcludeResponse(BaseModel):
    id: int
    filename: str
    status: str  # 'Excluded' | 'Active'
    chunk_count: int  # chunks parked (exclude) or brought back (include)


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
    # 'added' | 'updated' | 'renamed' | 'unchanged' | 'linked' | 'review' |
    # 'excluded' | 'duplicate' | 'older_version' | 'removed' | 'skipped' |
    # 'failed' (see app/services/cloud_sync.py)
    status: str
    detail: Optional[str] = None
    document_id: Optional[int] = None


class SyncResponse(BaseModel):
    connection_id: int
    files_added: int
    files_duplicate: int  # identical content already in the knowledge base
    files_skipped: int
    files_failed: int
    files_updated: int = 0  # new versions (changed in the cloud, or of a local upload)
    files_renamed: int = 0
    files_unchanged: int = 0
    files_linked: int = 0  # a local upload that now follows the cloud file
    files_review: int = 0  # added, marked REVIEW
    files_excluded: int = 0
    files_older: int = 0  # older editions, not added
    files_removed: int = 0  # gone from the cloud folder -> permanently deleted
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


class ConnectStartResponse(BaseModel):
    # The connection is saved; its first sync runs in the background
    connection: SourceConnectionItem
    job_id: int


class SyncStartResponse(BaseModel):
    connection_id: int
    job_id: int


# --- Background jobs (app/services/jobs.py) ---------------------------------


class UploadQueuedResponse(BaseModel):
    job_id: int
    item_id: int
    filename: str
    status: str = "queued"


class JobItemOut(BaseModel):
    id: int
    filename: str
    source_path: Optional[str] = None
    # 'queued' | 'checking' | 'downloading' | 'extracting' | 'embedding' |
    # 'saving' | 'done'
    step: str
    progress_current: Optional[int] = None
    progress_total: Optional[int] = None
    outcome: Optional[str] = None
    message: Optional[str] = None
    # when done: uploads -> DocumentUploadResponse fields (or
    # {outcome:'failed'|'cancelled', detail}); syncs -> {status, path, detail}
    result: Optional[dict] = None


class JobOut(BaseModel):
    id: int
    job_type: str  # 'upload' | 'sync'
    # 'queued' | 'running' | 'cancel_requested' | 'completed' | 'partial' |
    # 'failed' | 'cancelled' | 'interrupted'
    status: str
    connection_id: Optional[int] = None
    total_items: int = 0
    processed_items: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0
    message: Optional[str] = None
    started_by: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    # syncs: how many files ended in each result status
    status_counts: Optional[dict] = None
    # files being worked on right now (while the job runs)
    current: list[JobItemOut] = Field(default_factory=list)
    items: list[JobItemOut] = Field(default_factory=list)


# --- Settings page (app/api/routes/settings.py) ------------------------------
# Values are in the units the page shows: percentages as whole numbers
# (50 = 50%), closeness as a percentage, words/days as integers, switches as
# true/false. bool is listed first in the Union so true/false stay booleans.
SettingValue = Union[bool, int, float]


class SettingItem(BaseModel):
    key: str
    label: str
    help: str
    kind: str  # 'percent' | 'closeness' | 'int' | 'bool'
    unit: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    value: SettingValue
    default_value: SettingValue
    is_default: bool
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    warn_below: Optional[float] = None
    warn_text: Optional[str] = None


class SettingSection(BaseModel):
    key: str
    title: str
    description: str
    items: list[SettingItem]


class SettingsResponse(BaseModel):
    sections: list[SettingSection]
    # soft warnings after a save (e.g. a value set into a risky range)
    warnings: list[str] = Field(default_factory=list)
    # False when the signed-in user may view but not change settings
    can_edit: bool = True


class SettingsUpdateRequest(BaseModel):
    values: dict[str, SettingValue]


class SettingsResetRequest(BaseModel):
    keys: Optional[list[str]] = None  # omitted/empty = reset everything


class SettingChangeItem(BaseModel):
    key: str
    label: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    changed_at: datetime
    changed_by: Optional[str] = None


# --- History panel (Document Library) --------------------------------------


class HistoryDocument(BaseModel):
    id: int
    filename: str
    status: str
    version: int
    source_type: str


class HistoryVersionItem(BaseModel):
    id: int
    version: int
    filename: str
    # 'Latest' | 'Previous version' | 'Deleted' | 'Permanently deleted'
    status: str
    chunk_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    stored_at: Optional[datetime] = None
    stored_by: Optional[str] = None
    status_changed_at: Optional[datetime] = None
    status_changed_by: Optional[str] = None
    # 'First upload' | 'Same file name' | 'Same name and wording' |
    # 'Same wording' | 'Restored' | 'Restored as separate document'
    how_added: Optional[str] = None
    match_score: Optional[float] = None  # wording similarity 0-1
    restored_from_version: Optional[int] = None
    note: Optional[str] = None
    restorable: bool = False
    permanent_delete_on: Optional[datetime] = None
    # pre-filled name for "Restore as separate document"
    suggested_separate_name: Optional[str] = None


class HistoryMergedCopy(BaseModel):
    id: int
    filename: str
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    permanent_delete_on: Optional[datetime] = None


class DocumentHistoryResponse(BaseModel):
    document: HistoryDocument
    previous_retention_days: int
    deleted_retention_days: int
    versions: list[HistoryVersionItem] = Field(default_factory=list)
    # older copies merged into this document by the automatic matching
    # (they're in the Deleted tab and can be restored from there)
    merged_copies: list[HistoryMergedCopy] = Field(default_factory=list)
    # documents marked "different documents, never merge" with this one
    never_merge_with: list[str] = Field(default_factory=list)


class VersionRestoreResponse(BaseModel):
    id: int
    filename: str
    restored_version: int
    replaced_version: int
    new_version: int
    chunk_count: int
    kept_current_name: bool = False
    message: str


class SeparateRestoreRequest(BaseModel):
    new_name: str = Field(..., max_length=255)


class SeparateRestoreResponse(BaseModel):
    id: int
    filename: str
    source_filename: str
    source_version: int
    chunk_count: int
    message: str


class DocumentRenameRequest(BaseModel):
    filename: str = Field(..., max_length=255)


class DocumentRenameResponse(BaseModel):
    id: int
    filename: str
    old_filename: str
    changed: bool
