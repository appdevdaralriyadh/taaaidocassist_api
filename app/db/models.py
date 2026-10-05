"""
SQLAlchemy ORM models.

DarAI_Users / DarAI_ChatHistory map exactly to spec §4's DDL (Phase 1).
DarAI_Documents / DarAI_DocumentChunks (Phase 2) map to that same DDL plus
two additive columns -- ContentHash on DarAI_Documents, and EmbeddingModel
on DarAI_DocumentChunks, used by app/services/retrieval.py to detect an
embedding-model switch even when the old and new models happen to produce
the same vector dimension (see the comment on that column below).
DarAI_SourceConnections (final phase) is a new table for OneDrive/Google
Drive connectors (originally OneDrive/GitHub -- GitHub was replaced by
Google Drive; the table/column shapes didn't need to change).

DarAI_Conversations is a later addition (sql/sqlserver2025/002_add_
conversations_table.sql) backing the Angular sidebar's conversation list,
rename, and delete features -- one row per conversation (Title,
LastMessageAt), separate from DarAI_ChatHistory's one-row-per-message
rows.

sql/sqlserver2025/003_document_versioning.sql adds document identity /
version columns to DarAI_Documents (ConnectionId, ExternalId,
SourceVersionTag, SourceModifiedAt, FileSizeBytes, Version, UpdatedAt,
UpdatedBy) and two new tables, DarAI_IngestionJobs and
DarAI_IngestionJobItems (one row per upload/sync run, and one per file in
it -- used for progress and as each document's version history). Nothing
else deviates from the spec's schema.

This app now targets a separate, new SQL Server 2025 database, created
from scratch by sql/sqlserver2025/001_create_schema.sql -- that single
script bakes in ContentHash and EmbeddingModel from the start, so there's
no incremental-patch lineage for this database the way
sql/002_add_document_content_hash.sql / sql/004_add_embedding_model_to_
chunks.sql were for the old SQL Server 2019 database. That 2019 database
and its 002/004 patches still exist but are no longer used by this app.

DocumentChunk.Embedding is deliberately NOT declared as a Column below.
It's a native VECTOR(EMBEDDING_DIMENSIONS) column on the real table
(app/config.py's EMBEDDING_DIMENSIONS), and SQLAlchemy has no built-in
type for VECTOR and no way to express VECTOR_DISTANCE() in the ORM query
builder -- so every read or write of that column goes through raw
parameterized SQL instead (see app/services/ingestion.py and
app/services/retrieval.py), bypassing this class entirely for that one
column. Nothing else in the codebase touches Embedding via the ORM
(confirmed by grep), so leaving it off this class is safe.
"""

import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    Unicode,
    UnicodeText,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "DarAI_Users"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    EntraObjectId = Column(Unicode(100), nullable=False, unique=True)
    DisplayName = Column(Unicode(100))

    chat_messages = relationship("ChatMessage", back_populates="user")


class ChatMessage(Base):
    __tablename__ = "DarAI_ChatHistory"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    UserId = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=False)
    ConversationId = Column(Uuid, nullable=False, default=uuid.uuid4)
    Role = Column(Unicode(10), nullable=False)  # 'user' | 'assistant'
    Message = Column(UnicodeText, nullable=False)
    UsedDocuments = Column(Boolean, nullable=False, default=False)
    # DB-side default (SYSUTCDATETIME()) already exists on this column per
    # spec §4 -- server_default here just documents that, it doesn't
    # duplicate it.
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())

    user = relationship("User", back_populates="chat_messages")


class Conversation(Base):
    """
    One row per conversation -- metadata only (title, activity time), not
    messages (those stay in ChatMessage/DarAI_ChatHistory, unchanged).

    Created by app/api/routes/chat.py's send_message() the first time a
    given ConversationId is used, and updated (LastMessageAt) on every
    later message. Listing/renaming/deleting a conversation is NOT
    restricted to the UserId that created it, matching this app's existing
    shared-visibility design for chat history (see get_history()'s
    comment and ClearChatHistoryRequest, which already let either account
    act on either account's history).
    """

    __tablename__ = "DarAI_Conversations"

    ConversationId = Column(Uuid, primary_key=True)
    UserId = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=False)
    # NULL until send_message() creates the row with a title derived from
    # the first user message. A conversation should never actually be
    # listed with a NULL title in practice, but the API falls back to a
    # placeholder display string rather than assume this can't happen.
    Title = Column(Unicode(200), nullable=True)
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())
    LastMessageAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())

    owner = relationship("User")


class Document(Base):
    __tablename__ = "DarAI_Documents"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    SourceType = Column(Unicode(20), nullable=False)  # 'upload' | 'onedrive' | 'googledrive'
    SourcePath = Column(Unicode(500))
    FileName = Column(Unicode(255), nullable=False)
    UploadedBy = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=False)
    UploadedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())
    # Phase 2 addition beyond spec §4's original DDL -- see
    # backend/sql/002_add_document_content_hash.sql. SHA-256 of the
    # uploaded file's raw bytes; used only to flag likely duplicates
    # (spec §7), never to block an upload outright.
    ContentHash = Column(LargeBinary(32), nullable=True)

    # --- 003_document_versioning.sql additions ---------------------------
    # OneDrive/Google Drive connection this document came from; NULL for
    # local uploads (and set to NULL by the DB if the connection is deleted).
    ConnectionId = Column(
        Integer, ForeignKey("DarAI_SourceConnections.Id", ondelete="SET NULL"), nullable=True
    )
    # The cloud file's own ID -- stable across renames/moves, so it's how a
    # sync recognises "the same file". NULL for local uploads.
    ExternalId = Column(Unicode(200), nullable=True)
    # Cheap change marker from the source (OneDrive cTag, Google md5Checksum
    # or version), compared on sync to skip downloading unchanged files.
    SourceVersionTag = Column(Unicode(200), nullable=True)
    SourceModifiedAt = Column(DateTime, nullable=True)
    FileSizeBytes = Column(BigInteger, nullable=True)
    # 1 for a document's first version; incremented each time newer content
    # replaces it (same Document row, same Id -- only the chunks change).
    Version = Column(Integer, nullable=False, server_default=text("1"))
    # When/by whom the CURRENT version was stored (UploadedAt/UploadedBy keep
    # meaning "first added"). UpdatedBy may be NULL on rows stored before
    # step 2 started setting it -- fall back to UploadedBy when displaying.
    UpdatedAt = Column(DateTime, nullable=True, server_default=func.sysutcdatetime())
    UpdatedBy = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=True)

    # Two foreign keys to DarAI_Users now exist on this table (UploadedBy,
    # UpdatedBy), so each relationship must say which one it follows --
    # without foreign_keys=..., SQLAlchemy refuses to configure the mapper.
    uploader = relationship("User", foreign_keys=[UploadedBy])
    updater = relationship("User", foreign_keys=[UpdatedBy])
    connection = relationship("SourceConnection")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "DarAI_DocumentChunks"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    DocumentId = Column(Integer, ForeignKey("DarAI_Documents.Id"), nullable=False)
    ChunkIndex = Column(Integer, nullable=False)
    ChunkText = Column(UnicodeText, nullable=False)
    # Embedding (VECTOR(EMBEDDING_DIMENSIONS), NOT NULL on the real table)
    # is intentionally not mapped here -- see the module docstring above.
    # Read/write it only via app/services/ingestion.py and
    # app/services/retrieval.py's raw SQL.
    #
    # Which EMBEDDING_MODEL (app/config.py) produced that Embedding.
    # Dimension alone can't always tell two models apart (BAAI/bge-m3 and
    # BAAI/bge-large-en-v1.5 both output 1024-dim vectors despite being
    # incompatible vector spaces), so app/services/retrieval.py checks
    # this directly instead of relying on shape alone. NULL means "unknown
    # model, exclude this chunk," never an assumed match.
    EmbeddingModel = Column(Unicode(200), nullable=True)
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())

    document = relationship("Document", back_populates="chunks")


class SourceConnection(Base):
    __tablename__ = "DarAI_SourceConnections"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    SourceType = Column(Unicode(20), nullable=False)  # 'onedrive' | 'googledrive'
    DisplayLabel = Column(Unicode(255), nullable=False)
    # drive_id/item_id/path (onedrive) or folder_id/path (googledrive) --
    # a JSON blob rather than more columns, since the two source types'
    # config shapes don't overlap.
    ConfigJson = Column(UnicodeText, nullable=False)
    # Always NULL for both current source types -- neither persists a
    # per-connection secret (onedrive re-acquires a Graph token from the
    # frontend's live MSAL session on every call; googledrive uses the
    # one shared service-account credential in app/config.py, not
    # anything tied to a specific connection). Column kept nullable
    # rather than dropped, in case a future source type needs it (this
    # column previously held GitHub's Personal Access Token before that
    # connector was replaced by Google Drive).
    Secret = Column(UnicodeText, nullable=True)
    CreatedBy = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=False)
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())
    LastSyncedAt = Column(DateTime, nullable=True)
    LastSyncStatus = Column(Unicode(20), nullable=True)  # 'success' | 'partial' | 'error'
    LastSyncError = Column(UnicodeText, nullable=True)

    creator = relationship("User")


class IngestionJob(Base):
    """
    One upload batch or sync run (003_document_versioning.sql). Holds the
    run's overall status and counters -- what the Upload/OneDrive/Google
    Drive pages and the Document Library show as progress and as "last
    checked / last updated" info.
    """

    __tablename__ = "DarAI_IngestionJobs"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    JobType = Column(Unicode(20), nullable=False)  # 'upload' | 'sync'
    ConnectionId = Column(
        Integer, ForeignKey("DarAI_SourceConnections.Id", ondelete="SET NULL"), nullable=True
    )
    # 'queued' | 'running' | 'completed' | 'partial' | 'failed' | 'interrupted'
    Status = Column(Unicode(20), nullable=False, server_default=text("'queued'"))
    TotalItems = Column(Integer, nullable=False, server_default=text("0"))
    ProcessedItems = Column(Integer, nullable=False, server_default=text("0"))
    AddedCount = Column(Integer, nullable=False, server_default=text("0"))
    UpdatedCount = Column(Integer, nullable=False, server_default=text("0"))
    UnchangedCount = Column(Integer, nullable=False, server_default=text("0"))
    RemovedCount = Column(Integer, nullable=False, server_default=text("0"))
    SkippedCount = Column(Integer, nullable=False, server_default=text("0"))
    FailedCount = Column(Integer, nullable=False, server_default=text("0"))
    Message = Column(Unicode(1000), nullable=True)
    StartedBy = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=False)
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())
    StartedAt = Column(DateTime, nullable=True)
    FinishedAt = Column(DateTime, nullable=True)

    starter = relationship("User")
    connection = relationship("SourceConnection")
    items = relationship(
        "IngestionJobItem", back_populates="job", cascade="all, delete-orphan"
    )


class IngestionJobItem(Base):
    """
    One file within an IngestionJob: its live step/progress while the job
    runs, then its outcome. Also serves as each document's version history
    (all items with a given DocumentId, newest first).

    DocumentId / RelatedDocumentId are plain integers, not foreign keys, on
    purpose: history has to survive the document being replaced or deleted
    (see 003_document_versioning.sql).
    """

    __tablename__ = "DarAI_IngestionJobItems"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    JobId = Column(
        Integer, ForeignKey("DarAI_IngestionJobs.Id", ondelete="CASCADE"), nullable=False
    )
    FileName = Column(Unicode(255), nullable=False)
    SourcePath = Column(Unicode(500), nullable=True)
    ExternalId = Column(Unicode(200), nullable=True)
    # 'queued' | 'downloading' | 'extracting' | 'embedding' | 'saving' | 'done'
    Step = Column(Unicode(30), nullable=False, server_default=text("'queued'"))
    ProgressCurrent = Column(Integer, nullable=True)
    ProgressTotal = Column(Integer, nullable=True)
    # 'new' | 'updated' | 'unchanged' | 'possible_version' | 'renamed'
    # | 'removed' | 'skipped' | 'failed'
    Outcome = Column(Unicode(30), nullable=True)
    DocumentId = Column(Integer, nullable=True)
    RelatedDocumentId = Column(Integer, nullable=True)
    PreviousVersion = Column(Integer, nullable=True)
    NewVersion = Column(Integer, nullable=True)
    ContentHash = Column(LargeBinary(32), nullable=True)
    FileSizeBytes = Column(BigInteger, nullable=True)
    ChunkCount = Column(Integer, nullable=True)
    Message = Column(Unicode(1000), nullable=True)
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())
    UpdatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())

    job = relationship("IngestionJob", back_populates="items")
