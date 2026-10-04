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
rows. Nothing else deviates from the spec's schema.

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

    uploader = relationship("User")
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
