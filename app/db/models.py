"""
SQLAlchemy ORM models.

DarAI_Users / DarAI_ChatHistory map exactly to spec §4's DDL (Phase 1).
DarAI_Documents / DarAI_DocumentChunks (Phase 2) map to that same DDL plus
one additive column -- ContentHash on DarAI_Documents, added by
backend/sql/002_add_document_content_hash.sql -- used for duplicate-upload
detection. DarAI_SourceConnections (final phase) is a new table, added by
backend/sql/003_create_source_connections.sql, for OneDrive/GitHub
connectors. Nothing else deviates from the spec's schema.
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


class Document(Base):
    __tablename__ = "DarAI_Documents"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    SourceType = Column(Unicode(20), nullable=False)  # 'upload' | 'onedrive' | 'github'
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
    Embedding = Column(LargeBinary, nullable=False)  # VARBINARY(MAX): serialized float32 vector
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())

    document = relationship("Document", back_populates="chunks")


class SourceConnection(Base):
    __tablename__ = "DarAI_SourceConnections"

    Id = Column(Integer, primary_key=True, autoincrement=True)
    SourceType = Column(Unicode(20), nullable=False)  # 'onedrive' | 'github'
    DisplayLabel = Column(Unicode(255), nullable=False)
    # owner/repo/ref/path (github) or drive_id/item_id/path (onedrive) --
    # a JSON blob rather than more columns, since the two source types'
    # config shapes don't overlap.
    ConfigJson = Column(UnicodeText, nullable=False)
    # GitHub PAT only -- always NULL for onedrive (see
    # backend/sql/003_create_source_connections.sql for why).
    Secret = Column(UnicodeText, nullable=True)
    CreatedBy = Column(Integer, ForeignKey("DarAI_Users.Id"), nullable=False)
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())
    LastSyncedAt = Column(DateTime, nullable=True)
    LastSyncStatus = Column(Unicode(20), nullable=True)  # 'success' | 'partial' | 'error'
    LastSyncError = Column(UnicodeText, nullable=True)

    creator = relationship("User")
