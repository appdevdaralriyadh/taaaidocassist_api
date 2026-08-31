"""
SQLAlchemy ORM models for the tables used in Phase 1 (Users, ChatHistory).

These map to tables that already exist in SQL Server per spec §4 — this
module does not create or migrate schema, it only maps to it.
DarAI_Documents / DarAI_DocumentChunks are added when Phase 2 (ingestion)
lands.
"""

import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
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
    # spec §4 — server_default here just documents that, it doesn't
    # duplicate it.
    CreatedAt = Column(DateTime, nullable=False, server_default=func.sysutcdatetime())

    user = relationship("User", back_populates="chat_messages")
