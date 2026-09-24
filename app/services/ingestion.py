"""
Orchestrates ingesting one file end-to-end: hash -> parse -> chunk ->
embed -> persist (spec §3.2). Always additive -- an ingestion call only
ever adds a new Document + its DocumentChunk rows, and never touches any
existing ones, duplicate or not (per your choice: duplicates are allowed
through, just flagged).

Shared by local upload (app/api/routes/documents.py, source_type defaults
to "upload") and the OneDrive/Google Drive connectors
(app/api/routes/sources.py, which pass source_type="onedrive"/"googledrive"
and the file's path within that source) -- one pipeline, so a document
behaves identically (same chunking, same embedding, same dedupe-by-hash
flagging) no matter which source it came from.

Each chunk's Embedding is inserted via raw SQL, not the ORM -- see
app/db/models.py's docstring and app/services/embeddings.py's module
docstring for why (SQLAlchemy has no built-in VECTOR type).
"""

import hashlib
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document, User
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts, to_vector_literal
from app.services.parsing import extract_text


class IngestionResult:
    def __init__(self, *, document: Document, chunk_count: int, is_duplicate: bool):
        self.document = document
        self.chunk_count = chunk_count
        self.is_duplicate = is_duplicate


# The dimension in CAST(... AS VECTOR(n)) has to be a literal in the SQL
# text -- a type's size can't be a bind parameter in T-SQL -- so it's
# interpolated here from our own settings, never from user input, which
# is what makes that safe. Built once at import time since
# EMBEDDING_DIMENSIONS is fixed for the life of the process (same as
# every other setting in this app).
#
# CAST(:embedding AS NVARCHAR(MAX)) BEFORE casting to VECTOR, not
# straight to VECTOR: pyodbc/ODBC Driver 17 sends a long string parameter
# (the JSON-array embedding literal is ~20KB+) using a "long data" wire
# type that SQL Server reports as `ntext`, and VECTOR's CAST rules
# explicitly refuse ntext as a source type ("Explicit conversion from
# data type ntext to vector is not allowed") even though they accept
# nvarchar fine. ntext -> nvarchar(max) is always allowed, so this
# intermediate cast sidesteps the restriction without needing to change
# how pyodbc binds the parameter.
_INSERT_CHUNK_SQL = text(
    f"""
    INSERT INTO dbo.DarAI_DocumentChunks
        (DocumentId, ChunkIndex, ChunkText, Embedding, EmbeddingModel)
    VALUES
        (:document_id, :chunk_index, :chunk_text,
         CAST(CAST(:embedding AS NVARCHAR(MAX)) AS VECTOR({settings.EMBEDDING_DIMENSIONS})),
         :embedding_model)
    """
)


def ingest_upload(
    *,
    db: Session,
    filename: str,
    content: bytes,
    uploaded_by: User,
    source_type: str = "upload",
    source_path: Optional[str] = None,
) -> IngestionResult:
    content_hash = hashlib.sha256(content).digest()

    # Flag only -- per spec §7 this is "dedupe by content hash or warn",
    # and you chose warn: an existing match doesn't stop ingestion. Dedupe
    # is by content hash only, so the same file arriving via a different
    # source_type (e.g. uploaded locally, then also pulled from Google
    # Drive) still gets flagged.
    is_duplicate = (
        db.query(Document).filter(Document.ContentHash == content_hash).first()
        is not None
    )

    extracted_text = extract_text(filename, content)
    chunks = chunk_text(extracted_text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No extractable text found in '{filename}' "
                "(possibly a scanned/image-only file)."
            ),
        )

    vectors = embed_texts(chunks)

    document = Document(
        SourceType=source_type,
        SourcePath=source_path,
        FileName=filename,
        UploadedBy=uploaded_by.Id,
        ContentHash=content_hash,
    )
    db.add(document)
    db.flush()  # assigns document.Id without ending the transaction

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        db.execute(
            _INSERT_CHUNK_SQL,
            {
                "document_id": document.Id,
                "chunk_index": index,
                "chunk_text": chunk,
                "embedding": to_vector_literal(vector),
                # Stamped at ingestion time so retrieval.py can detect an
                # embedding-model switch later, even one that doesn't
                # change the vector dimension (see app/config.py's
                # comment on EMBEDDING_MODEL).
                "embedding_model": settings.EMBEDDING_MODEL,
            },
        )

    db.commit()
    db.refresh(document)

    return IngestionResult(
        document=document, chunk_count=len(chunks), is_duplicate=is_duplicate
    )
