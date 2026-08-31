"""
Orchestrates ingesting one uploaded file end-to-end: hash -> parse ->
chunk -> embed -> persist (spec §3.2). Always additive -- an ingestion
call only ever adds a new Document + its DocumentChunk rows, and never
touches any existing ones, duplicate or not (per your choice: duplicates
are allowed through, just flagged).
"""

import hashlib

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import Document, DocumentChunk, User
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts, serialize_embedding
from app.services.parsing import extract_text


class IngestionResult:
    def __init__(self, *, document: Document, chunk_count: int, is_duplicate: bool):
        self.document = document
        self.chunk_count = chunk_count
        self.is_duplicate = is_duplicate


def ingest_upload(
    *, db: Session, filename: str, content: bytes, uploaded_by: User
) -> IngestionResult:
    content_hash = hashlib.sha256(content).digest()

    # Flag only -- per spec §7 this is "dedupe by content hash or warn",
    # and you chose warn: an existing match doesn't stop ingestion.
    is_duplicate = (
        db.query(Document).filter(Document.ContentHash == content_hash).first()
        is not None
    )

    text = extract_text(filename, content)
    chunks = chunk_text(text)
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
        SourceType="upload",
        SourcePath=None,
        FileName=filename,
        UploadedBy=uploaded_by.Id,
        ContentHash=content_hash,
    )
    db.add(document)
    db.flush()  # assigns document.Id without ending the transaction

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        db.add(
            DocumentChunk(
                DocumentId=document.Id,
                ChunkIndex=index,
                ChunkText=chunk,
                Embedding=serialize_embedding(vector),
            )
        )

    db.commit()
    db.refresh(document)

    return IngestionResult(
        document=document, chunk_count=len(chunks), is_duplicate=is_duplicate
    )
