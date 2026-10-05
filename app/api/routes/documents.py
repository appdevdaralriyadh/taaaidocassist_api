"""
Document ingestion (upload) and the read-only Document Library listing --
spec §3.2, §3.6. One file per request (see the frontend's upload
component for why: independent per-file progress bars and one bad file
not blocking the rest of a drag-and-drop batch). Add/clear
knowledge-base actions are Phase 5.

Uploads are handled fully automatically (app/services/ingestion.py's
ingest_local_upload): identical content is skipped, a recognised new
version replaces the older one, an older edition of a stored document is
not added, and anything else is added (flagged for review when it looks
like a version but doesn't meet the rules).
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.config import settings
from app.db.models import Document, DocumentChunk, User
from app.db.session import get_db
from app.schemas import DocumentDeleteResponse, DocumentListItem, DocumentUploadResponse
from app.services.ingestion import delete_document as delete_document_service
from app.services.ingestion import ingest_local_upload

router = APIRouter()

_MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{file.filename}' is empty.",
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"'{file.filename}' exceeds the "
                f"{settings.MAX_UPLOAD_SIZE_MB}MB upload limit."
            ),
        )

    result = ingest_local_upload(
        db=db, filename=file.filename, content=content, uploaded_by=user
    )

    return DocumentUploadResponse(
        id=result.document.Id,
        filename=result.document.FileName,
        chunk_count=result.chunk_count,
        is_duplicate=result.outcome == "unchanged",
        outcome=result.outcome,
        version=result.version,
        previous_version=result.previous_version,
        replaced_filename=result.replaced_filename,
        replaced_at=result.replaced_at,
        replaced_by=result.replaced_by,
        matched_by=result.matched_by,
        similarity=result.similarity,
        similarity_max=result.similarity_max,
        removed_copies=result.removed_copies,
        removed_filenames=result.removed_filenames,
        matched_filename=result.matched_filename,
        matched_source_type=result.matched_source_type,
        upload_year=result.upload_year,
        existing_year=result.existing_year,
        needs_review=result.needs_review,
        note=result.note,
        job_id=result.job_id,
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Permanently removes one document and its chunks from the knowledge
    base (recorded in the upload history). Any account may delete any
    document -- same equal-permission model as Data Management.
    """
    result = delete_document_service(db, document_id=document_id, deleted_by=user)
    return DocumentDeleteResponse(
        id=result.document_id, filename=result.filename, chunks_deleted=result.chunks_deleted
    )


@router.get("", response_model=list[DocumentListItem])
def list_documents(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Chunk counts come from a subquery grouped only by DocumentId, then
    # LEFT JOINed back onto DarAI_Documents -- not a GROUP BY over every
    # selected Document column. SQL Server (unlike MySQL) requires every
    # non-aggregated selected column to appear in GROUP BY, so that
    # approach breaks the moment a new column (e.g. ContentHash) gets
    # added to Document; this structure has no such failure mode, since
    # the outer query isn't aggregating at all.
    chunk_counts = (
        db.query(
            DocumentChunk.DocumentId.label("document_id"),
            func.count(DocumentChunk.Id).label("chunk_count"),
        )
        .group_by(DocumentChunk.DocumentId)
        .subquery()
    )

    rows = (
        db.query(
            Document,
            User.DisplayName,
            func.coalesce(chunk_counts.c.chunk_count, 0).label("chunk_count"),
        )
        .join(User, User.Id == Document.UploadedBy)
        .outerjoin(chunk_counts, chunk_counts.c.document_id == Document.Id)
        .order_by(Document.UploadedAt.desc())
        .all()
    )

    return [
        DocumentListItem(
            id=doc.Id,
            filename=doc.FileName,
            source_type=doc.SourceType,
            uploaded_by=display_name,
            uploaded_at=doc.UploadedAt,
            chunk_count=chunk_count,
            version=doc.Version,
        )
        for doc, display_name, chunk_count in rows
    ]
