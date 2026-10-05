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

Deleting a document (DELETE /{id}) is permanent. Only the automatic
version matching moves documents to the Library's Deleted tab (GET
/deleted) -- older copies replaced by a newer upload -- where they can be
restored (POST /{id}/restore) or deleted for good, and are permanently
deleted automatically once the retention period on the Settings page ends.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.config import settings
from app.db.models import Document, DocumentChunk, User
from app.db.session import get_db
from app.schemas import (
    DeletedDocumentItem,
    DeletedDocumentsResponse,
    DocumentDeleteResponse,
    DocumentHistoryResponse,
    DocumentListItem,
    DocumentRenameRequest,
    DocumentRenameResponse,
    DocumentRestoreRequest,
    DocumentRestoreResponse,
    DocumentUploadResponse,
    SeparateRestoreRequest,
    SeparateRestoreResponse,
    VersionRestoreResponse,
)
from app.services import app_settings, doc_history, ingestion, retention
from app.services.ingestion import delete_document as delete_document_service
from app.services.ingestion import ingest_local_upload, restore_document

router = APIRouter()

# Permanently deletes items past their retention period, in the background
# (app/services/retention.py) -- started here so main.py needn't change.
retention.start_background_cleanup()

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
        retention_days=result.retention_days,
        removed_retention_days=result.removed_retention_days,
        renamed_from=result.renamed_from,
        never_merge_with=result.never_merge_with,
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Permanently deletes one document -- active, or in the Deleted tab --
    with all its versions. It cannot be restored (recorded in the upload
    history). Any account may delete any document -- same equal-permission
    model as Data Management.
    """
    result = delete_document_service(db, document_id=document_id, deleted_by=user)
    return DocumentDeleteResponse(
        id=result.document_id,
        filename=result.filename,
        chunks_deleted=result.chunks_deleted,
        deleted_at=result.deleted_at,
        permanent=True,
        was_in_deleted=result.was_in_deleted,
    )


@router.get("/deleted", response_model=DeletedDocumentsResponse)
def list_deleted_documents(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The Library's Deleted tab: documents that can still be restored."""
    retention.purge_if_due(db)  # so nothing past its date is listed
    retention_days = app_settings.get_effective(db)["DELETED_RETENTION_DAYS"]
    return DeletedDocumentsResponse(
        retention_days=retention_days,
        items=[
            DeletedDocumentItem(**item)
            for item in doc_history.deleted_documents(db, retention_days)
        ],
    )


@router.post("/{document_id}/restore", response_model=DocumentRestoreResponse)
def restore_deleted_document(
    document_id: int,
    payload: Optional[DocumentRestoreRequest] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Brings a document back from Deleted, optionally under a new name. 409
    with {message, suggested_name, name_conflict: true} when an active
    document already uses the name.
    """
    result = restore_document(
        db,
        document_id=document_id,
        restored_by=user,
        new_name=payload.new_name if payload else None,
    )
    return DocumentRestoreResponse(
        id=result.document.Id,
        filename=result.document.FileName,
        version=result.document.Version,
        chunk_count=result.chunk_count,
        renamed_from=result.renamed_from,
        never_merge_with=result.never_merge_with,
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
        # Active documents only -- Deleted ones are listed by GET /deleted
        .filter(Document.Status == doc_history.DOC_ACTIVE)
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


# --- History panel ---------------------------------------------------------
#
# 409 responses with {message, suggested_name, name_conflict: true} mean
# another active document already uses the requested name.


@router.get("/{document_id}/history", response_model=DocumentHistoryResponse)
def document_history(
    document_id: int,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Every version of one document, plus older copies merged into it."""
    return ingestion.document_history(db, document_id=document_id)


@router.post(
    "/{document_id}/versions/{version_id}/restore", response_model=VersionRestoreResponse
)
def restore_document_version(
    document_id: int,
    version_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Makes a previous version current again, as a new version."""
    result = ingestion.restore_version(
        db, document_id=document_id, version_id=version_id, restored_by=user
    )
    return VersionRestoreResponse(
        id=result["document"].Id,
        filename=result["document"].FileName,
        restored_version=result["restored_version"],
        replaced_version=result["replaced_version"],
        new_version=result["new_version"],
        chunk_count=result["chunk_count"],
        kept_current_name=result["kept_current_name"],
        message=result["message"],
    )


@router.post(
    "/{document_id}/versions/{version_id}/restore-separate",
    response_model=SeparateRestoreResponse,
)
def restore_document_version_as_separate(
    document_id: int,
    version_id: int,
    payload: SeparateRestoreRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Copies a previous version out as its own document under a new name."""
    result = ingestion.restore_version_as_separate(
        db,
        document_id=document_id,
        version_id=version_id,
        new_name=payload.new_name,
        restored_by=user,
    )
    return SeparateRestoreResponse(
        id=result["document"].Id,
        filename=result["document"].FileName,
        source_filename=result["source_filename"],
        source_version=result["source_version"],
        chunk_count=result["chunk_count"],
        message=result["message"],
    )


@router.patch("/{document_id}", response_model=DocumentRenameResponse)
def rename_document(
    document_id: int,
    payload: DocumentRenameRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Renames an active document."""
    result = ingestion.rename_document(
        db, document_id=document_id, new_name=payload.filename, renamed_by=user
    )
    return DocumentRenameResponse(
        id=result["document"].Id,
        filename=result["document"].FileName,
        old_filename=result["old_name"],
        changed=result["changed"],
    )
