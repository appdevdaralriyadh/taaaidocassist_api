"""
Version history, soft delete and restore (004_versions_archive_settings.sql).

Nothing here ever removes a document row or re-embeds anything:

  * archive_current()   moves a document's live chunks (DarAI_DocumentChunks,
                        what the chat searches) into DarAI_DocumentChunkArchive
                        (never read by the chat), marking its Latest version
                        row 'Previous version' or 'Deleted'.
  * record_new_latest() adds the version row for content just stored.
  * soft_delete()       archives the content and flags the document 'Deleted'.
  * restore_deleted()   moves archived chunks back and flags it 'Active'.
  * restore_version()   makes an earlier version current again, as a new version.
  * restore_version_as_separate()  copies a version out as its own document
                        (new name, marked "never merge" with this one).
  * rename_document()   renames an active document.
  * document_history()  what the Library's History panel shows.

Moving chunks is done with INSERT ... SELECT inside SQL Server, so the
VECTOR embeddings are copied as-is and restoring is instant.

Plain-word status values, used both here and in the database:
  document: 'Active' | 'Deleted' | 'Permanently deleted'
  version:  'Latest' | 'Previous version' | 'Deleted' | 'Permanently deleted'
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from app.db.models import (
    Document,
    DocumentChunk,
    DocumentChunkArchive,
    DocumentMatchExclusion,
    DocumentVersion,
    User,
)

DOC_ACTIVE = "Active"
DOC_DELETED = "Deleted"
DOC_PERMANENTLY_DELETED = "Permanently deleted"

LATEST = "Latest"
PREVIOUS = "Previous version"
DELETED = "Deleted"
PERMANENTLY_DELETED = "Permanently deleted"

# matched_by (ingestion.py) -> how the version row describes it
HOW_ADDED = {
    "filename": "Same file name",
    "name_and_content": "Same name and wording",
    "content": "Same wording",
}

_NOT_RECOVERABLE = "Replaced before version history was kept -- this version cannot be restored."

# Module-level so tests can swap them for a database without VECTOR support.
ARCHIVE_CHUNKS_SQL = text(
    """
    INSERT INTO dbo.DarAI_DocumentChunkArchive
        (VersionId, DocumentId, ChunkIndex, ChunkText, Embedding, EmbeddingModel, ChunkCreatedAt)
    SELECT :version_id, c.DocumentId, c.ChunkIndex, c.ChunkText, c.Embedding,
           c.EmbeddingModel, c.CreatedAt
    FROM dbo.DarAI_DocumentChunks c
    WHERE c.DocumentId = :document_id
    """
)
DELETE_LIVE_CHUNKS_SQL = text("DELETE FROM dbo.DarAI_DocumentChunks WHERE DocumentId = :document_id")
RESTORE_CHUNKS_SQL = text(
    """
    INSERT INTO dbo.DarAI_DocumentChunks
        (DocumentId, ChunkIndex, ChunkText, Embedding, EmbeddingModel)
    SELECT :document_id, a.ChunkIndex, a.ChunkText, a.Embedding, a.EmbeddingModel
    FROM dbo.DarAI_DocumentChunkArchive a
    WHERE a.VersionId = :version_id
    """
)
DELETE_ARCHIVED_CHUNKS_SQL = text(
    "DELETE FROM dbo.DarAI_DocumentChunkArchive WHERE VersionId = :version_id"
)


def live_chunk_count(db: Session, document_id: int) -> int:
    return (
        db.query(func.count(DocumentChunk.Id)).filter(DocumentChunk.DocumentId == document_id).scalar()
        or 0
    )


def ensure_latest_version(db: Session, doc: Document) -> DocumentVersion:
    """
    The document's 'Latest' version row, matching its current Version --
    created if missing. Repairs rows left stale by content that was
    replaced before history was kept (e.g. uploads between running
    004_versions_archive_settings.sql and installing this step): such a row
    is marked 'Permanently deleted', since its chunks are already gone.
    """
    rows = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.DocumentId == doc.Id, DocumentVersion.Status == LATEST)
        .order_by(DocumentVersion.Id.desc())
        .all()
    )
    current = None
    for row in rows:
        if current is None and row.VersionNumber == doc.Version:
            current = row
    if current is None:
        # Same content under a stale number (shouldn't happen, but cheap to
        # handle): renumber it rather than losing it.
        for row in rows:
            if row.ContentHash is not None and row.ContentHash == doc.ContentHash:
                row.VersionNumber = doc.Version
                row.FileName = doc.FileName
                current = row
                break
    for row in rows:
        if row is not current:
            # Its content was replaced by the old (pre-history) code, which
            # deleted the chunks -- so it can't be restored.
            row.Status = PERMANENTLY_DELETED
            row.StatusChangedAt = doc.UpdatedAt or func.sysutcdatetime()
            row.Note = _NOT_RECOVERABLE
    if current is None:
        current = DocumentVersion(
            DocumentId=doc.Id,
            VersionNumber=doc.Version,
            FileName=doc.FileName,
            ContentHash=doc.ContentHash,
            FileSizeBytes=doc.FileSizeBytes,
            ChunkCount=live_chunk_count(db, doc.Id),
            Status=LATEST,
            StoredAt=doc.UpdatedAt or doc.UploadedAt or datetime.utcnow(),
            StoredBy=doc.UpdatedBy or doc.UploadedBy,
            HowAdded="First upload" if doc.Version == 1 else None,
            Note=(
                None
                if doc.Version == 1
                else "Uploaded before version history was kept, so how it was matched isn't recorded."
            ),
        )
        db.add(current)
    db.flush()
    return current


def archive_current(
    db: Session, doc: Document, *, new_status: str, by_user_id: Optional[int], note: Optional[str] = None
) -> DocumentVersion:
    """
    Moves the document's live chunks to the archive and marks its Latest
    version row `new_status` ('Previous version' or 'Deleted'). The chat
    stops seeing that content immediately. Call BEFORE changing
    doc.Version for a replacement. Does not commit.
    """
    version = ensure_latest_version(db, doc)
    params = {"version_id": version.Id, "document_id": doc.Id}
    db.execute(ARCHIVE_CHUNKS_SQL, params)
    db.execute(DELETE_LIVE_CHUNKS_SQL, params)
    version.Status = new_status
    version.StatusChangedAt = func.sysutcdatetime()
    version.StatusChangedBy = by_user_id
    if note:
        version.Note = note[:1000]
    db.flush()
    return version


def record_new_latest(
    db: Session,
    doc: Document,
    *,
    chunk_count: int,
    how_added: str,
    by_user_id: Optional[int],
    score: Optional[float] = None,
    note: Optional[str] = None,
    restored_from_version_id: Optional[int] = None,
) -> DocumentVersion:
    """Adds the 'Latest' version row for content just stored. Does not commit."""
    version = DocumentVersion(
        DocumentId=doc.Id,
        VersionNumber=doc.Version,
        FileName=doc.FileName,
        ContentHash=doc.ContentHash,
        FileSizeBytes=doc.FileSizeBytes,
        ChunkCount=chunk_count,
        Status=LATEST,
        StoredBy=by_user_id,
        StatusChangedBy=by_user_id,
        HowAdded=how_added,
        MatchScore=round(score, 4) if score is not None else None,
        RestoredFromVersionId=restored_from_version_id,
        Note=note[:1000] if note else None,
    )
    db.add(version)
    db.flush()
    return version


def soft_delete(
    db: Session,
    doc: Document,
    *,
    by_user_id: Optional[int],
    reason: str,
    merged_into_document_id: Optional[int] = None,
) -> DocumentVersion:
    """
    Flags the document 'Deleted' and archives its content -- the row stays,
    and it can be restored until the retention period ends. Does not commit.
    """
    if doc.Status != DOC_ACTIVE:
        raise HTTPException(status_code=409, detail=f"'{doc.FileName}' is already {doc.Status.lower()}.")
    version = archive_current(db, doc, new_status=DELETED, by_user_id=by_user_id)
    doc.Status = DOC_DELETED
    doc.DeletedAt = func.sysutcdatetime()
    doc.DeletedBy = by_user_id
    doc.DeletedReason = reason[:500]
    doc.MergedIntoDocumentId = merged_into_document_id
    db.flush()
    return version


def name_in_use(db: Session, filename: str, *, except_document_id: Optional[int] = None) -> Optional[Document]:
    """An ACTIVE document already using this exact filename (case-insensitive), if any."""
    query = db.query(Document).filter(
        Document.Status == DOC_ACTIVE, func.lower(Document.FileName) == filename.lower()
    )
    if except_document_id is not None:
        query = query.filter(Document.Id != except_document_id)
    return query.first()


def suggest_free_name(db: Session, filename: str) -> str:
    """'Policy.pdf' -> 'Policy (restored).pdf' -> 'Policy (restored 2).pdf' ..."""
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem, ext = filename, ""
    for n in range(1, 100):
        label = "restored" if n == 1 else f"restored {n}"
        candidate = f"{stem} ({label}){dot}{ext}"
        if name_in_use(db, candidate) is None:
            return candidate
    return f"{stem} (restored {datetime.utcnow():%Y%m%d%H%M%S}){dot}{ext}"


def add_exclusion(
    db: Session, doc_a: int, doc_b: int, *, by_user_id: Optional[int], reason: str
) -> None:
    """Marks two documents "different documents, never merge" (idempotent)."""
    a, b = sorted((doc_a, doc_b))
    if a == b:
        return
    exists = (
        db.query(DocumentMatchExclusion)
        .filter(DocumentMatchExclusion.DocumentIdA == a, DocumentMatchExclusion.DocumentIdB == b)
        .first()
    )
    if exists is None:
        db.add(DocumentMatchExclusion(DocumentIdA=a, DocumentIdB=b, Reason=reason[:500], CreatedBy=by_user_id))
        db.flush()


def excluded_ids(db: Session, document_id: int) -> set:
    """Documents marked "never merge" with this one."""
    rows = (
        db.query(DocumentMatchExclusion)
        .filter(
            or_(
                DocumentMatchExclusion.DocumentIdA == document_id,
                DocumentMatchExclusion.DocumentIdB == document_id,
            )
        )
        .all()
    )
    return {r.DocumentIdB if r.DocumentIdA == document_id else r.DocumentIdA for r in rows}


def latest_deleted_version(db: Session, doc: Document) -> Optional[DocumentVersion]:
    return (
        db.query(DocumentVersion)
        .filter(DocumentVersion.DocumentId == doc.Id, DocumentVersion.Status == DELETED)
        .order_by(DocumentVersion.VersionNumber.desc(), DocumentVersion.Id.desc())
        .first()
    )


def restore_deleted(
    db: Session, doc: Document, *, by_user_id: Optional[int], new_name: Optional[str] = None
) -> dict:
    """
    Brings a Deleted document back: its archived chunks return to the live
    table (no re-embedding), its version row becomes 'Latest' again, and
    it's flagged 'Active'. If it had been merged into another document by
    the automatic matching, the two are marked "never merge" so it won't
    happen again. Raises 409 with a suggested name if another active
    document already uses its name. Does not commit.
    """
    if doc.Status != DOC_DELETED:
        raise HTTPException(status_code=409, detail=f"'{doc.FileName}' is not in Deleted.")
    version = latest_deleted_version(db, doc)
    if version is None:
        raise HTTPException(
            status_code=409, detail=f"'{doc.FileName}' has no restorable content left."
        )

    target_name = (new_name or "").strip() or doc.FileName
    if name_in_use(db, target_name, except_document_id=doc.Id) is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    f"Another document is already called '{target_name}'. Restore it under a "
                    "different name -- two documents with the same name would be treated as "
                    "versions of each other by the next upload."
                ),
                "suggested_name": suggest_free_name(db, target_name),
                "name_conflict": True,
            },
        )

    params = {"version_id": version.Id, "document_id": doc.Id}
    db.execute(RESTORE_CHUNKS_SQL, params)
    db.execute(DELETE_ARCHIVED_CHUNKS_SQL, params)

    renamed_from = doc.FileName if target_name != doc.FileName else None
    version.Status = LATEST
    version.StatusChangedAt = func.sysutcdatetime()
    version.StatusChangedBy = by_user_id
    if renamed_from:
        version.Note = f"Restored as '{target_name}' (was '{renamed_from}')."
    doc.FileName = target_name
    doc.Version = version.VersionNumber
    doc.Status = DOC_ACTIVE

    never_merge_with = None
    if doc.MergedIntoDocumentId:
        other = db.get(Document, doc.MergedIntoDocumentId)
        add_exclusion(
            db,
            doc.Id,
            doc.MergedIntoDocumentId,
            by_user_id=by_user_id,
            reason="Restored after being merged automatically -- treated as a different document.",
        )
        never_merge_with = other.FileName if other is not None else None

    doc.DeletedAt = None
    doc.DeletedBy = None
    doc.DeletedReason = None
    doc.MergedIntoDocumentId = None
    db.flush()
    return {
        "document": doc,
        "version": version,
        "chunk_count": live_chunk_count(db, doc.Id),
        "renamed_from": renamed_from,
        "never_merge_with": never_merge_with,
    }


def find_in_history_by_hash(db: Session, content_hash: bytes):
    """
    Where identical content sits outside the active documents, if anywhere:
      ('deleted',  doc, version) -- a Deleted document's content
      ('previous', doc, version) -- an older version of an active document
      None
    """
    deleted = (
        db.query(DocumentVersion, Document)
        .join(Document, Document.Id == DocumentVersion.DocumentId)
        .filter(
            DocumentVersion.ContentHash == content_hash,
            DocumentVersion.Status == DELETED,
            Document.Status == DOC_DELETED,
        )
        .order_by(DocumentVersion.Id.desc())
        .first()
    )
    if deleted is not None:
        version, doc = deleted
        return "deleted", doc, version
    previous = (
        db.query(DocumentVersion, Document)
        .join(Document, Document.Id == DocumentVersion.DocumentId)
        .filter(
            DocumentVersion.ContentHash == content_hash,
            # Only versions still held: one whose content is already gone
            # can't be offered back, so such an upload is handled normally.
            DocumentVersion.Status == PREVIOUS,
            Document.Status == DOC_ACTIVE,
        )
        .order_by(DocumentVersion.Id.desc())
        .first()
    )
    if previous is not None:
        version, doc = previous
        return "previous", doc, version
    return None


def permanent_delete_on(deleted_at: Optional[datetime], days: int) -> Optional[datetime]:
    """When an item will be permanently deleted (None when kept forever)."""
    if deleted_at is None or not days:
        return None
    return deleted_at + timedelta(days=days)


def deleted_documents(db: Session, retention_days: int) -> list[dict]:
    """The Deleted tab: every document currently in Deleted, newest first."""
    rows = (
        db.query(Document)
        .filter(Document.Status == DOC_DELETED)
        .order_by(Document.DeletedAt.desc(), Document.Id.desc())
        .all()
    )
    user_ids = {d.DeletedBy for d in rows if d.DeletedBy}
    names = (
        {u.Id: u.DisplayName for u in db.query(User).filter(User.Id.in_(list(user_ids)))}
        if user_ids
        else {}
    )
    merged_ids = {d.MergedIntoDocumentId for d in rows if d.MergedIntoDocumentId}
    merged_names = (
        {d.Id: d.FileName for d in db.query(Document).filter(Document.Id.in_(list(merged_ids)))}
        if merged_ids
        else {}
    )
    out = []
    for d in rows:
        version = latest_deleted_version(db, d)
        out.append(
            {
                "id": d.Id,
                "filename": d.FileName,
                "source_type": d.SourceType,
                "version": d.Version,
                "chunk_count": (version.ChunkCount if version and version.ChunkCount is not None else 0),
                "deleted_at": d.DeletedAt,
                "deleted_by": names.get(d.DeletedBy),
                "reason": d.DeletedReason,
                "merged_into_filename": merged_names.get(d.MergedIntoDocumentId),
                "restorable": version is not None,
                "permanent_delete_on": permanent_delete_on(d.DeletedAt, retention_days),
            }
        )
    return out


# ---------------------------------------------------------------------------
# History panel: list versions, restore a version, restore as a separate
# document, rename
# ---------------------------------------------------------------------------

HOW_RESTORED = "Restored"
HOW_RESTORED_SEPARATE = "Restored as separate document"

_INVALID_NAME_CHARS = set('\\/:*?"<>|')
_DOC_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".xls", ".pptx"}


def clean_new_name(new_name: Optional[str], current_name: str) -> str:
    """
    Validates a name typed by the user. Keeps the current file extension
    when the new name has none (so 'Gift Policy' stays 'Gift Policy.pdf').
    Raises 422 with a plain message when it can't be used.
    """
    name = " ".join((new_name or "").split())
    if not name:
        raise HTTPException(status_code=422, detail="Enter a name.")
    if any(ch in _INVALID_NAME_CHARS for ch in name):
        raise HTTPException(
            status_code=422, detail='A name can\'t contain any of these characters: \\ / : * ? " < > |'
        )
    current_ext = os.path.splitext(current_name)[1]
    if current_ext and os.path.splitext(name)[1].lower() not in _DOC_EXTENSIONS:
        name = f"{name}{current_ext}"
    if len(name) > 255:
        raise HTTPException(status_code=422, detail="That name is too long (255 characters at most).")
    return name


def _name_conflict(db: Session, name: str, except_document_id: Optional[int]) -> None:
    if name_in_use(db, name, except_document_id=except_document_id) is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    f"Another document is already called '{name}'. Choose a different name -- two "
                    "documents with the same name would be treated as versions of each other."
                ),
                "suggested_name": suggest_free_name(db, name),
                "name_conflict": True,
            },
        )


def archived_chunk_count(db: Session, version_id: int) -> int:
    return (
        db.query(func.count(DocumentChunkArchive.Id))
        .filter(DocumentChunkArchive.VersionId == version_id)
        .scalar()
        or 0
    )


def _restorable_version(db: Session, doc: Document, version_id: int) -> DocumentVersion:
    if doc.Status != DOC_ACTIVE:
        raise HTTPException(
            status_code=409,
            detail=f"'{doc.FileName}' is in Deleted -- restore the document first.",
        )
    version = db.get(DocumentVersion, version_id)
    if version is None or version.DocumentId != doc.Id:
        raise HTTPException(status_code=404, detail="Version not found.")
    if version.Status != PREVIOUS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"v{version.VersionNumber} is '{version.Status}' -- only a previous version can be "
                "restored."
            ),
        )
    if archived_chunk_count(db, version.Id) == 0:
        raise HTTPException(
            status_code=409, detail=f"v{version.VersionNumber} has no stored content to restore."
        )
    return version


def restore_version(db: Session, doc: Document, version_id: int, *, by_user_id: Optional[int]) -> dict:
    """
    Makes an earlier version current again -- as a NEW version (e.g. v1's
    content comes back as v4), so nothing in the history is lost: the
    content being replaced becomes a 'Previous version' too, and v1 stays
    in the list. Its chunks are copied from the archive (no re-embedding).
    The document takes that version's file name back, unless another
    document now uses it. Does not commit.
    """
    version = _restorable_version(db, doc, version_id)
    old_number = version.VersionNumber
    current_number = doc.Version
    new_number = current_number + 1

    archive_current(
        db,
        doc,
        new_status=PREVIOUS,
        by_user_id=by_user_id,
        note=f"Replaced when v{old_number} was restored (as v{new_number}).",
    )
    db.execute(RESTORE_CHUNKS_SQL, {"version_id": version.Id, "document_id": doc.Id})

    name = version.FileName
    kept_current_name = False
    if name.lower() != doc.FileName.lower() and name_in_use(db, name, except_document_id=doc.Id):
        name = doc.FileName
        kept_current_name = True

    doc.FileName = name
    doc.ContentHash = version.ContentHash
    doc.FileSizeBytes = version.FileSizeBytes
    doc.Version = new_number
    doc.UpdatedAt = func.sysutcdatetime()
    doc.UpdatedBy = by_user_id
    db.flush()

    chunk_count = live_chunk_count(db, doc.Id)
    record_new_latest(
        db,
        doc,
        chunk_count=chunk_count,
        how_added=HOW_RESTORED,
        by_user_id=by_user_id,
        note=f"Restored from v{old_number}.",
        restored_from_version_id=version.Id,
    )
    return {
        "document": doc,
        "restored_version": old_number,
        "replaced_version": current_number,
        "new_version": new_number,
        "chunk_count": chunk_count,
        "kept_current_name": kept_current_name,
    }


def restore_version_as_separate(
    db: Session, doc: Document, version_id: int, *, new_name: str, by_user_id: Optional[int]
) -> dict:
    """
    For a version that was really a different document (merged in by the
    automatic matching): copies it out as a new, separate document under a
    new name, and marks the two "never merge". The version also stays in
    this document's history. Does not commit.
    """
    version = _restorable_version(db, doc, version_id)
    name = clean_new_name(new_name, version.FileName)
    _name_conflict(db, name, None)

    separate = Document(
        SourceType=doc.SourceType,
        SourcePath=None,
        FileName=name,
        UploadedBy=by_user_id or doc.UploadedBy,
        ContentHash=version.ContentHash,
        FileSizeBytes=version.FileSizeBytes,
        Version=1,
        UpdatedBy=by_user_id,
        Status=DOC_ACTIVE,
    )
    db.add(separate)
    db.flush()
    db.execute(RESTORE_CHUNKS_SQL, {"version_id": version.Id, "document_id": separate.Id})
    chunk_count = live_chunk_count(db, separate.Id)
    record_new_latest(
        db,
        separate,
        chunk_count=chunk_count,
        how_added=HOW_RESTORED_SEPARATE,
        by_user_id=by_user_id,
        note=f"From v{version.VersionNumber} of '{doc.FileName}'.",
        restored_from_version_id=version.Id,
    )
    version.Note = (f"Also restored as a separate document: '{name}'." + (
        f" {version.Note}" if version.Note else ""
    ))[:1000]
    add_exclusion(
        db,
        doc.Id,
        separate.Id,
        by_user_id=by_user_id,
        reason=f"'{name}' was restored from v{version.VersionNumber} as a separate document.",
    )
    db.flush()
    return {"document": separate, "source_version": version.VersionNumber, "chunk_count": chunk_count}


def rename_document(db: Session, doc: Document, new_name: str, *, by_user_id: Optional[int]) -> dict:
    """Renames an active document (409 with a suggestion if the name is taken). Does not commit."""
    if doc.Status != DOC_ACTIVE:
        raise HTTPException(status_code=409, detail=f"'{doc.FileName}' is in Deleted -- restore it first.")
    name = clean_new_name(new_name, doc.FileName)
    old_name = doc.FileName
    if name == old_name:
        return {"document": doc, "old_name": old_name, "changed": False}
    _name_conflict(db, name, doc.Id)
    doc.FileName = name
    doc.UpdatedAt = func.sysutcdatetime()
    doc.UpdatedBy = by_user_id
    db.flush()
    return {"document": doc, "old_name": old_name, "changed": True}


def _user_names(db: Session, ids) -> dict:
    ids = [i for i in set(ids) if i]
    if not ids:
        return {}
    return {u.Id: u.DisplayName for u in db.query(User).filter(User.Id.in_(ids))}


def document_history(db: Session, doc: Document, *, previous_days: int, deleted_days: int) -> dict:
    """
    Everything the History panel shows for one document. For an active
    document, first makes sure its version rows match its current content
    (ensure_latest_version) -- documents updated between running
    004_versions_archive_settings.sql and installing the history code
    otherwise show a stale 'Latest' row and no row for what's really
    current. The caller commits that repair.
    """
    if doc.Status == DOC_ACTIVE:
        ensure_latest_version(db, doc)
    versions = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.DocumentId == doc.Id)
        .order_by(DocumentVersion.VersionNumber.desc(), DocumentVersion.Id.desc())
        .all()
    )
    merged = (
        db.query(Document)
        .filter(Document.MergedIntoDocumentId == doc.Id, Document.Status == DOC_DELETED)
        .order_by(Document.DeletedAt.desc())
        .all()
    )
    names = _user_names(
        db,
        [v.StoredBy for v in versions] + [v.StatusChangedBy for v in versions] + [m.DeletedBy for m in merged],
    )
    by_id = {v.Id: v.VersionNumber for v in versions}
    can_act = doc.Status == DOC_ACTIVE

    version_items = []
    for v in versions:
        restorable = can_act and v.Status == PREVIOUS and archived_chunk_count(db, v.Id) > 0
        if v.Status == PREVIOUS:
            until = permanent_delete_on(v.StatusChangedAt, previous_days)
        elif v.Status == DELETED:
            until = permanent_delete_on(v.StatusChangedAt, deleted_days)
        else:
            until = None
        version_items.append(
            {
                "id": v.Id,
                "version": v.VersionNumber,
                "filename": v.FileName,
                "status": v.Status,
                "chunk_count": v.ChunkCount,
                "file_size_bytes": v.FileSizeBytes,
                "stored_at": v.StoredAt,
                "stored_by": names.get(v.StoredBy),
                # unknown for versions replaced before history was kept
                "status_changed_at": None if v.Note == _NOT_RECOVERABLE else v.StatusChangedAt,
                "status_changed_by": names.get(v.StatusChangedBy),
                "how_added": v.HowAdded,
                "match_score": float(v.MatchScore) if v.MatchScore is not None else None,
                "restored_from_version": by_id.get(v.RestoredFromVersionId),
                "note": v.Note,
                "restorable": restorable,
                "permanent_delete_on": until,
                "suggested_separate_name": (
                    suggest_free_name(db, v.FileName)
                    if restorable and name_in_use(db, v.FileName) is not None
                    else v.FileName
                )
                if restorable
                else None,
            }
        )

    never_merge = excluded_ids(db, doc.Id)
    never_merge_names = (
        [d.FileName for d in db.query(Document).filter(Document.Id.in_(list(never_merge)))]
        if never_merge
        else []
    )
    return {
        "document": {
            "id": doc.Id,
            "filename": doc.FileName,
            "status": doc.Status,
            "version": doc.Version,
            "source_type": doc.SourceType,
        },
        "previous_retention_days": previous_days,
        "deleted_retention_days": deleted_days,
        "versions": version_items,
        "merged_copies": [
            {
                "id": m.Id,
                "filename": m.FileName,
                "deleted_at": m.DeletedAt,
                "deleted_by": names.get(m.DeletedBy),
                "permanent_delete_on": permanent_delete_on(m.DeletedAt, deleted_days),
            }
            for m in merged
        ],
        "never_merge_with": sorted(never_merge_names),
    }


# ---------------------------------------------------------------------------
# Permanent delete: removes the content for good (live chunks and every
# archived version). The document row stays as a 'Permanently deleted'
# record -- name, dates, who -- so the upload history still makes sense,
# but it's hidden everywhere and can never be restored or matched again.
# ---------------------------------------------------------------------------


def purge_version(db: Session, version: DocumentVersion, *, by_user_id: Optional[int], note: str) -> int:
    """Deletes one version's archived content for good. Returns chunks removed. Does not commit."""
    removed = archived_chunk_count(db, version.Id)
    db.execute(DELETE_ARCHIVED_CHUNKS_SQL, {"version_id": version.Id})
    version.Status = PERMANENTLY_DELETED
    version.StatusChangedAt = func.sysutcdatetime()
    version.StatusChangedBy = by_user_id
    version.Note = note[:1000]
    db.flush()
    return removed


def purge_document(db: Session, doc: Document, *, by_user_id: Optional[int], reason: str) -> int:
    """
    Permanently deletes a document -- Active or in Deleted -- with all of
    its versions. Returns the number of chunks removed (live + archived).
    Does not commit.
    """
    if doc.Status == DOC_PERMANENTLY_DELETED:
        raise HTTPException(status_code=404, detail="Document not found.")
    removed = live_chunk_count(db, doc.Id)
    db.execute(DELETE_LIVE_CHUNKS_SQL, {"document_id": doc.Id})
    versions = db.query(DocumentVersion).filter(DocumentVersion.DocumentId == doc.Id).all()
    for v in versions:
        if v.Status != PERMANENTLY_DELETED:
            removed += purge_version(db, v, by_user_id=by_user_id, note=reason)
    doc.Status = DOC_PERMANENTLY_DELETED
    if doc.DeletedAt is None:
        doc.DeletedAt = func.sysutcdatetime()
    doc.DeletedBy = by_user_id
    doc.DeletedReason = reason[:500]
    db.flush()
    return removed


def expired_items(db: Session, *, deleted_days: int, previous_days: int, now: Optional[datetime] = None):
    """
    What the retention clean-up removes now: documents in Deleted for more
    than `deleted_days`, and previous versions replaced more than
    `previous_days` ago (0 = keep until removed by hand).
    """
    now = now or datetime.utcnow()
    docs = []
    versions = []
    if deleted_days:
        docs = (
            db.query(Document)
            .filter(
                Document.Status == DOC_DELETED,
                Document.DeletedAt.isnot(None),
                Document.DeletedAt <= now - timedelta(days=deleted_days),
            )
            .all()
        )
    if previous_days:
        versions = (
            db.query(DocumentVersion)
            .join(Document, Document.Id == DocumentVersion.DocumentId)
            .filter(
                DocumentVersion.Status == PREVIOUS,
                Document.Status == DOC_ACTIVE,
                DocumentVersion.StatusChangedAt <= now - timedelta(days=previous_days),
            )
            .all()
        )
    return docs, versions
