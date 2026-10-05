"""
OneDrive / Google Drive sync, on the same version rules as local uploads
(app/services/ingestion.py's ingest_local_upload) and with the same
history (app/services/doc_history.py).

Every cloud file is tracked by its own ID in the cloud (Document.ExternalId,
per Document.ConnectionId -- unique, see 003_document_versioning.sql), which
doesn't change when the file is renamed or moved. On each sync, for every
supported file in the folder:

  * Already synced, unchanged in the cloud (same change marker:
    OneDrive cTag / Google md5Checksum or version) -> "unchanged", without
    downloading it. Renamed or moved -> "renamed" (name/path updated; the
    Library name only follows if nobody renamed it in the Library).
  * Already synced, changed in the cloud -> "updated": a new version of the
    SAME document; the old content becomes a 'Previous version' in History.
  * Excluded in the Library -> "excluded": left alone (not updated) until
    it's included again.
  * New to this connection -> the upload rules:
      identical content already in the knowledge base -> not added again
        ("linked" when it was a local upload: that document now follows
        the cloud file -- the cloud copy takes over);
      a new version (same name / 80%+ same wording) of a local upload ->
        "updated": that document takes the new content and follows the
        cloud file from now on;
      an older edition of a stored document -> "older_version", not added;
      otherwise -> "added" ("review" when it looked like a version but
        didn't meet the rules -- both kept).
  * Files synced the old way (before cloud IDs were stored -- every sync
    added another copy) are linked to the document with the same path on
    the first sync; extra copies are set aside in the Deleted tab.

Then every document of this connection whose file is no longer in the
folder is permanently deleted ("removed"). Safety: if the folder lists no
files at all, nothing is removed (an empty listing is more likely an
access problem than a deliberate wipe).

Each file is committed on its own, so one bad file never undoes the rest,
and every sync is recorded as an IngestionJob (JobType 'sync') with one
IngestionJobItem per file.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Document, IngestionJob, IngestionJobItem, SourceConnection, User
from app.services import app_settings, doc_history
from app.services.doc_matching import doc_year, text_shingles
from app.services.embeddings import embed_texts
from app.services.ingestion import (
    _extract_chunks,
    _find_version_matches,
    _insert_chunks,
    _near_miss_note,
    _pct_range,
)

logger = logging.getLogger(__name__)

SOURCE_LABELS = {"onedrive": "OneDrive", "googledrive": "Google Drive", "upload": "an upload"}

# FileResult.status -> IngestionJobItem.Outcome (kept in line with the
# upload outcomes, which the version matching reads back as known names)
_OUTCOMES = {
    "added": "new",
    "review": "possible_version",
    "updated": "updated",
    "linked": "linked",
    "renamed": "renamed",
    "unchanged": "unchanged",
    "duplicate": "unchanged",
    "excluded": "excluded",
    "older_version": "older_version",
    "removed": "permanently_deleted",
    "skipped": "skipped",
    "failed": "failed",
}


@dataclass
class FileResult:
    path: str
    # 'added' | 'updated' | 'renamed' | 'unchanged' | 'linked' | 'review' |
    # 'excluded' | 'duplicate' | 'older_version' | 'removed' | 'skipped' | 'failed'
    status: str
    detail: Optional[str] = None
    document_id: Optional[int] = None
    external_id: Optional[str] = None
    previous_version: Optional[int] = None
    new_version: Optional[int] = None
    chunk_count: Optional[int] = None


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    """'2026-10-05T10:00:00.123Z' -> naive UTC datetime (None if unparseable)."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _file_name(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _link(doc: Document, *, connection: SourceConnection, source_type: str, entry: dict) -> None:
    doc.SourceType = source_type
    doc.ConnectionId = connection.Id
    doc.ExternalId = entry["external_id"]
    doc.SourcePath = entry["path"]
    doc.SourceVersionTag = entry.get("version_tag")
    doc.SourceModifiedAt = _parse_time(entry.get("modified_at"))


def _fetch(fetch_fn: Callable[[dict], bytes], entry: dict) -> bytes:
    content = fetch_fn(entry)
    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="empty file")
    return content


# ---------------------------------------------------------------------------
# Files already known to this connection
# ---------------------------------------------------------------------------


def _adopt_legacy(
    db: Session, *, connection: SourceConnection, source_type: str, entry: dict, user: User
) -> tuple[Optional[Document], int]:
    """
    A file synced before cloud IDs were stored: find its document(s) by
    path, link the most recent one, and set any extra copies aside in the
    Deleted tab. Returns (linked document or None, copies set aside).
    """
    candidates = (
        db.query(Document)
        .filter(
            Document.SourceType == source_type,
            Document.ExternalId.is_(None),
            Document.SourcePath == entry["path"],
            Document.Status.in_((doc_history.DOC_ACTIVE, doc_history.DOC_EXCLUDED)),
        )
        .order_by(Document.UpdatedAt.desc(), Document.UploadedAt.desc(), Document.Id.desc())
        .all()
    )
    if not candidates:
        return None, 0
    keep = candidates[0]
    _link(keep, connection=connection, source_type=source_type, entry=entry)
    # Unknown until compared once: forces a download + content check.
    keep.SourceVersionTag = None
    set_aside = 0
    for extra in candidates[1:]:
        if extra.Status == doc_history.DOC_ACTIVE:
            doc_history.soft_delete(
                db,
                extra,
                by_user_id=user.Id,
                reason="Duplicate copy of the same cloud file, added by an earlier sync.",
                merged_into_document_id=keep.Id,
            )
            set_aside += 1
    db.flush()
    return keep, set_aside


def _sync_known(
    db: Session,
    doc: Document,
    *,
    entry: dict,
    user: User,
    label: str,
    fetch_fn: Callable[[dict], bytes],
    set_aside: int,
) -> FileResult:
    path = entry["path"]
    name = _file_name(path)
    tag = entry.get("version_tag")
    modified = _parse_time(entry.get("modified_at"))
    notes = []
    if set_aside:
        notes.append(
            f"{set_aside} duplicate {'copy' if set_aside == 1 else 'copies'} from earlier syncs "
            "set aside in Deleted"
        )

    if doc.Status == doc_history.DOC_DELETED:
        return FileResult(
            path, "skipped", "Set aside in the Library's Deleted tab -- restore it to sync it again.",
            document_id=doc.Id,
        )

    renamed_from = None
    moved = False
    if doc.SourcePath != path:
        moved = True
        old_cloud_name = _file_name(doc.SourcePath) if doc.SourcePath else None
        # Follow the cloud name only if nobody renamed it in the Library.
        if old_cloud_name is None or doc.FileName == old_cloud_name:
            if doc.FileName != name:
                renamed_from = doc.FileName
                doc.FileName = name
        doc.SourcePath = path
        if renamed_from is None:
            notes.append("moved or renamed in the cloud")
    if renamed_from:
        notes.insert(0, f"renamed from '{renamed_from}'")

    def _detail() -> Optional[str]:
        return "; ".join(notes) if notes else None

    if doc.Status == doc_history.DOC_EXCLUDED:
        notes.append("excluded from the chat -- not updated")
        return FileResult(path, "excluded", _detail(), document_id=doc.Id)

    if tag and doc.SourceVersionTag == tag:
        if modified:
            doc.SourceModifiedAt = modified
        return FileResult(path, "renamed" if moved else "unchanged", _detail(), document_id=doc.Id)

    content = _fetch(fetch_fn, entry)
    content_hash = hashlib.sha256(content).digest()
    if content_hash == doc.ContentHash:
        doc.SourceVersionTag = tag
        doc.SourceModifiedAt = modified
        doc.FileSizeBytes = len(content)
        return FileResult(path, "renamed" if moved else "unchanged", _detail(), document_id=doc.Id)

    # Changed in the cloud: a new version of this same document.
    chunks = _extract_chunks(name, content)
    vectors = embed_texts(chunks)
    previous = doc.Version
    new = previous + 1
    doc_history.archive_current(
        db,
        doc,
        new_status=doc_history.PREVIOUS,
        by_user_id=user.Id,
        note=f"Replaced when the file changed in {label} (v{new}).",
    )
    doc.ContentHash = content_hash
    doc.FileSizeBytes = len(content)
    doc.Version = new
    doc.UpdatedAt = func.sysutcdatetime()
    doc.UpdatedBy = user.Id
    doc.SourceVersionTag = tag
    doc.SourceModifiedAt = modified
    db.flush()
    _insert_chunks(db, doc.Id, chunks, vectors)
    doc_history.record_new_latest(
        db, doc, chunk_count=len(chunks), how_added=f"Changed in {label}", by_user_id=user.Id
    )
    notes.insert(0, f"changed in {label}: v{previous} -> v{new}")
    return FileResult(
        path, "updated", _detail(), document_id=doc.Id,
        previous_version=previous, new_version=new, chunk_count=len(chunks),
    )


# ---------------------------------------------------------------------------
# Files new to this connection: the upload rules
# ---------------------------------------------------------------------------


def _sync_new(
    db: Session,
    *,
    entry: dict,
    connection: SourceConnection,
    source_type: str,
    user: User,
    label: str,
    cfg: dict,
    fetch_fn: Callable[[dict], bytes],
) -> FileResult:
    path = entry["path"]
    name = _file_name(path)
    content = _fetch(fetch_fn, entry)
    content_hash = hashlib.sha256(content).digest()

    identical = (
        db.query(Document)
        .filter(
            Document.ContentHash == content_hash,
            Document.Status.in_((doc_history.DOC_ACTIVE, doc_history.DOC_EXCLUDED)),
        )
        .order_by(Document.Id)
        .first()
    )
    if identical is not None:
        if (
            identical.SourceType == "upload"
            and identical.ExternalId is None
            and identical.Status == doc_history.DOC_ACTIVE
        ):
            old_name = identical.FileName
            _link(identical, connection=connection, source_type=source_type, entry=entry)
            identical.FileName = name
            identical.UpdatedAt = func.sysutcdatetime()
            identical.UpdatedBy = user.Id
            db.flush()
            return FileResult(
                path, "linked",
                f"Same content as the uploaded '{old_name}' -- that document now follows {label}.",
                document_id=identical.Id,
            )
        where = (
            "an excluded document"
            if identical.Status == doc_history.DOC_EXCLUDED
            else f"'{identical.FileName}' (from {SOURCE_LABELS.get(identical.SourceType, identical.SourceType)})"
        )
        return FileResult(
            path, "duplicate", f"Same content as {where} -- not added again.", document_id=identical.Id
        )

    chunks = _extract_chunks(name, content)
    new_text = " ".join(chunks)
    vectors = embed_texts(chunks)
    matches, near_miss = _find_version_matches(
        db,
        filename=name,
        new_shingles=text_shingles(new_text),
        new_word_count=len(new_text.split()),
        vectors=vectors,
        cfg=cfg,
    )

    if matches and cfg["BLOCK_OLDER_EDITIONS"]:
        file_year = doc_year(name, new_text)
        newest = None
        for m in matches:
            year = doc_year(m.document.FileName, m.text)
            if year is not None and (newest is None or year > newest[1]):
                newest = (m, year)
        if file_year is not None and newest is not None and newest[1] > file_year:
            return FileResult(
                path, "older_version",
                f"Older version of '{newest[0].document.FileName}' ({file_year} vs {newest[1]}) -- "
                "the newer one is already in the knowledge base, so this file wasn't added.",
                document_id=newest[0].document.Id,
            )

    if matches:
        # A new version of a local upload: the cloud copy takes over.
        target_match = matches[0]
        target = target_match.document
        never_merge = doc_history.excluded_ids(db, target.Id)
        extras = [m.document for m in matches[1:] if m.document.Id not in never_merge]
        previous = target.Version
        new = previous + 1
        old_name = target.FileName
        doc_history.archive_current(
            db,
            target,
            new_status=doc_history.PREVIOUS,
            by_user_id=user.Id,
            note=f"Replaced by '{name}' from {label} (v{new}).",
        )
        for extra in extras:
            doc_history.soft_delete(
                db,
                extra,
                by_user_id=user.Id,
                reason=f"Older copy of the same document -- merged into '{name}' from {label}.",
                merged_into_document_id=target.Id,
            )
        _link(target, connection=connection, source_type=source_type, entry=entry)
        target.FileName = name
        target.ContentHash = content_hash
        target.FileSizeBytes = len(content)
        target.Version = new
        target.UpdatedAt = func.sysutcdatetime()
        target.UpdatedBy = user.Id
        db.flush()
        _insert_chunks(db, target.Id, chunks, vectors)
        how = doc_history.HOW_ADDED[target_match.matched_by]
        doc_history.record_new_latest(
            db, target, chunk_count=len(chunks), how_added=f"{how} ({label})"[:50],
            by_user_id=user.Id, score=target_match.similarity,
        )
        if target_match.matched_by == "filename":
            why = "same file name"
        elif target_match.matched_by == "name_and_content":
            why = f"name matched, wording {round(target_match.similarity * 100)}% the same"
        else:
            why = (
                f"same content, {_pct_range(target_match.similarity, target_match.similarity_max)} "
                "match"
            )
        detail = (
            f"New version of the uploaded '{old_name}' ({why}): v{previous} -> v{new}; it now "
            f"follows {label}."
        )
        if extras:
            detail += " Older copies moved to Deleted: " + ", ".join(f"'{d.FileName}'" for d in extras) + "."
        return FileResult(
            path, "updated", detail, document_id=target.Id,
            previous_version=previous, new_version=new, chunk_count=len(chunks),
        )

    note = _near_miss_note(near_miss, cfg) if near_miss is not None else None
    document = Document(
        SourceType=source_type,
        FileName=name,
        UploadedBy=user.Id,
        ContentHash=content_hash,
        FileSizeBytes=len(content),
        Version=1,
        UpdatedBy=user.Id,
        Status=doc_history.DOC_ACTIVE,
    )
    _link(document, connection=connection, source_type=source_type, entry=entry)
    db.add(document)
    db.flush()
    _insert_chunks(db, document.Id, chunks, vectors)
    doc_history.record_new_latest(
        db, document, chunk_count=len(chunks), how_added=f"Added from {label}", by_user_id=user.Id
    )
    return FileResult(
        path, "review" if note else "added", note, document_id=document.Id,
        new_version=1, chunk_count=len(chunks),
    )


# ---------------------------------------------------------------------------
# One sync run
# ---------------------------------------------------------------------------


def _add_item(db: Session, job_id: int, res: FileResult, *, related_id: Optional[int] = None) -> None:
    db.add(
        IngestionJobItem(
            JobId=job_id,
            FileName=_file_name(res.path)[:255],
            SourcePath=res.path[:500],
            ExternalId=res.external_id,
            Step="done",
            Outcome=_OUTCOMES[res.status],
            DocumentId=res.document_id if res.status not in ("duplicate", "older_version") else None,
            RelatedDocumentId=res.document_id if res.status in ("duplicate", "older_version") else related_id,
            PreviousVersion=res.previous_version,
            NewVersion=res.new_version,
            ChunkCount=res.chunk_count,
            Message=(res.detail or None) and res.detail[:1000],
            UpdatedAt=func.sysutcdatetime(),
        )
    )


def run_sync(
    db: Session,
    *,
    connection: SourceConnection,
    user: User,
    source_type: str,
    target_files: list[dict],
    skipped: list[tuple],
    fetch_fn: Callable[[dict], bytes],
) -> list[FileResult]:
    """Syncs one connection (see module docstring). Commits as it goes."""
    label = SOURCE_LABELS.get(source_type, source_type)
    cfg = app_settings.get_effective(db)

    job = IngestionJob(
        JobType="sync",
        ConnectionId=connection.Id,
        Status="running",
        TotalItems=len(target_files) + len(skipped),
        StartedBy=user.Id,
        StartedAt=func.sysutcdatetime(),
    )
    db.add(job)
    db.flush()
    job_id = job.Id
    db.commit()

    results: list[FileResult] = []
    for path, reason, external_id in skipped:
        res = FileResult(path, "skipped", reason, external_id=external_id)
        results.append(res)
        _add_item(db, job_id, res)
    db.commit()

    for entry in target_files:
        path = entry["path"]
        try:
            doc = (
                db.query(Document)
                .filter(
                    Document.ConnectionId == connection.Id,
                    Document.ExternalId == entry["external_id"],
                    Document.Status != doc_history.DOC_PERMANENTLY_DELETED,
                )
                .first()
            )
            set_aside = 0
            if doc is None:
                doc, set_aside = _adopt_legacy(
                    db, connection=connection, source_type=source_type, entry=entry, user=user
                )
            if doc is not None:
                res = _sync_known(
                    db, doc, entry=entry, user=user, label=label, fetch_fn=fetch_fn, set_aside=set_aside
                )
            else:
                res = _sync_new(
                    db, entry=entry, connection=connection, source_type=source_type,
                    user=user, label=label, cfg=cfg, fetch_fn=fetch_fn,
                )
            res.external_id = entry["external_id"]
            _add_item(db, job_id, res)
            db.commit()
        except Exception as exc:  # noqa: BLE001 -- one bad file must never abort the batch
            db.rollback()
            detail = exc.detail if isinstance(exc, HTTPException) else (str(exc)[:300] or exc.__class__.__name__)
            res = FileResult(path, "failed", str(detail), external_id=entry.get("external_id"))
            try:
                _add_item(db, job_id, res)
                db.commit()
            except Exception:  # noqa: BLE001
                db.rollback()
            if not isinstance(exc, HTTPException):
                logger.exception("Sync of %s failed", path)
        results.append(res)

    # Files no longer in the cloud folder -> permanently deleted.
    listed = {e["external_id"] for e in target_files} | {s[2] for s in skipped if s[2]}
    gone = (
        db.query(Document)
        .filter(
            Document.ConnectionId == connection.Id,
            Document.ExternalId.isnot(None),
            Document.Status.in_((doc_history.DOC_ACTIVE, doc_history.DOC_EXCLUDED)),
        )
        .all()
    )
    gone = [d for d in gone if d.ExternalId not in listed]
    if gone and not listed:
        results.append(
            FileResult(
                "(whole folder)", "skipped",
                f"The {label} folder listed no files, so nothing was removed -- check that the "
                "folder still exists and is shared with you.",
            )
        )
    else:
        for doc in gone:
            path = doc.SourcePath or doc.FileName
            try:
                chunks = doc_history.purge_document(
                    db, doc, by_user_id=user.Id, reason=f"Removed from the {label} folder."
                )
                res = FileResult(
                    path, "removed", f"No longer in the {label} folder -- permanently deleted.",
                    document_id=doc.Id, chunk_count=chunks,
                )
                _add_item(db, job_id, res)
                db.commit()
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                res = FileResult(path, "failed", f"Couldn't remove: {str(exc)[:200]}")
            results.append(res)

    counts = summarize(results)
    job = db.get(IngestionJob, job_id)
    job.ProcessedItems = len(results)
    job.AddedCount = counts["added"] + counts["review"]
    job.UpdatedCount = counts["updated"] + counts["linked"] + counts["renamed"]
    job.UnchangedCount = counts["unchanged"] + counts["duplicate"] + counts["excluded"]
    job.RemovedCount = counts["removed"]
    job.SkippedCount = counts["skipped"] + counts["older_version"]
    job.FailedCount = counts["failed"]
    ok = len(results) - counts["failed"]
    job.Status = "failed" if counts["failed"] and not ok else ("partial" if counts["failed"] else "completed")
    job.Message = (
        f"Synced {label} '{connection.DisplayLabel}': "
        + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in counts.items() if n)
    )[:1000]
    job.FinishedAt = func.sysutcdatetime()

    connection.LastSyncedAt = datetime.utcnow()
    if counts["failed"] and not ok:
        connection.LastSyncStatus = "error"
        connection.LastSyncError = f"All {counts['failed']} file(s) failed -- see sync details."
    elif counts["failed"]:
        connection.LastSyncStatus = "partial"
        connection.LastSyncError = f"{counts['failed']} file(s) failed -- see sync details."
    else:
        connection.LastSyncStatus = "success"
        connection.LastSyncError = None
    db.commit()
    return results


def summarize(results: list[FileResult]) -> dict:
    counts = {k: 0 for k in _OUTCOMES}
    for r in results:
        counts[r.status] += 1
    return counts
