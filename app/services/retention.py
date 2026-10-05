"""
Retention clean-up (Settings > History and retention): permanently deletes
  * documents that have been in the Library's Deleted tab longer than
    DELETED_RETENTION_DAYS, and
  * previous versions replaced longer than PREVIOUS_VERSION_RETENTION_DAYS
    ago (the document itself stays; only that old version's content goes).
A setting of 0 keeps those items until someone removes them by hand.

Runs by itself -- no change to main.py needed: a background thread starts
the first time this module is imported (app/api/routes/documents.py does
that at API start-up), runs once a minute after start-up and then every
few hours. The Deleted tab also triggers it (at most every few minutes),
so what it lists is always current. Running it twice is harmless, so
several API worker processes each running their own thread is fine too.
"""

import logging
import threading
import time

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import IngestionJob, IngestionJobItem
from app.services import app_settings, doc_history

logger = logging.getLogger(__name__)

_RUN_EVERY_SECONDS = 6 * 60 * 60
_FIRST_RUN_AFTER_SECONDS = 60
_ON_DEMAND_MIN_GAP_SECONDS = 5 * 60

_lock = threading.Lock()
_state = {"last_run": 0.0, "thread": None}


def purge_expired(db: Session) -> dict:
    """Removes everything past its retention period. Commits. Returns counts."""
    cfg = app_settings.get_effective(db)
    deleted_days = cfg["DELETED_RETENTION_DAYS"]
    previous_days = cfg["PREVIOUS_VERSION_RETENTION_DAYS"]
    docs, versions = doc_history.expired_items(
        db, deleted_days=deleted_days, previous_days=previous_days
    )
    if not docs and not versions:
        return {"documents": 0, "versions": 0}

    job = IngestionJob(
        JobType="cleanup",
        Status="completed",
        TotalItems=len(docs) + len(versions),
        ProcessedItems=len(docs) + len(versions),
        RemovedCount=len(docs) + len(versions),
        # no person started it; the history shows it as the system's
        StartedBy=_system_user_id(db, docs, versions),
        StartedAt=func.sysutcdatetime(),
        FinishedAt=func.sysutcdatetime(),
        Message=(
            f"Retention clean-up: {len(docs)} deleted document(s) older than {deleted_days} days "
            f"and {len(versions)} previous version(s) older than {previous_days} days were "
            "permanently deleted."
        )[:1000],
    )
    db.add(job)
    db.flush()

    for doc in docs:
        chunks = doc_history.purge_document(
            db,
            doc,
            by_user_id=None,
            reason=f"Permanently deleted automatically after {deleted_days} days in Deleted.",
        )
        db.add(
            IngestionJobItem(
                JobId=job.Id,
                FileName=doc.FileName,
                Step="done",
                Outcome="permanently_deleted",
                DocumentId=doc.Id,
                PreviousVersion=doc.Version,
                ChunkCount=chunks,
                Message=f"In Deleted for more than {deleted_days} days.",
            )
        )
    for version in versions:
        chunks = doc_history.purge_version(
            db,
            version,
            by_user_id=None,
            note=f"Permanently deleted automatically {previous_days} days after it was replaced.",
        )
        db.add(
            IngestionJobItem(
                JobId=job.Id,
                FileName=version.FileName,
                Step="done",
                Outcome="version_removed",
                DocumentId=version.DocumentId,
                PreviousVersion=version.VersionNumber,
                ChunkCount=chunks,
                Message=f"Previous version older than {previous_days} days.",
            )
        )
    db.commit()
    logger.info("Retention clean-up removed %s document(s), %s version(s)", len(docs), len(versions))
    return {"documents": len(docs), "versions": len(versions)}


def _system_user_id(db: Session, docs, versions) -> int:
    # DarAI_IngestionJobs.StartedBy is NOT NULL; use whoever deleted /
    # replaced the first item, else the first account.
    from app.db.models import User

    for doc in docs:
        if doc.DeletedBy:
            return doc.DeletedBy
    for v in versions:
        if v.StatusChangedBy:
            return v.StatusChangedBy
    first = db.query(User.Id).order_by(User.Id).first()
    return first[0]


def purge_if_due(db: Session) -> None:
    """Runs the clean-up if it hasn't run in the last few minutes (never raises)."""
    with _lock:
        if time.monotonic() - _state["last_run"] < _ON_DEMAND_MIN_GAP_SECONDS:
            return
        _state["last_run"] = time.monotonic()
    try:
        purge_expired(db)
    except Exception:  # noqa: BLE001 -- a clean-up problem must never break the page
        db.rollback()
        logger.exception("Retention clean-up failed")


def _loop() -> None:
    from app.db.session import SessionLocal

    time.sleep(_FIRST_RUN_AFTER_SECONDS)
    while True:
        db = SessionLocal()
        try:
            with _lock:
                _state["last_run"] = time.monotonic()
            purge_expired(db)
        except Exception:  # noqa: BLE001
            db.rollback()
            logger.exception("Retention clean-up failed")
        finally:
            db.close()
        time.sleep(_RUN_EVERY_SECONDS)


def start_background_cleanup() -> None:
    """Starts the clean-up thread once per process."""
    with _lock:
        if _state["thread"] is not None:
            return
        thread = threading.Thread(target=_loop, name="retention-cleanup", daemon=True)
        _state["thread"] = thread
    thread.start()
