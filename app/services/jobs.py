"""
Background uploads and syncs, with live progress (006_background_jobs.sql).

Uploads and "Sync Now" used to run inside the web request; a big file or
folder could take minutes behind a spinner. Now the request only queues
the work and returns a job number; the pages ask for its progress every
second or two (GET /api/documents/jobs/...).

How the work runs -- all inside the API process, nothing extra to install:

  * A small pool of JOB_WORKERS threads (app/config.py, default 2) does the
    slow part of each file -- downloading (cloud), reading the text and
    creating embeddings -- for uploads and syncs alike.
  * Checking versions and saving (ingestion.ingest_local_upload /
    cloud_sync.process_entry) runs under one lock, one file at a time, so
    two files can never both replace the same document. The look-ahead
    done before the lock is re-checked under it.
  * Each file's IngestionJobItem carries its live Step ('queued',
    'checking', 'downloading', 'extracting', 'embedding' with
    ProgressCurrent/ProgressTotal, 'saving', 'done') and, when done, its
    full result in ResultJson.
  * Cancel: the job is marked 'cancel_requested'; files not started yet
    are skipped as 'cancelled', the ones in progress finish normally.

Restarts: uploaded files wait in UPLOAD_SPOOL_DIR until processed, so an
upload that was queued or in progress when the API stopped is picked up
again automatically (within about two minutes of the restart). A cloud
sync can't resume -- it needs the person's short-lived Microsoft/Google
sign-in -- so it's marked 'interrupted' and Sync Now simply runs again
(unchanged files are skipped).

The upkeep thread keeps this process's waiting work "fresh" in the
database (UpdatedAt) so that, should several API processes ever run side
by side, one never takes over another's live work.
"""

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException, status
from sqlalchemy import func, update
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import IngestionJob, IngestionJobItem, SourceConnection, User
from app.services import app_settings, cloud_sync
from app.services.embeddings import embed_texts
from app.services.ingestion import (
    _extract_chunks,
    _record_failure,
    ingest_local_upload,
    upload_needs_embedding,
    upload_result_dict,
)

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running", "cancel_requested")
_EMBED_BATCH = 8  # chunks per embedding call -- also how often progress moves
_CANCEL_CHECK_SECONDS = 2.0
_STALE_AFTER = timedelta(seconds=90)
_UPKEEP_SECONDS = 30

# Checking versions + saving: one file at a time across the whole process.
SAVE_LOCK = threading.Lock()

_pool = ThreadPoolExecutor(max_workers=max(1, settings.JOB_WORKERS), thread_name_prefix="ingest")
_state_lock = threading.Lock()
_mine_items: set = set()  # upload items queued or running in this process
_mine_jobs: set = set()  # sync jobs this process is running
_upkeep_started = False


def _session() -> Session:
    from app.db.session import SessionLocal

    return SessionLocal()


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Spool folder (uploaded files waiting to be processed)
# ---------------------------------------------------------------------------


def spool_dir() -> Path:
    folder = (
        Path(settings.UPLOAD_SPOOL_DIR)
        if settings.UPLOAD_SPOOL_DIR
        else Path(__file__).resolve().parents[2] / "upload_spool"
    )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def spool_path(item_id: int) -> Path:
    return spool_dir() / f"{item_id}.upload"


def _drop_spool(item_id: int) -> None:
    try:
        spool_path(item_id).unlink(missing_ok=True)
    except OSError:
        logger.warning("Couldn't remove spooled upload %s", item_id)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def set_step(item_id: int, step: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
    """Records a file's live step in its own short transaction (never raises)."""
    db = _session()
    try:
        db.execute(
            update(IngestionJobItem)
            .where(IngestionJobItem.Id == item_id)
            .values(
                Step=step,
                ProgressCurrent=current,
                ProgressTotal=total,
                UpdatedAt=func.sysutcdatetime(),
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001 -- progress is cosmetic; never break the work
        db.rollback()
        logger.exception("Couldn't record progress for item %s", item_id)
    finally:
        db.close()


def embed_with_progress(chunks: list, report: Callable[[int, int], None]) -> list:
    """embed_texts in small batches, reporting (done, total) after each.
    `report` may raise JobCancelled to stop between batches."""
    total = len(chunks)
    report(0, total)
    vectors: list = []
    for start in range(0, total, _EMBED_BATCH):
        vectors.extend(embed_texts(chunks[start : start + _EMBED_BATCH]))
        report(min(start + _EMBED_BATCH, total), total)
    return vectors


class JobCancelled(Exception):
    """Raised inside a file's work when its job was cancelled (before saving)."""


def _cancel_requested(db: Session, job_id: int) -> bool:
    return db.query(IngestionJob.Status).filter(IngestionJob.Id == job_id).scalar() == "cancel_requested"


class _CancelWatch:
    """Cheap "was Cancel pressed?" check, asked often: hits the database at
    most every couple of seconds."""

    def __init__(self, job_id: int):
        self.job_id = job_id
        self._checked = 0.0
        self._cancelled = False

    def check(self) -> None:
        if not self._cancelled and time.monotonic() - self._checked >= _CANCEL_CHECK_SECONDS:
            self._checked = time.monotonic()
            db = _session()
            try:
                self._cancelled = _cancel_requested(db, self.job_id)
            except Exception:  # noqa: BLE001
                db.rollback()
            finally:
                db.close()
        if self._cancelled:
            raise JobCancelled()


def _stepper(item_id: int, watch: "_CancelWatch"):
    """set_step that also stops the work when the job is cancelled."""

    def step(name: str, done: Optional[int] = None, total: Optional[int] = None) -> None:
        watch.check()
        set_step(item_id, name, done, total)

    return step


def request_cancel(db: Session, job_id: int) -> IngestionJob:
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if job.Status in ("queued", "running"):
        job.Status = "cancel_requested"
        db.commit()
        db.refresh(job)
    return job


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


def queue_upload(db: Session, *, user: User, filename: str, content: bytes) -> tuple[int, int]:
    """Stores the file in the spool folder and queues it. Returns (job_id, item_id)."""
    job = IngestionJob(JobType="upload", Status="queued", TotalItems=1, StartedBy=user.Id)
    db.add(job)
    db.flush()
    item = IngestionJobItem(
        JobId=job.Id,
        FileName=filename[:255],
        Step="queued",
        ContentHash=hashlib.sha256(content).digest(),
        FileSizeBytes=len(content),
    )
    db.add(item)
    db.flush()
    job_id, item_id = job.Id, item.Id
    try:
        spool_path(item_id).write_bytes(content)
    except OSError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Couldn't store the uploaded file on the server: {exc}"
        ) from exc
    db.commit()
    _submit_upload(item_id)
    return job_id, item_id


def _submit_upload(item_id: int) -> None:
    with _state_lock:
        if item_id in _mine_items:
            return
        _mine_items.add(item_id)
    _pool.submit(_run_upload_item, item_id)


def _set_result(db: Session, item_id: int, result: dict, *, job_status: str) -> None:
    """Stores the result and finishes the job in ONE commit, so the page never
    sees a finished upload without its result."""
    item = db.get(IngestionJobItem, item_id)
    if item is None:
        return
    item.ResultJson = json.dumps(result)
    item.ProgressCurrent = None
    item.ProgressTotal = None
    job = db.get(IngestionJob, item.JobId)
    job.Status = job_status
    job.FinishedAt = func.sysutcdatetime()
    db.commit()


def _fail_upload(db: Session, item_id: int, message: str) -> None:
    item = db.get(IngestionJobItem, item_id)
    if item is None:
        return
    if item.Step != "done":
        _record_failure(db, item.JobId, item_id, message, keep_running=True)
    _set_result(
        db, item_id, {"outcome": "failed", "detail": message, "filename": item.FileName},
        job_status="failed",
    )
    _drop_spool(item_id)


def _finish_upload_as(db: Session, item: IngestionJobItem, *, outcome: str, job_status: str, message: str) -> None:
    item.Step = "done"
    item.Outcome = outcome
    item.Message = message
    item.UpdatedAt = func.sysutcdatetime()
    item.ResultJson = json.dumps({"outcome": outcome, "detail": message, "filename": item.FileName})
    job = db.get(IngestionJob, item.JobId)
    job.Status = job_status
    job.ProcessedItems = 1
    job.SkippedCount = 1
    job.Message = message
    job.FinishedAt = func.sysutcdatetime()
    db.commit()
    _drop_spool(item.Id)


def _run_upload_item(item_id: int) -> None:
    db = _session()
    try:
        item = db.get(IngestionJobItem, item_id)
        if item is None or item.Step == "done":
            return
        job = db.get(IngestionJob, item.JobId)
        if job.Status == "cancel_requested":
            _finish_upload_as(
                db, item, outcome="cancelled", job_status="cancelled",
                message="Cancelled before it was processed -- nothing was added.",
            )
            return
        path = spool_path(item_id)
        if not path.exists():
            _fail_upload(
                db, item_id,
                "The uploaded file is no longer on the server (it may have been cleaned up) -- "
                "please upload it again.",
            )
            return
        content = path.read_bytes()
        filename = item.FileName
        user = db.get(User, job.StartedBy)
        job.Status = "running"
        if job.StartedAt is None:
            job.StartedAt = func.sysutcdatetime()
        db.commit()

        step = _stepper(item_id, _CancelWatch(job.Id))
        try:
            step("checking")
            prepared = None
            if upload_needs_embedding(db, filename=filename, content_hash=hashlib.sha256(content).digest()):
                step("extracting")
                chunks = _extract_chunks(filename, content)
                vectors = embed_with_progress(chunks, lambda done, total: step("embedding", done, total))
                prepared = (chunks, vectors)
            step("saving")
        except JobCancelled:
            db.rollback()
            _finish_upload_as(
                db, db.get(IngestionJobItem, item_id), outcome="cancelled", job_status="cancelled",
                message="Cancelled while it was being processed -- nothing was added.",
            )
            return
        db.expire_all()
        with SAVE_LOCK:
            result = ingest_local_upload(
                db=db,
                filename=filename,
                content=content,
                uploaded_by=user,
                job_ids=(job.Id, item_id),
                prepared=prepared,
            )
        _set_result(db, item_id, upload_result_dict(result), job_status="completed")
        _drop_spool(item_id)
    except HTTPException as exc:
        db.rollback()
        _fail_upload(db, item_id, str(exc.detail))
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Upload item %s failed", item_id)
        _fail_upload(db, item_id, str(exc)[:1000] or exc.__class__.__name__)
    finally:
        with _state_lock:
            _mine_items.discard(item_id)
        db.close()


# ---------------------------------------------------------------------------
# Cloud syncs
# ---------------------------------------------------------------------------


def start_sync(
    db: Session,
    *,
    connection: SourceConnection,
    user: User,
    list_fn: Callable[[], tuple],
    fetch_fn: Callable[[dict], bytes],
) -> int:
    """Queues a sync of one connection and starts it. Returns the job id."""
    active = (
        db.query(IngestionJob)
        .filter(
            IngestionJob.JobType == "sync",
            IngestionJob.ConnectionId == connection.Id,
            IngestionJob.Status.in_(ACTIVE_STATUSES),
        )
        .first()
    )
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail="This folder is already syncing -- its progress is shown below.",
        )
    job = IngestionJob(
        JobType="sync",
        ConnectionId=connection.Id,
        Status="queued",
        StartedBy=user.Id,
        Message="Waiting to start...",
    )
    db.add(job)
    db.flush()
    job_id = job.Id
    db.commit()
    with _state_lock:
        _mine_jobs.add(job_id)
    threading.Thread(
        target=_run_sync_job,
        args=(job_id, connection.Id, user.Id, list_fn, fetch_fn),
        name=f"sync-{job_id}",
        daemon=True,
    ).start()
    return job_id


def _bump_processed(db: Session, job_id: int) -> None:
    db.execute(
        update(IngestionJob)
        .where(IngestionJob.Id == job_id)
        .values(ProcessedItems=IngestionJob.ProcessedItems + 1)
    )


def _run_sync_file(job_id: int, item_id: int, connection_id: int, user_id: int, entry: dict, fetch_fn) -> "cloud_sync.FileResult":
    db = _session()
    try:
        if _cancel_requested(db, job_id):
            res = cloud_sync.FileResult(
                entry["path"], "cancelled", "Cancelled before this file was synced.",
                external_id=entry.get("external_id"),
            )
            cloud_sync.fill_item(db.get(IngestionJobItem, item_id), res)
            _bump_processed(db, job_id)
            db.commit()
            return res
        connection = db.get(SourceConnection, connection_id)
        user = db.get(User, user_id)
        cfg = app_settings.get_effective(db)
        step = _stepper(item_id, _CancelWatch(job_id))
        try:
            step("checking")
            prepared = cloud_sync.prepare_entry(
                db,
                connection=connection,
                source_type=connection.SourceType,
                entry=entry,
                fetch_fn=fetch_fn,
                on_step=step,
                embed_batch=embed_with_progress,
            )
            step("saving")
        except JobCancelled:
            db.rollback()
            res = cloud_sync.FileResult(
                entry["path"], "cancelled", "Cancelled while this file was being synced -- not changed.",
                external_id=entry.get("external_id"),
            )
            cloud_sync.fill_item(db.get(IngestionJobItem, item_id), res)
            _bump_processed(db, job_id)
            db.commit()
            return res
        except Exception as exc:  # noqa: BLE001 -- download / text problems: this file only
            db.rollback()
            res = cloud_sync.failure_result(entry, exc)
            cloud_sync.fill_item(db.get(IngestionJobItem, item_id), res)
            _bump_processed(db, job_id)
            db.commit()
            return res
        db.expire_all()
        with SAVE_LOCK:
            try:
                res = cloud_sync.process_entry(
                    db,
                    connection=db.get(SourceConnection, connection_id),
                    user=user,
                    source_type=connection.SourceType,
                    entry=entry,
                    cfg=cfg,
                    fetch_fn=fetch_fn,
                    prepared=prepared,
                )
                cloud_sync.fill_item(db.get(IngestionJobItem, item_id), res)
                _bump_processed(db, job_id)
                db.commit()
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                res = cloud_sync.failure_result(entry, exc)
                cloud_sync.fill_item(db.get(IngestionJobItem, item_id), res)
                _bump_processed(db, job_id)
                db.commit()
        return res
    except Exception as exc:  # noqa: BLE001 -- last resort: never leave the item hanging
        db.rollback()
        logger.exception("Sync item %s failed", item_id)
        res = cloud_sync.failure_result(entry, exc)
        try:
            cloud_sync.fill_item(db.get(IngestionJobItem, item_id), res)
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
        return res
    finally:
        db.close()


def _run_sync_job(job_id: int, connection_id: int, user_id: int, list_fn, fetch_fn) -> None:
    db = _session()
    try:
        job = db.get(IngestionJob, job_id)
        connection = db.get(SourceConnection, connection_id)
        user = db.get(User, user_id)
        label = cloud_sync.SOURCE_LABELS.get(connection.SourceType, connection.SourceType)
        job.Status = "running"
        job.StartedAt = func.sysutcdatetime()
        job.Message = f"Listing the files in the {label} folder..."
        db.commit()

        watch = _CancelWatch(job_id)
        last_note = [0.0]

        def listing_progress(found: int, folders: int) -> None:
            watch.check()
            if time.monotonic() - last_note[0] >= 1.0:
                last_note[0] = time.monotonic()
                note = db.get(IngestionJob, job_id)
                note.Message = (
                    f"Listing the files in the {label} folder... {found} found so far "
                    f"({folders} folder{'s' if folders != 1 else ''} read)"
                )
                db.commit()

        try:
            target_files, skipped = list_fn(on_progress=listing_progress)
        except JobCancelled:
            db.rollback()
            job = db.get(IngestionJob, job_id)
            job.Status = "cancelled"
            job.Message = "Cancelled while listing the folder -- nothing was changed."
            job.FinishedAt = func.sysutcdatetime()
            connection = db.get(SourceConnection, connection_id)
            connection.LastSyncedAt = _now()
            connection.LastSyncStatus = "partial"
            connection.LastSyncError = "The last sync was cancelled before it finished."
            db.commit()
            return
        except Exception as exc:  # noqa: BLE001 -- e.g. folder gone, no access, sign-in expired
            message = str(getattr(exc, "detail", None) or exc)[:1000]
            job = db.get(IngestionJob, job_id)
            job.Status = "failed"
            job.Message = message
            job.FinishedAt = func.sysutcdatetime()
            connection.LastSyncedAt = _now()
            connection.LastSyncStatus = "error"
            connection.LastSyncError = message
            db.commit()
            return

        results: list = []
        for path, reason, external_id in skipped:
            res = cloud_sync.FileResult(path, "skipped", reason, external_id=external_id)
            results.append(res)
            cloud_sync._add_item(db, job_id, res)
        queued = []
        for entry in target_files:
            item = IngestionJobItem(
                JobId=job_id,
                FileName=cloud_sync._file_name(entry["path"])[:255],
                SourcePath=entry["path"][:500],
                ExternalId=entry.get("external_id"),
                Step="queued",
            )
            db.add(item)
            db.flush()
            queued.append((item.Id, entry))
        job = db.get(IngestionJob, job_id)
        job.TotalItems = len(target_files) + len(skipped)
        job.ProcessedItems = len(skipped)
        job.Message = f"Syncing {len(target_files)} file(s) from {label}..."
        db.commit()

        futures = [
            _pool.submit(_run_sync_file, job_id, item_id, connection_id, user_id, entry, fetch_fn)
            for item_id, entry in queued
        ]
        for future in futures:
            results.append(future.result())

        db.expire_all()
        cancelled = _cancel_requested(db, job_id)
        if not cancelled:
            listed = {e["external_id"] for e in target_files} | {s[2] for s in skipped if s[2]}
            with SAVE_LOCK:
                results.extend(
                    cloud_sync.remove_missing(
                        db,
                        connection=db.get(SourceConnection, connection_id),
                        user=user,
                        source_type=connection.SourceType,
                        listed=listed,
                        on_result=lambda r: cloud_sync._add_item(db, job_id, r),
                    )
                )
        db.expire_all()
        cloud_sync.finish_sync_job(
            db,
            job=db.get(IngestionJob, job_id),
            connection=db.get(SourceConnection, connection_id),
            results=results,
            cancelled=cancelled,
        )
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Sync job %s failed", job_id)
        try:
            job = db.get(IngestionJob, job_id)
            job.Status = "failed"
            job.Message = f"The sync stopped unexpectedly: {str(exc)[:300]}"
            job.FinishedAt = func.sysutcdatetime()
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
    finally:
        with _state_lock:
            _mine_jobs.discard(job_id)
        db.close()


# ---------------------------------------------------------------------------
# Upkeep: keep this process's work fresh; pick up work left by a restart
# ---------------------------------------------------------------------------

_INTERRUPTED = (
    "Interrupted -- the API restarted while this sync was running. Run Sync Now again; "
    "files already synced are skipped."
)


def _keep_fresh(db: Session) -> None:
    with _state_lock:
        items = list(_mine_items)
        jobs = list(_mine_jobs)
    for start in range(0, len(items), 500):
        db.execute(
            update(IngestionJobItem)
            .where(IngestionJobItem.Id.in_(items[start : start + 500]), IngestionJobItem.Step != "done")
            .values(UpdatedAt=func.sysutcdatetime())
        )
    if jobs:
        db.execute(
            update(IngestionJobItem)
            .where(IngestionJobItem.JobId.in_(jobs), IngestionJobItem.Step != "done")
            .values(UpdatedAt=func.sysutcdatetime())
        )
    db.commit()


def _pick_up_left_work(db: Session) -> None:
    cutoff = _now() - _STALE_AFTER

    # Uploads: resume from the spool folder (or say why not).
    stale = (
        db.query(IngestionJobItem.Id)
        .join(IngestionJob, IngestionJob.Id == IngestionJobItem.JobId)
        .filter(
            IngestionJob.JobType == "upload",
            IngestionJob.Status.in_(ACTIVE_STATUSES),
            IngestionJobItem.Step != "done",
            IngestionJobItem.UpdatedAt < cutoff,
        )
        .all()
    )
    for (item_id,) in stale:
        with _state_lock:
            if item_id in _mine_items:
                continue
        claimed = db.execute(
            update(IngestionJobItem)
            .where(
                IngestionJobItem.Id == item_id,
                IngestionJobItem.UpdatedAt < cutoff,
                IngestionJobItem.Step != "done",
            )
            .values(Step="queued", ProgressCurrent=None, ProgressTotal=None, UpdatedAt=func.sysutcdatetime())
        ).rowcount
        db.commit()
        if claimed != 1:
            continue
        if spool_path(item_id).exists():
            logger.info("Resuming upload item %s after a restart", item_id)
            _submit_upload(item_id)
        else:
            _fail_upload(
                db, item_id,
                "The API restarted before this file was processed and the file wasn't kept -- "
                "please upload it again.",
            )

    # Syncs: can't resume (no sign-in token kept) -> interrupted.
    syncs = (
        db.query(IngestionJob)
        .filter(IngestionJob.JobType == "sync", IngestionJob.Status.in_(ACTIVE_STATUSES))
        .all()
    )
    for job in syncs:
        with _state_lock:
            if job.Id in _mine_jobs:
                continue
        latest = (
            db.query(func.max(IngestionJobItem.UpdatedAt)).filter(IngestionJobItem.JobId == job.Id).scalar()
            or job.StartedAt
            or job.CreatedAt
        )
        if latest is None or latest >= cutoff:
            continue
        claimed = db.execute(
            update(IngestionJob)
            .where(IngestionJob.Id == job.Id, IngestionJob.Status.in_(ACTIVE_STATUSES))
            .values(Status="interrupted", Message=_INTERRUPTED, FinishedAt=func.sysutcdatetime())
        ).rowcount
        if claimed != 1:
            db.rollback()
            continue
        db.execute(
            update(IngestionJobItem)
            .where(IngestionJobItem.JobId == job.Id, IngestionJobItem.Step != "done")
            .values(
                Step="done",
                Outcome="interrupted",
                Message="Not synced -- the API restarted.",
                ResultJson=json.dumps({"status": "interrupted", "detail": "Not synced -- the API restarted."}),
                UpdatedAt=func.sysutcdatetime(),
            )
        )
        if job.ConnectionId:
            connection = db.get(SourceConnection, job.ConnectionId)
            if connection is not None:
                connection.LastSyncStatus = "error"
                connection.LastSyncError = _INTERRUPTED
        db.commit()


def _warm_up_model() -> None:
    """Loads the embedding model in the background at start-up, so the first
    upload or sync doesn't sit at "Creating embeddings 0 of N" while it loads."""
    try:
        embed_texts(["warm-up"])
    except Exception:  # noqa: BLE001
        logger.exception("Embedding model warm-up failed (it will load on first use)")


def _upkeep_loop() -> None:
    _warm_up_model()
    time.sleep(5)
    while True:
        db = _session()
        try:
            _keep_fresh(db)
            _pick_up_left_work(db)
        except Exception:  # noqa: BLE001
            db.rollback()
            logger.exception("Background job upkeep failed")
        finally:
            db.close()
        time.sleep(_UPKEEP_SECONDS)


def start_upkeep() -> None:
    """Starts the upkeep thread once per process (app/api/routes/documents.py)."""
    global _upkeep_started
    with _state_lock:
        if _upkeep_started:
            return
        _upkeep_started = True
    threading.Thread(target=_upkeep_loop, name="job-upkeep", daemon=True).start()


# ---------------------------------------------------------------------------
# What the pages read
# ---------------------------------------------------------------------------


def item_dict(item: IngestionJobItem) -> dict:
    try:
        result = json.loads(item.ResultJson) if item.ResultJson else None
    except ValueError:
        result = None
    return {
        "id": item.Id,
        "filename": item.FileName,
        "source_path": item.SourcePath,
        "step": item.Step,
        "progress_current": item.ProgressCurrent,
        "progress_total": item.ProgressTotal,
        "outcome": item.Outcome,
        "message": item.Message,
        "result": result,
    }


def job_dict(db: Session, job: IngestionJob, *, with_items: bool = True) -> dict:
    items = []
    summary = None
    if with_items:
        rows = (
            db.query(IngestionJobItem)
            .filter(IngestionJobItem.JobId == job.Id)
            .order_by(IngestionJobItem.Id)
            .all()
        )
        items = [item_dict(r) for r in rows]
        if job.JobType == "sync":
            counts: dict = {}
            for it in items:
                st = (it["result"] or {}).get("status")
                if st:
                    counts[st] = counts.get(st, 0) + 1
            summary = counts
    # Files being worked on right now (always included, even without items)
    current_rows = (
        db.query(IngestionJobItem)
        .filter(
            IngestionJobItem.JobId == job.Id,
            IngestionJobItem.Step.notin_(("queued", "done")),
        )
        .order_by(IngestionJobItem.Id)
        .limit(5)
        .all()
        if job.Status in ACTIVE_STATUSES
        else []
    )
    starter = db.get(User, job.StartedBy) if job.StartedBy else None
    return {
        "id": job.Id,
        "job_type": job.JobType,
        "status": job.Status,
        "connection_id": job.ConnectionId,
        "total_items": job.TotalItems,
        "processed_items": job.ProcessedItems,
        "added": job.AddedCount,
        "updated": job.UpdatedCount,
        "unchanged": job.UnchangedCount,
        "removed": job.RemovedCount,
        "skipped": job.SkippedCount,
        "failed": job.FailedCount,
        "message": job.Message,
        "started_by": starter.DisplayName if starter else None,
        "created_at": job.CreatedAt,
        "started_at": job.StartedAt,
        "finished_at": job.FinishedAt,
        "status_counts": summary,
        "current": [item_dict(r) for r in current_rows],
        "items": items,
    }


def list_jobs(
    db: Session,
    *,
    kind: Optional[str],
    active_only: bool,
    connection_id: Optional[int],
    ids: Optional[list],
    started_by: Optional[int],
    since_hours: Optional[int],
    limit: int,
    with_items: bool,
) -> list:
    query = db.query(IngestionJob)
    if kind:
        query = query.filter(IngestionJob.JobType == kind)
    if active_only:
        query = query.filter(IngestionJob.Status.in_(ACTIVE_STATUSES))
    if connection_id is not None:
        query = query.filter(IngestionJob.ConnectionId == connection_id)
    if ids:
        query = query.filter(IngestionJob.Id.in_(ids[:200]))
    if started_by is not None:
        query = query.filter(IngestionJob.StartedBy == started_by)
    if since_hours:
        query = query.filter(IngestionJob.CreatedAt >= _now() - timedelta(hours=since_hours))
    jobs = query.order_by(IngestionJob.Id.desc()).limit(max(1, min(limit, 200))).all()
    return [job_dict(db, j, with_items=with_items) for j in jobs]
