"""
OneDrive / Google Drive connectors (spec §3.2, §7, §8 item 4 -- the final
phase; Google Drive replaces the connector originally built for GitHub).
Both connect flows are "paste a path/link, connect, and it reads +
ingests immediately" -- no folder-browsing UI. Every connection can be
re-synced later via POST /connections/{id}/sync (spec §7: manual, on
demand -- no background scheduler).

Neither connector persists anything auth-related server-side
(DarAI_SourceConnections.Secret stays NULL for both source types) -- both
authenticate as whichever person is using the app, via a short-lived
access token the frontend acquires in-browser right before each call and
sends along with the request (OneDrive via MSAL against Microsoft Graph,
Google Drive via Google Identity Services against the Drive API -- see
app/services/onedrive_source.py's and app/services/googledrive_source.py's
docstrings). Practical effect for Google Drive: a person can only pull in
files their own Google account can actually see, and if their browser's
Google session lapses, the next call fails with a clear "sign in again"
error rather than silently using someone else's access.

Both connect and sync reuse app/services/ingestion.py's ingest_upload()
per file -- the same parse/chunk/embed/store/dedupe pipeline as a local
upload, just with source_type set to "onedrive"/"googledrive" and
source_path set to the file's path within that source. One bad file
(parse failure, transient network error) is recorded and skipped rather
than aborting the whole sync.
"""

import json
from datetime import datetime
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.models import SourceConnection, User
from app.db.session import get_db
from app.schemas import (
    ConnectResponse,
    GoogleDriveConnectRequest,
    OneDriveConnectRequest,
    SourceConnectionItem,
    SyncFileResult,
    SyncRequest,
    SyncResponse,
)
from app.services import googledrive_source, onedrive_source
from app.services.ingestion import ingest_upload

router = APIRouter()


def _to_item(connection: SourceConnection, created_by_name: Optional[str]) -> SourceConnectionItem:
    return SourceConnectionItem(
        id=connection.Id,
        source_type=connection.SourceType,
        display_label=connection.DisplayLabel,
        created_by=created_by_name,
        created_at=connection.CreatedAt,
        last_synced_at=connection.LastSyncedAt,
        last_sync_status=connection.LastSyncStatus,
        last_sync_error=connection.LastSyncError,
    )


def _sync_and_record(
    db: Session,
    *,
    connection: SourceConnection,
    uploaded_by: User,
    source_type: str,
    target_files: list[dict],
    skipped: list[tuple[str, str]],
    fetch_fn: Callable[[dict], bytes],
) -> SyncResponse:
    added = duplicate = failed = 0
    details: list[SyncFileResult] = [
        SyncFileResult(path=path, status="skipped", detail=reason) for path, reason in skipped
    ]

    for entry in target_files:
        path = entry["path"]
        try:
            content = fetch_fn(entry)
            if not content:
                failed += 1
                details.append(SyncFileResult(path=path, status="failed", detail="empty file"))
                continue

            filename = path.rsplit("/", 1)[-1]
            result = ingest_upload(
                db=db,
                filename=filename,
                content=content,
                uploaded_by=uploaded_by,
                source_type=source_type,
                source_path=path,
            )
            if result.is_duplicate:
                duplicate += 1
                details.append(SyncFileResult(path=path, status="duplicate"))
            else:
                added += 1
                details.append(SyncFileResult(path=path, status="added"))
        except HTTPException as exc:
            # ingest_upload raises this for e.g. an unparseable/empty-text
            # file -- always before any DB write in that call, but roll
            # back regardless so a half-flushed session never poisons the
            # next file in this loop.
            db.rollback()
            failed += 1
            details.append(SyncFileResult(path=path, status="failed", detail=str(exc.detail)))
        except Exception as exc:  # noqa: BLE001 -- one bad file must never abort the batch
            db.rollback()
            failed += 1
            details.append(SyncFileResult(path=path, status="failed", detail=str(exc)[:300]))

    connection.LastSyncedAt = datetime.utcnow()
    if failed and not added and not duplicate:
        connection.LastSyncStatus = "error"
        connection.LastSyncError = f"All {failed} file(s) failed -- see sync details."
    elif failed:
        connection.LastSyncStatus = "partial"
        connection.LastSyncError = f"{failed} file(s) failed -- see sync details."
    else:
        connection.LastSyncStatus = "success"
        connection.LastSyncError = None

    return SyncResponse(
        connection_id=connection.Id,
        files_added=added,
        files_duplicate=duplicate,
        files_skipped=len(skipped),
        files_failed=failed,
        details=details,
    )


@router.get("/connections", response_model=list[SourceConnectionItem])
def list_connections(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(SourceConnection, User.DisplayName)
        .join(User, User.Id == SourceConnection.CreatedBy)
        .order_by(SourceConnection.CreatedAt.desc())
        .all()
    )
    return [_to_item(connection, display_name) for connection, display_name in rows]


@router.post("/googledrive/connect", response_model=ConnectResponse)
def connect_googledrive(
    payload: GoogleDriveConnectRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        folder_id = googledrive_source.parse_drive_path(payload.folder_path)
    except googledrive_source.GoogleDrivePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        folder = googledrive_source.resolve_folder(payload.access_token, folder_id)
        target_files, skipped = googledrive_source.list_target_files(
            payload.access_token, folder_id, folder.get("name", "")
        )
    except googledrive_source.GoogleDriveConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    label = payload.display_label or folder.get("name") or "Google Drive folder"
    config = {"folder_id": folder_id, "path": folder.get("name", "")}
    connection = SourceConnection(
        SourceType="googledrive",
        DisplayLabel=label,
        ConfigJson=json.dumps(config),
        Secret=None,
        CreatedBy=user.Id,
    )
    db.add(connection)
    db.flush()  # assigns connection.Id without ending the transaction

    def _fetch(entry: dict) -> bytes:
        return googledrive_source.fetch_file_content(payload.access_token, entry)

    sync_result = _sync_and_record(
        db,
        connection=connection,
        uploaded_by=user,
        source_type="googledrive",
        target_files=target_files,
        skipped=skipped,
        fetch_fn=_fetch,
    )
    db.commit()
    db.refresh(connection)

    return ConnectResponse(connection=_to_item(connection, user.DisplayName), sync=sync_result)


@router.post("/onedrive/connect", response_model=ConnectResponse)
def connect_onedrive(
    payload: OneDriveConnectRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        item = onedrive_source.resolve_shared_folder(payload.access_token, payload.shared_link)
    except onedrive_source.OneDriveConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    drive_id = item["parentReference"]["driveId"]
    item_id = item["id"]
    base_path = item.get("name", "")

    try:
        target_files, skipped = onedrive_source.list_target_files(
            payload.access_token, drive_id, item_id, base_path
        )
    except onedrive_source.OneDriveConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    label = payload.display_label or base_path or "OneDrive folder"
    config = {"drive_id": drive_id, "item_id": item_id, "path": base_path}
    connection = SourceConnection(
        SourceType="onedrive",
        DisplayLabel=label,
        ConfigJson=json.dumps(config),
        Secret=None,
        CreatedBy=user.Id,
    )
    db.add(connection)
    db.flush()

    def _fetch(entry: dict) -> bytes:
        return onedrive_source.fetch_file_content(payload.access_token, drive_id, entry["item_id"])

    sync_result = _sync_and_record(
        db,
        connection=connection,
        uploaded_by=user,
        source_type="onedrive",
        target_files=target_files,
        skipped=skipped,
        fetch_fn=_fetch,
    )
    db.commit()
    db.refresh(connection)

    return ConnectResponse(connection=_to_item(connection, user.DisplayName), sync=sync_result)


@router.post("/connections/{connection_id}/sync", response_model=SyncResponse)
def sync_connection(
    connection_id: int,
    payload: SyncRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Connection not found.")

    config = json.loads(connection.ConfigJson)

    if connection.SourceType == "googledrive":
        if not payload.access_token:
            raise HTTPException(
                status_code=400,
                detail="access_token is required to sync a Google Drive connection.",
            )
        try:
            target_files, skipped = googledrive_source.list_target_files(
                payload.access_token, config["folder_id"], config["path"]
            )
        except googledrive_source.GoogleDriveConnectionError as exc:
            connection.LastSyncedAt = datetime.utcnow()
            connection.LastSyncStatus = "error"
            connection.LastSyncError = str(exc)
            db.commit()
            raise HTTPException(status_code=400, detail=str(exc))

        def _fetch(entry: dict) -> bytes:
            return googledrive_source.fetch_file_content(payload.access_token, entry)

        sync_result = _sync_and_record(
            db,
            connection=connection,
            uploaded_by=user,
            source_type="googledrive",
            target_files=target_files,
            skipped=skipped,
            fetch_fn=_fetch,
        )

    elif connection.SourceType == "onedrive":
        if not payload.access_token:
            raise HTTPException(
                status_code=400, detail="access_token is required to sync a OneDrive connection."
            )
        try:
            target_files, skipped = onedrive_source.list_target_files(
                payload.access_token, config["drive_id"], config["item_id"], config["path"]
            )
        except onedrive_source.OneDriveConnectionError as exc:
            connection.LastSyncedAt = datetime.utcnow()
            connection.LastSyncStatus = "error"
            connection.LastSyncError = str(exc)
            db.commit()
            raise HTTPException(status_code=400, detail=str(exc))

        def _fetch(entry: dict) -> bytes:
            return onedrive_source.fetch_file_content(
                payload.access_token, config["drive_id"], entry["item_id"]
            )

        sync_result = _sync_and_record(
            db,
            connection=connection,
            uploaded_by=user,
            source_type="onedrive",
            target_files=target_files,
            skipped=skipped,
            fetch_fn=_fetch,
        )

    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown source type '{connection.SourceType}'."
        )

    db.commit()
    return sync_result
