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

Connect checks the folder and saves the connection; the sync itself --
on connect and on every Sync Now -- runs in the background
(app/services/jobs.py, using app/services/cloud_sync.py's rules: each
cloud file is tracked by its own cloud ID, unchanged files are skipped,
changed ones become new versions with History, renames/moves are
followed, new files get the local-upload version rules, and documents
whose file left the folder are permanently deleted). The page follows
its progress via GET /api/documents/jobs/{job_id}. One bad file is
recorded and skipped rather than aborting the whole sync.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.models import SourceConnection, User
from app.db.session import get_db
from app.schemas import (
    ConnectStartResponse,
    GoogleDriveConnectRequest,
    OneDriveConnectRequest,
    SourceConnectionItem,
    SyncRequest,
    SyncStartResponse,
)
from app.services import googledrive_source, jobs, onedrive_source

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


def _source_fns(connection: SourceConnection, access_token: str):
    """(list_fn, fetch_fn) for one connection, bound to the caller's token."""
    config = json.loads(connection.ConfigJson)
    if connection.SourceType == "googledrive":
        def list_fn(on_progress=None):
            return googledrive_source.list_target_files(
                access_token, config["folder_id"], config["path"], on_progress=on_progress
            )

        def fetch_fn(entry: dict) -> bytes:
            return googledrive_source.fetch_file_content(access_token, entry)

        return list_fn, fetch_fn
    if connection.SourceType == "onedrive":
        def list_fn(on_progress=None):
            return onedrive_source.list_target_files(
                access_token, config["drive_id"], config["item_id"], config["path"],
                on_progress=on_progress,
            )

        def fetch_fn(entry: dict) -> bytes:
            return onedrive_source.fetch_file_content(
                access_token, config["drive_id"], entry["item_id"]
            )

        return list_fn, fetch_fn
    raise HTTPException(status_code=400, detail=f"Unknown source type '{connection.SourceType}'.")


@router.post("/googledrive/connect", response_model=ConnectStartResponse)
def connect_googledrive(
    payload: GoogleDriveConnectRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Checks the folder, saves the connection, and starts its first sync in the background."""
    try:
        folder_id = googledrive_source.parse_drive_path(payload.folder_path)
    except googledrive_source.GoogleDrivePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        folder = googledrive_source.resolve_folder(payload.access_token, folder_id)
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

    list_fn, fetch_fn = _source_fns(connection, payload.access_token)
    job_id = jobs.start_sync(db, connection=connection, user=user, list_fn=list_fn, fetch_fn=fetch_fn)
    db.refresh(connection)
    return ConnectStartResponse(connection=_to_item(connection, user.DisplayName), job_id=job_id)


@router.post("/onedrive/connect", response_model=ConnectStartResponse)
def connect_onedrive(
    payload: OneDriveConnectRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Checks the folder, saves the connection, and starts its first sync in the background."""
    try:
        item = onedrive_source.resolve_shared_folder(payload.access_token, payload.shared_link)
    except onedrive_source.OneDriveConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    drive_id = item["parentReference"]["driveId"]
    item_id = item["id"]
    base_path = item.get("name", "")

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

    list_fn, fetch_fn = _source_fns(connection, payload.access_token)
    job_id = jobs.start_sync(db, connection=connection, user=user, list_fn=list_fn, fetch_fn=fetch_fn)
    db.refresh(connection)
    return ConnectStartResponse(connection=_to_item(connection, user.DisplayName), job_id=job_id)


@router.post("/connections/{connection_id}/sync", response_model=SyncStartResponse)
def sync_connection(
    connection_id: int,
    payload: SyncRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Starts a sync in the background; progress via GET /api/documents/jobs/{job_id}."""
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Connection not found.")
    if not payload.access_token:
        raise HTTPException(
            status_code=400,
            detail="Your sign-in for this folder is missing -- click Sync Now again and sign in when asked.",
        )
    list_fn, fetch_fn = _source_fns(connection, payload.access_token)
    job_id = jobs.start_sync(db, connection=connection, user=user, list_fn=list_fn, fetch_fn=fetch_fn)
    return SyncStartResponse(connection_id=connection.Id, job_id=job_id)
