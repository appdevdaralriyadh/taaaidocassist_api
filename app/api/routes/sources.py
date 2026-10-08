"""
OneDrive / Google Drive connectors (spec §3.2, §7, §8 item 4 -- the final
phase; Google Drive replaces the connector originally built for GitHub).
Both connect flows are "paste a path/link, connect, and it reads +
ingests immediately" -- no folder-browsing UI. Every connection can be
re-synced later via POST /connections/{id}/sync, and Google Drive folders
can also sync automatically on a schedule (see below).

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

Google Drive service account: when GOOGLE_SERVICE_ACCOUNT_FILE is set in
.env, Google Drive is read as the app's own Google account instead of the
person's browser sign-in (no popup; no access_token needed). Each folder
is shared with that account's email as Viewer -- GET /googledrive/info
tells the page which mode is on and the email to share with. Existing
Google connections switch over automatically. Only in this mode can a
folder sync automatically (PUT /connections/{id}/schedule: every 1, 6, 12
or 24 hours); automatic syncs move files removed from the folder to the
Library's Deleted tab rather than deleting them for good.

OneDrive automatic sync (ENTRA_CLIENT_SECRET in .env): turning it on for a
folder stores a renewable Microsoft permission for the person doing it
(app/services/onedrive_auth.py) -- automatic syncs run as them. If
Microsoft later wants a new sign-in, the folder shows "Reconnect needed"
(POST /connections/{id}/reconnect); POST /onedrive/renew refreshes it
quietly whenever that person opens the OneDrive page.

DELETE /connections/{id}?documents=keep|delete removes a folder from sync,
either keeping its documents in the Library or moving them to Deleted.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.models import SourceConnection, User
from app.db.session import get_db
from app.schemas import (
    ApiTokenRequest,
    ConnectionRemovedResponse,
    ConnectStartResponse,
    GoogleDriveConnectRequest,
    GoogleDriveInfo,
    OneDriveConnectRequest,
    OneDriveInfo,
    RenewResponse,
    ScheduleRequest,
    SourceConnectionItem,
    SyncRequest,
    SyncStartResponse,
)
from app.services import cloud_sync, googledrive_source, jobs, onedrive_auth, onedrive_source

router = APIRouter()


def _to_item(
    db: Session, connection: SourceConnection, created_by_name: Optional[str]
) -> SourceConnectionItem:
    return SourceConnectionItem(
        id=connection.Id,
        source_type=connection.SourceType,
        display_label=connection.DisplayLabel,
        created_by=created_by_name,
        created_at=connection.CreatedAt,
        last_synced_at=connection.LastSyncedAt,
        last_sync_status=connection.LastSyncStatus,
        last_sync_error=connection.LastSyncError,
        schedule_hours=cloud_sync.schedule_hours(connection),
        next_sync_at=cloud_sync.next_sync_at(db, connection),
        auth_mode=cloud_sync.auth_mode(connection),
        document_count=cloud_sync.connection_document_count(db, connection),
        auto_sync_account=_auto_sync_account(db, connection),
        needs_reconnect=onedrive_auth.needs_reconnect(connection),
        reconnect_reason=onedrive_auth.reconnect_reason(connection),
    )


def _auto_sync_account(db: Session, connection: SourceConnection) -> Optional[str]:
    owner_id = onedrive_auth.owner_user_id(connection)
    if not owner_id:
        return None
    owner = db.get(User, owner_id)
    return owner.DisplayName if owner else None


def _grant(connection: SourceConnection, api_token: Optional[str], user: User) -> None:
    try:
        onedrive_auth.grant(connection, api_token=api_token or "", user_id=user.Id)
    except onedrive_auth.OneDriveAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
    return [_to_item(db, connection, display_name) for connection, display_name in rows]


@router.get("/googledrive/info", response_model=GoogleDriveInfo)
def googledrive_info(_user: User = Depends(get_current_user)):
    """How Google Drive is reached: the app's service account (and its email) or browser sign-in."""
    if not googledrive_source.service_account_configured():
        return GoogleDriveInfo(auth_mode="browser", schedule_options=[])
    email = googledrive_source.service_account_email()
    return GoogleDriveInfo(
        auth_mode="service_account",
        service_account_email=email,
        problem=None
        if email
        else (
            "The Google service account key file in GOOGLE_SERVICE_ACCOUNT_FILE (.env) can't be "
            "read -- check the path on the API server and restart the API."
        ),
        schedule_options=list(cloud_sync.SCHEDULE_OPTIONS),
    )


@router.get("/onedrive/info", response_model=OneDriveInfo)
def onedrive_info(_user: User = Depends(get_current_user)):
    """Whether OneDrive folders can sync automatically, and what the page needs for it."""
    problem = onedrive_auth.problem()
    return OneDriveInfo(
        automatic_available=problem is None,
        problem=problem,
        api_scope=onedrive_auth.api_scope() if problem is None else None,
        schedule_options=list(cloud_sync.SCHEDULE_OPTIONS) if problem is None else [],
    )


@router.post("/onedrive/renew", response_model=RenewResponse)
def renew_onedrive(
    payload: ApiTokenRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Quietly renews the stored OneDrive permission of every folder whose
    automatic sync runs as you, from your current sign-in (the page calls
    this when you open it) -- also clears "Reconnect needed" on them.
    """
    if not onedrive_auth.available():
        return RenewResponse(renewed=0)
    renewed, failed = 0, []
    for connection in db.query(SourceConnection).filter(SourceConnection.SourceType == "onedrive").all():
        if onedrive_auth.owner_user_id(connection) != user.Id:
            continue
        try:
            onedrive_auth.grant(connection, api_token=payload.api_token, user_id=user.Id)
            db.commit()
            renewed += 1
        except onedrive_auth.OneDriveAuthError as exc:
            db.rollback()
            failed.append(f"{connection.DisplayLabel}: {exc}")
    return RenewResponse(renewed=renewed, failed=failed)


@router.post("/connections/{connection_id}/reconnect", response_model=SourceConnectionItem)
def reconnect(
    connection_id: int,
    payload: ApiTokenRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Renews a OneDrive folder's stored permission from your sign-in; automatic sync runs as you from now on."""
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Connection not found.")
    if connection.SourceType != "onedrive":
        raise HTTPException(status_code=400, detail="Reconnect is only needed for OneDrive folders.")
    _grant(connection, payload.api_token, user)
    if connection.LastSyncStatus == "error" and (connection.LastSyncError or "").startswith("Microsoft asked"):
        connection.LastSyncError = None
    db.commit()
    db.refresh(connection)
    creator = db.get(User, connection.CreatedBy)
    return _to_item(db, connection, creator.DisplayName if creator else None)


def _source_fns(connection: SourceConnection, access_token: Optional[str]):
    """(list_fn, fetch_fn) for one connection (see cloud_sync.source_functions)."""
    try:
        return cloud_sync.source_functions(connection, access_token)
    except cloud_sync.SourceAccessError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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

    if googledrive_source.service_account_configured():
        token = googledrive_source.service_account_token
    elif payload.access_token:
        token = payload.access_token
    else:
        raise HTTPException(
            status_code=400,
            detail="Your Google sign-in is missing -- click Connect again and sign in when asked.",
        )
    try:
        folder = googledrive_source.resolve_folder(token, folder_id)
    except googledrive_source.GoogleDriveConnectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    label = payload.display_label or folder.get("name") or "Google Drive folder"
    config = {"folder_id": folder_id, "path": folder.get("name", "")}
    if payload.schedule_hours in cloud_sync.SCHEDULE_OPTIONS and payload.schedule_hours:
        config["schedule_hours"] = payload.schedule_hours
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
    return ConnectStartResponse(connection=_to_item(db, connection, user.DisplayName), job_id=job_id)


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
    if payload.schedule_hours in cloud_sync.SCHEDULE_OPTIONS and payload.schedule_hours:
        config["schedule_hours"] = payload.schedule_hours
    connection = SourceConnection(
        SourceType="onedrive",
        DisplayLabel=label,
        ConfigJson=json.dumps(config),
        Secret=None,
        CreatedBy=user.Id,
    )
    if config.get("schedule_hours"):
        # automatic sync from the start: keep a renewable permission for you
        # (checked before anything is saved)
        _grant(connection, payload.api_token, user)
    db.add(connection)
    db.flush()

    list_fn, fetch_fn = _source_fns(connection, payload.access_token)
    job_id = jobs.start_sync(db, connection=connection, user=user, list_fn=list_fn, fetch_fn=fetch_fn)
    db.refresh(connection)
    return ConnectStartResponse(connection=_to_item(db, connection, user.DisplayName), job_id=job_id)


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
    list_fn, fetch_fn = _source_fns(connection, payload.access_token)
    job_id = jobs.start_sync(db, connection=connection, user=user, list_fn=list_fn, fetch_fn=fetch_fn)
    return SyncStartResponse(connection_id=connection.Id, job_id=job_id)


@router.put("/connections/{connection_id}/schedule", response_model=SourceConnectionItem)
def set_schedule(
    connection_id: int,
    payload: ScheduleRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Turns automatic sync on (every 1, 6, 12 or 24 hours) or off (0) for one
    folder. OneDrive: turning it on (with api_token) stores a renewable
    permission for you -- automatic syncs then run as you; turning it off
    forgets that permission.
    """
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Connection not found.")
    if payload.every_hours not in cloud_sync.SCHEDULE_OPTIONS:
        raise HTTPException(
            status_code=400,
            detail="Choose Off, or every 1, 6, 12 or 24 hours.",
        )
    if connection.SourceType == "onedrive":
        if payload.every_hours:
            if not onedrive_auth.available():
                raise HTTPException(status_code=400, detail=onedrive_auth.problem())
            if payload.api_token:
                _grant(connection, payload.api_token, user)
            elif not onedrive_auth.has_permission(connection):
                raise HTTPException(
                    status_code=400,
                    detail="Your sign-in wasn't sent -- reload the page and try again.",
                )
        else:
            onedrive_auth.clear(connection)
    elif payload.every_hours and not cloud_sync.can_run_unattended(connection):
        raise HTTPException(
            status_code=400,
            detail=(
                "Automatic sync needs the app's Google service account (GOOGLE_SERVICE_ACCOUNT_FILE "
                "in .env) -- it isn't available for Google Drive with browser sign-in."
            ),
        )
    cloud_sync.set_schedule_hours(connection, payload.every_hours)
    db.commit()
    db.refresh(connection)
    creator = db.get(User, connection.CreatedBy)
    return _to_item(db, connection, creator.DisplayName if creator else None)


@router.delete("/connections/{connection_id}", response_model=ConnectionRemovedResponse)
def remove_connection(
    connection_id: int,
    documents: str = Query(..., description="'keep' or 'delete' -- what happens to its documents"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Stops syncing a folder; its documents are kept in the Library or moved to Deleted."""
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Connection not found.")
    result = cloud_sync.remove_connection(db, connection=connection, user=user, documents=documents)
    return ConnectionRemovedResponse(connection_id=connection_id, **result)
