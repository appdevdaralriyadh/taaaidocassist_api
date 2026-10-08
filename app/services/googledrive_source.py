"""
Google Drive connector (spec §3.2, §8 item 4 -- replaces the connector
originally built for GitHub). Paste a Drive folder link or a bare folder
ID -- every supported file at or under that folder (recursively) is
pulled and ingested through the same pipeline as a local upload
(app/services/ingestion.py).

Auth is per-user Google sign-in, acquired in the browser via Google
Identity Services (the frontend's googledrive.component.ts) -- the same
"acquire a short-lived token fresh before every call, never persist it"
shape app/services/onedrive_source.py already uses for Microsoft Graph.
Every function here takes `access_token` as an explicit parameter rather
than deriving one internally, and nothing about Google auth is stored
server-side: no credential file, no refresh token, no per-connection
secret (DarAI_SourceConnections.Secret stays NULL for source_type=
'googledrive', same as onedrive).

Practical effect of this choice: a person can only pull in what their own
Google account can actually see in Drive -- there's no shared "bot"
identity with its own separate access grants to reason about, and if
their browser's Google session lapses, the next connect/sync attempt
simply re-prompts sign-in (handled entirely in the frontend; this module
just receives whatever token it's given and uses it as-is).

Talks to the Drive API v3 directly over REST (the same style
onedrive_source.py uses for Microsoft Graph).

Service account (when GOOGLE_SERVICE_ACCOUNT_FILE is set in .env): the
app reads Drive as its own Google service account instead of the person's
browser sign-in. Each folder is shared with the service account's email
as Viewer, and the app gets its own short-lived tokens from the key file
(google-auth, read-only Drive scope; cached and renewed automatically).
That's what lets folders sync on a schedule with nobody signed in. The
key file is only ever read here, at run time -- it is never stored in the
database, sent to the browser, or logged.

`access_token` below is either a token string (browser sign-in) or a
function returning a current token (service account) -- the function is
called for every request, so a long sync never runs on an expired token.
"""

import json
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlparse

import requests

from app.config import settings
from app.services.parsing import SUPPORTED_EXTENSIONS

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Google Docs/Sheets aren't real downloadable files -- they only exist as
# native Google formats and have to be *exported* to a format
# app/services/parsing.py can read (a plain files.get?alt=media download
# returns 403 for these). Slides has no entry: this app has no .pptx
# parser, so a Slides file is skipped with a clear reason rather than
# silently mis-exported. Maps mimeType -> (export mimeType, file extension
# to append -- see the comment in list_target_files for why that matters).
_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}
_GOOGLE_APPS_PREFIX = "application/vnd.google-apps."

_FOLDER_ID_IN_URL = re.compile(r"/folders/([a-zA-Z0-9_-]+)")
_ID_QUERY_PARAM = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


_LINK_HELP = (
    "A Google Drive folder link looks like https://drive.google.com/drive/folders/... -- in "
    "Google Drive, right-click the folder, choose Share, then Copy link."
)


class GoogleDrivePathError(ValueError):
    """The pasted folder link/ID couldn't be parsed."""


class GoogleDriveConnectionError(RuntimeError):
    """The parsed folder couldn't be reached with the given access token."""


# ---------------------------------------------------------------------------
# Service account
# ---------------------------------------------------------------------------

_DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
_sa_lock = threading.Lock()
_sa_credentials = None
_sa_email: Optional[str] = None

TokenSource = Union[str, Callable[[], str]]


def service_account_configured() -> bool:
    """True when GOOGLE_SERVICE_ACCOUNT_FILE is set (whether or not the file is usable)."""
    return bool((settings.GOOGLE_SERVICE_ACCOUNT_FILE or "").strip())


def _key_path() -> Path:
    return Path(settings.GOOGLE_SERVICE_ACCOUNT_FILE.strip().strip('"'))


def service_account_email() -> Optional[str]:
    """The email folders must be shared with, or None when not configured / unreadable."""
    global _sa_email
    if not service_account_configured():
        return None
    if _sa_email is None:
        try:
            with open(_key_path(), encoding="utf-8") as fh:
                _sa_email = json.load(fh).get("client_email") or None
        except (OSError, ValueError):
            return None
    return _sa_email


def service_account_token() -> str:
    """A current access token for the service account (cached, renewed shortly before expiry)."""
    global _sa_credentials
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as exc:  # pragma: no cover -- depends on the install
        raise GoogleDriveConnectionError(
            "The Google service account needs the 'google-auth' package -- run "
            "'pip install google-auth' on the API server and restart it."
        ) from exc
    with _sa_lock:
        if _sa_credentials is None:
            path = _key_path()
            if not path.is_file():
                raise GoogleDriveConnectionError(
                    "The Google service account key file wasn't found at the path in "
                    "GOOGLE_SERVICE_ACCOUNT_FILE (.env). Check the path and restart the API."
                )
            try:
                _sa_credentials = service_account.Credentials.from_service_account_file(
                    str(path), scopes=[_DRIVE_READONLY]
                )
            except (OSError, ValueError) as exc:
                raise GoogleDriveConnectionError(
                    "The Google service account key file couldn't be read -- make sure "
                    "GOOGLE_SERVICE_ACCOUNT_FILE points to the JSON key downloaded from Google "
                    "Cloud Console."
                ) from exc
        if not _sa_credentials.valid:
            try:
                _sa_credentials.refresh(Request())
            except Exception as exc:  # noqa: BLE001 -- google.auth.exceptions.*, network errors
                raise GoogleDriveConnectionError(
                    "Google didn't accept the app's service account sign-in -- the key may have "
                    "been deleted or disabled in Google Cloud Console, or this server can't reach "
                    f"Google ({str(exc)[:200]})."
                ) from exc
        return _sa_credentials.token


def _share_hint() -> str:
    email = service_account_email()
    who = email or "the app's Google service account"
    return (
        f"In Google Drive, share the folder with {who} as Viewer (Share > add that email > "
        "Viewer), then try again."
    )


def parse_drive_path(raw: str) -> str:
    """
    Accepts:
      - https://drive.google.com/drive/folders/{folderId}[?usp=sharing...]
      - https://drive.google.com/drive/u/0/folders/{folderId}
      - https://drive.google.com/open?id={folderId}
      - a bare folder ID (no slashes/spaces)
    Returns the bare folder ID.
    """
    text = (raw or "").strip()
    if not text:
        raise GoogleDrivePathError("Paste a Google Drive folder link or folder ID.")

    lowered = text.lower()
    if "sharepoint.com" in lowered or "1drv.ms" in lowered or "onedrive.live.com" in lowered:
        raise GoogleDrivePathError(
            "That's a OneDrive / SharePoint link -- connect it on the OneDrive page instead."
        )
    if lowered.startswith(("http://", "https://")):
        host = urlparse(text).netloc.lower().split(":")[0]
        if host not in ("drive.google.com", "docs.google.com"):
            raise GoogleDrivePathError(
                f"That isn't a valid Google Drive link -- '{host}' isn't a Google Drive address. "
                + _LINK_HELP
            )
        if "/file/d/" in lowered or "docs.google.com/" in lowered:
            raise GoogleDrivePathError(
                "That's a link to a single file, not a folder. Share the folder that contains "
                "it and paste that link instead."
            )

    match = _FOLDER_ID_IN_URL.search(text)
    if match:
        return match.group(1)

    match = _ID_QUERY_PARAM.search(text)
    if match:
        return match.group(1)

    if text.lower().startswith(("http://", "https://")):
        raise GoogleDrivePathError(
            "That isn't a valid Google Drive folder link -- no folder ID was found in it. "
            + _LINK_HELP
        )

    # Bare ID shorthand -- Drive IDs never contain whitespace or slashes.
    if " " in text or "/" in text:
        raise GoogleDrivePathError(
            "That isn't a valid Google Drive folder link or folder ID. " + _LINK_HELP
        )
    return text


def _headers(access_token: TokenSource) -> dict:
    token = access_token() if callable(access_token) else access_token
    return {"Authorization": f"Bearer {token}"}


def _google_error_reason(resp: requests.Response) -> Optional[str]:
    """
    Pulls Google's own machine-readable reason out of a Drive API error
    body, e.g. "accessNotConfigured" (the API isn't enabled on the
    project), "insufficientFilePermissions"/"forbidden" (a real
    permissions gap), "dailyLimitExceededUnreg", etc. Returns None if the
    body isn't the JSON shape Google normally sends (so callers always
    have resp.text as a fallback).
    """
    try:
        body = resp.json()
    except ValueError:
        return None
    error = body.get("error") or {}
    errors = error.get("errors") or []
    if errors and isinstance(errors, list) and errors[0].get("reason"):
        return errors[0]["reason"]
    return error.get("status") or error.get("message")


def _raise_for_response(resp: requests.Response, action: str, *, as_service: bool = False) -> None:
    if as_service and resp.status_code in (401, 403, 404):
        reason = _google_error_reason(resp)
        if resp.status_code == 403 and reason == "accessNotConfigured":
            raise GoogleDriveConnectionError(
                f"Access denied while {action} -- Google says the Drive API isn't enabled for the "
                "service account's Google Cloud project. In Google Cloud Console, open 'APIs & "
                "Services' > 'Library', search for 'Google Drive API', and click Enable."
            )
        if resp.status_code == 401:
            raise GoogleDriveConnectionError(
                f"Google didn't accept the app's service account while {action} -- the key may "
                "have been deleted or disabled in Google Cloud Console."
            )
        if resp.status_code == 403:
            raise GoogleDriveConnectionError(
                f"The app's Google account isn't allowed to read this while {action}"
                + (f" (Google said: {reason})" if reason else "")
                + ". "
                + _share_hint()
            )
        raise GoogleDriveConnectionError(
            f"Google Drive couldn't find this folder for the app while {action} -- it isn't "
            "shared with the app's Google account yet, or it was deleted or moved. "
            + _share_hint()
        )
    if resp.status_code == 401:
        # A stale/expired browser-acquired token -- distinct from 403
        # (a genuinely valid sign-in that just lacks access to this
        # specific file/folder). The frontend should prompt sign-in again
        # and retry, not show this as a permissions problem.
        raise GoogleDriveConnectionError(
            "Your Google sign-in has expired -- please sign in with Google again and retry."
        )
    if resp.status_code == 403:
        reason = _google_error_reason(resp)
        if reason == "accessNotConfigured":
            # The single most common cause of a 403 here that has nothing
            # to do with file sharing at all: the Google Cloud project
            # behind the OAuth Client ID has never had the Drive API
            # switched on. Sharing/ownership is irrelevant until this is
            # fixed -- every Drive API call for every account will 403
            # the same way.
            raise GoogleDriveConnectionError(
                "Access denied while "
                f"{action} -- Google says the Drive API isn't enabled for this "
                "app's Google Cloud project. In Google Cloud Console, open "
                "'APIs & Services' > 'Library', search for 'Google Drive API', "
                "and click Enable, then try again."
            )
        raise GoogleDriveConnectionError(
            f"Access denied while {action}"
            + (f" (Google said: {reason})" if reason else f": {resp.text[:300]}")
            + ". Either the signed-in Google account genuinely doesn't have "
            "access to this file/folder, or the sign-in popup didn't actually "
            "grant Drive permission -- try revoking this app's access at "
            "https://myaccount.google.com/permissions and reconnecting so it "
            "asks for permission again, and double-check the Drive scope is "
            "listed under the OAuth consent screen's Data Access tab."
        )
    if resp.status_code == 404:
        raise GoogleDriveConnectionError(
            f"Not found while {action} -- the folder may have been deleted or moved, the link "
            "may be incomplete, or the Google account you signed in with doesn't have access to "
            "it. Copy a fresh link from the folder's Share menu, or ask its owner to share it "
            "with that account."
        )
    if not resp.ok:
        raise GoogleDriveConnectionError(
            f"Google Drive API error while {action}: {resp.status_code} {resp.text[:300]}"
        )


def resolve_folder(access_token: TokenSource, folder_id: str) -> dict:
    """
    Confirms `folder_id` exists, is reachable with this access token, and
    is actually a folder. Raises GoogleDriveConnectionError with a clean
    message otherwise. Returns the Drive file resource (id, name,
    mimeType).
    """
    url = f"{_DRIVE_API}/files/{folder_id}"
    resp = requests.get(
        url,
        headers=_headers(access_token),
        params={"fields": "id,name,mimeType", "supportsAllDrives": "true"},
        timeout=30,
    )
    _raise_for_response(resp, "resolving the folder", as_service=callable(access_token))
    item = resp.json()
    if item.get("mimeType") != "application/vnd.google-apps.folder":
        raise GoogleDriveConnectionError("That ID points to a file, not a folder.")
    return item


def _list_children(access_token: TokenSource, folder_id: str) -> list[dict]:
    children: list[dict] = []
    params = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "fields": "nextPageToken, files(id,name,mimeType,size,md5Checksum,version,modifiedTime)",
        "pageSize": 1000,
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }
    page_token = None
    while True:
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(
            f"{_DRIVE_API}/files", headers=_headers(access_token), params=params, timeout=30
        )
        _raise_for_response(resp, "listing folder contents", as_service=callable(access_token))
        body = resp.json()
        children.extend(body.get("files", []))
        page_token = body.get("nextPageToken")
        if not page_token:
            break
    return children


def _extension(name: str) -> str:
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def list_target_files(
    access_token: TokenSource, folder_id: str, base_path: str, on_progress=None
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """
    Recursively walks the folder (Drive has no single-call recursive
    listing, the same constraint as OneDrive's Graph API -- walked one
    folder at a time). Returns (target_files, skipped) where target_files
    is [{"file_id", "external_id", "path", "size", "mime_type",
    "version_tag", "modified_at"}, ...] for every supported,
    within-size-cap file, and skipped is [(path, reason, external_id), ...]
    for everything else (wrong/unexportable type, too large). version_tag
    is the file's md5Checksum, or "v<version>" for native Google
    Docs/Sheets (which have no checksum) -- sync compares it to skip
    unchanged files without downloading them.
    """
    target_files: list[dict] = []
    skipped: list[tuple[str, str, str]] = []
    stack = [(folder_id, base_path)]

    folders_done = 0
    while stack:
        if on_progress is not None:
            # (files found so far, folders read so far) -- for the sync's
            # progress line; may raise to stop listing (Cancel)
            on_progress(len(target_files) + len(skipped), folders_done)
        folders_done += 1
        current_id, current_path = stack.pop()
        for child in _list_children(access_token, current_id):
            name = child["name"]
            mime_type = child["mimeType"]
            child_path = f"{current_path}/{name}" if current_path else name

            if mime_type == "application/vnd.google-apps.folder":
                stack.append((child["id"], child_path))
                continue

            if mime_type in _EXPORT_MIME_TYPES:
                _export_mime, export_ext = _EXPORT_MIME_TYPES[mime_type]
                target_files.append(
                    {
                        "file_id": child["id"],
                        # Google Docs/Sheets have no file extension of
                        # their own in Drive's naming -- append the
                        # export format's extension so the downstream
                        # parser (which dispatches purely on filename
                        # extension, see app/services/parsing.py) picks
                        # the right one, and so the ingested document's
                        # stored filename actually reflects what got
                        # saved.
                        "path": f"{child_path}{export_ext}",
                        "size": 0,  # unknown pre-export; native Docs/Sheets report no size
                        "mime_type": mime_type,
                        "external_id": child["id"],
                        "version_tag": f"v{child['version']}" if child.get("version") else None,
                        "modified_at": child.get("modifiedTime"),
                    }
                )
                continue

            if mime_type.startswith(_GOOGLE_APPS_PREFIX):
                skipped.append((child_path, "unsupported Google Workspace file type", child["id"]))
                continue

            ext = _extension(name)
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append((child_path, "unsupported file type", child["id"]))
                continue

            size = int(child.get("size") or 0)
            if size > _MAX_BYTES:
                skipped.append(
                    (child_path, f"exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit", child["id"])
                )
                continue

            target_files.append(
                {
                    "file_id": child["id"],
                    "external_id": child["id"],
                    "path": child_path,
                    "size": size,
                    "mime_type": mime_type,
                    "version_tag": child.get("md5Checksum")
                    or (f"v{child['version']}" if child.get("version") else None),
                    "modified_at": child.get("modifiedTime"),
                }
            )

    return target_files, skipped


def fetch_file_content(access_token: TokenSource, entry: dict) -> bytes:
    """
    Downloads a regular file directly, or exports a native Google Docs/
    Sheets file to the format chosen in _EXPORT_MIME_TYPES. `entry` is one
    item from list_target_files().
    """
    file_id = entry["file_id"]
    mime_type = entry["mime_type"]

    if mime_type in _EXPORT_MIME_TYPES:
        export_mime, _ext = _EXPORT_MIME_TYPES[mime_type]
        url = f"{_DRIVE_API}/files/{file_id}/export"
        resp = requests.get(
            url, headers=_headers(access_token), params={"mimeType": export_mime}, timeout=60
        )
    else:
        url = f"{_DRIVE_API}/files/{file_id}"
        resp = requests.get(
            url,
            headers=_headers(access_token),
            params={"alt": "media", "supportsAllDrives": "true"},
            timeout=60,
        )
    _raise_for_response(resp, "downloading a file", as_service=callable(access_token))
    return resp.content
