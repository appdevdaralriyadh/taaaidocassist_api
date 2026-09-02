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
onedrive_source.py uses for Microsoft Graph) -- no Google SDK dependency
needed at all for this token-based approach.
"""

import re
from typing import Optional

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


class GoogleDrivePathError(ValueError):
    """The pasted folder link/ID couldn't be parsed."""


class GoogleDriveConnectionError(RuntimeError):
    """The parsed folder couldn't be reached with the given access token."""


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

    match = _FOLDER_ID_IN_URL.search(text)
    if match:
        return match.group(1)

    match = _ID_QUERY_PARAM.search(text)
    if match:
        return match.group(1)

    if text.lower().startswith(("http://", "https://")):
        raise GoogleDrivePathError(
            "Couldn't find a folder ID in that link -- paste the full "
            "folder URL from Google Drive's 'Get link' option."
        )

    # Bare ID shorthand -- Drive IDs never contain whitespace or slashes.
    if " " in text or "/" in text:
        raise GoogleDrivePathError("Expected a Google Drive folder link or a bare folder ID.")
    return text


def _headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


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


def _raise_for_response(resp: requests.Response, action: str) -> None:
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
            f"Not found while {action} -- check the folder link/ID, and that the "
            "Google account you're signed in as has access to it."
        )
    if not resp.ok:
        raise GoogleDriveConnectionError(
            f"Google Drive API error while {action}: {resp.status_code} {resp.text[:300]}"
        )


def resolve_folder(access_token: str, folder_id: str) -> dict:
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
    _raise_for_response(resp, "resolving the folder")
    item = resp.json()
    if item.get("mimeType") != "application/vnd.google-apps.folder":
        raise GoogleDriveConnectionError("That ID points to a file, not a folder.")
    return item


def _list_children(access_token: str, folder_id: str) -> list[dict]:
    children: list[dict] = []
    params = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "fields": "nextPageToken, files(id,name,mimeType,size)",
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
        _raise_for_response(resp, "listing folder contents")
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
    access_token: str, folder_id: str, base_path: str
) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    Recursively walks the folder (Drive has no single-call recursive
    listing, the same constraint as OneDrive's Graph API -- walked one
    folder at a time). Returns (target_files, skipped) where target_files
    is [{"file_id", "path", "size", "mime_type"}, ...] for every
    supported, within-size-cap file, and skipped is [(path, reason), ...]
    for everything else (wrong/unexportable type, too large).
    """
    target_files: list[dict] = []
    skipped: list[tuple[str, str]] = []
    stack = [(folder_id, base_path)]

    while stack:
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
                    }
                )
                continue

            if mime_type.startswith(_GOOGLE_APPS_PREFIX):
                skipped.append((child_path, "unsupported Google Workspace file type"))
                continue

            ext = _extension(name)
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append((child_path, "unsupported file type"))
                continue

            size = int(child.get("size") or 0)
            if size > _MAX_BYTES:
                skipped.append((child_path, f"exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"))
                continue

            target_files.append(
                {"file_id": child["id"], "path": child_path, "size": size, "mime_type": mime_type}
            )

    return target_files, skipped


def fetch_file_content(access_token: str, entry: dict) -> bytes:
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
    _raise_for_response(resp, "downloading a file")
    return resp.content
