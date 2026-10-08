"""
OneDrive/SharePoint connector (spec §3.2, §8 item 4): paste a shared
folder link. The frontend already holds a live MSAL session (the same
Entra app registration used for login, spec §3.1), so it fetches a
short-lived Microsoft Graph access token for the Files.Read.All scope and
sends it along with the link on every call -- nothing OneDrive-specific
is stored server-side except the resolved folder reference itself
(drive_id/item_id/path), never a token. That's not a shortcut: a SPA
(public client) app registration can't refresh a Graph token server-side
without a client secret, so there is no long-lived credential to persist
for OneDrive the way GitHub's PAT is persisted.

graph.microsoft.com is unreachable from the sandbox this was built in
(same block as huggingface.co/api.openai.com) -- this module is verified
via mocked Graph responses against the real route/service code. A live
Graph token and the Files.Read.All delegated permission added to the Entra
app registration are both required before this works end to end.
Files.Read.All (rather than Files.Read) is what lets a person connect a
folder someone else shared with them, not only folders in their own
OneDrive.
"""

import base64
import re
from urllib.parse import urlparse
from dataclasses import dataclass

import requests

from app.config import settings
from app.services.parsing import SUPPORTED_EXTENSIONS

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class OneDriveConnectionError(RuntimeError):
    pass


@dataclass
class OneDriveFileRef:
    item_id: str
    path: str
    size: int


def _encode_share_id(shared_link: str) -> str:
    # https://learn.microsoft.com/en-us/graph/api/shares-get -- a sharing
    # URL becomes an addressable "shares" resource via base64url(url),
    # prefixed with "u!" and with the "=" padding stripped.
    b64 = base64.urlsafe_b64encode(shared_link.strip().encode("utf-8")).decode("ascii")
    return "u!" + b64.rstrip("=")


def _headers(access_token) -> dict:
    # a token string (browser sign-in), or a function returning a current
    # token (automatic sync -- app/services/onedrive_auth.py)
    token = access_token() if callable(access_token) else access_token
    return {"Authorization": f"Bearer {token}"}


_LINK_HELP = (
    "A OneDrive/SharePoint shared-folder link looks like "
    "https://yourcompany-my.sharepoint.com/:f:/g/... or https://1drv.ms/f/... -- in OneDrive, "
    "open the folder's Share menu and choose Copy link."
)
_ONEDRIVE_HOSTS = ("sharepoint.com", "1drv.ms", "onedrive.live.com", "onedrive.com")
# SharePoint/OneDrive short links say what they point to: /:f:/ is a
# folder; /:w:/ Word, /:x:/ Excel, /:p:/ PowerPoint, /:b:/ PDF, /:t:/ text,
# /:i:/ image, /:v:/ video, /:u:/ other file.
_FILE_LINK_CODES = {"w", "x", "p", "b", "t", "i", "v", "u"}


def validate_shared_link(shared_link: str) -> str:
    """
    Checks a pasted link looks like a OneDrive/SharePoint folder link before
    calling Microsoft, so the person gets a message that says what's wrong.
    Returns the cleaned link; raises OneDriveConnectionError otherwise.
    """
    link = (shared_link or "").strip()
    if not link:
        raise OneDriveConnectionError("Paste a OneDrive or SharePoint shared-folder link.")
    lowered = link.lower()
    if "drive.google.com" in lowered or "docs.google.com" in lowered:
        raise OneDriveConnectionError(
            "That's a Google Drive link -- connect it on the Google Drive page instead."
        )
    parsed = urlparse(link)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OneDriveConnectionError(
            "That isn't a valid link -- paste the full shared-folder link (it starts with "
            "https://). " + _LINK_HELP
        )
    host = parsed.netloc.lower().split(":")[0]
    if not any(host == h or host.endswith("." + h) for h in _ONEDRIVE_HOSTS):
        raise OneDriveConnectionError(
            f"That isn't a valid OneDrive link -- '{host}' isn't a OneDrive or SharePoint "
            "address. " + _LINK_HELP
        )
    code = re.match(r"^/:([a-z]):/", parsed.path.lower())
    if code and code.group(1) in _FILE_LINK_CODES:
        raise OneDriveConnectionError(
            "That's a link to a single file, not a folder. Share the folder that contains it "
            "and paste that link instead."
        )
    return link


def _raise_for_response(resp: requests.Response, action: str) -> None:
    if resp.status_code == 401:
        raise OneDriveConnectionError(
            "Your Microsoft sign-in for OneDrive has expired or doesn't include file access -- "
            "click Connect (or Sync Now) again and sign in when asked."
        )
    if resp.status_code == 403:
        raise OneDriveConnectionError(
            f"Access denied while {action} -- your Microsoft account doesn't have access to "
            "this folder. Ask the folder's owner to share it with you, then try again. If it is "
            "already shared with you, the app may not have permission to read shared files yet: "
            "an admin needs to grant consent for 'Files.Read.All' on the app in Microsoft Entra."
        )
    if resp.status_code == 404:
        raise OneDriveConnectionError(f"Not found while {action} (404 from Microsoft Graph).")
    if not resp.ok:
        raise OneDriveConnectionError(
            f"Microsoft Graph error while {action}: {resp.status_code} {resp.text[:200]}"
        )


def resolve_shared_folder(access_token: str, shared_link: str) -> dict:
    """
    Resolves a pasted sharing link to a Graph driveItem. Raises
    OneDriveConnectionError with a clean message on any failure, or if the
    link points to a file rather than a folder.
    """
    shared_link = validate_shared_link(shared_link)

    share_id = _encode_share_id(shared_link)
    url = f"{_GRAPH_BASE}/shares/{share_id}/driveItem"
    resp = requests.get(url, headers=_headers(access_token), timeout=30)
    if resp.status_code in (400, 404):
        raise OneDriveConnectionError(
            "That OneDrive link doesn't work -- the folder may have been deleted or moved, the "
            "sharing link may have been turned off, or the link was copied incompletely. Copy a "
            "fresh link from the folder's Share menu and try again."
        )
    _raise_for_response(resp, "resolving the shared link")

    item = resp.json()
    if "folder" not in item:
        raise OneDriveConnectionError("That link points to a file, not a folder. Paste a link to a folder.")
    return item  # has id, parentReference.driveId, name, webUrl, folder{...}


def _list_children(access_token: str, drive_id: str, item_id: str) -> list[dict]:
    children: list[dict] = []
    url = (
        f"{_GRAPH_BASE}/drives/{drive_id}/items/{item_id}/children"
        "?$select=id,name,size,file,folder,cTag,eTag,lastModifiedDateTime"
    )
    while url:
        resp = requests.get(url, headers=_headers(access_token), timeout=30)
        _raise_for_response(resp, "listing folder contents")
        body = resp.json()
        children.extend(body.get("value", []))
        url = body.get("@odata.nextLink")
    return children


def _extension(name: str) -> str:
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def list_target_files(
    access_token: str,
    drive_id: str,
    item_id: str,
    base_path: str,
    on_progress=None,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """
    Recursively walks the folder (Graph has no single-call recursive
    listing the way GitHub's tree API does, so this queues and walks
    subfolders one Graph call at a time). Returns (target_files, skipped)
    where target_files is [{"item_id", "external_id", "path", "size",
    "version_tag", "modified_at"}, ...] for every supported, within-size-cap
    file, and skipped is [(path, reason, external_id), ...] for everything
    else (wrong extension, too large). base_path is prefixed onto every
    entry purely for a readable filename/SourcePath. version_tag is the
    cTag, which changes only when the file's CONTENT changes (eTag also
    changes on a rename), so sync can skip unchanged files without
    downloading them.
    """
    target_files: list[dict] = []
    skipped: list[tuple[str, str, str]] = []
    stack = [(item_id, base_path)]

    folders_done = 0
    while stack:
        if on_progress is not None:
            # (files found so far, folders read so far) -- for the sync's
            # progress line; may raise to stop listing (Cancel)
            on_progress(len(target_files) + len(skipped), folders_done)
        folders_done += 1
        current_id, current_path = stack.pop()
        for child in _list_children(access_token, drive_id, current_id):
            name = child["name"]
            child_path = f"{current_path}/{name}" if current_path else name

            if "folder" in child:
                stack.append((child["id"], child_path))
                continue
            if "file" not in child:
                continue  # some other item type (package, etc.) -- skip silently

            size = child.get("size", 0)
            ext = _extension(name)
            if ext not in SUPPORTED_EXTENSIONS:
                skipped.append((child_path, "unsupported file type", child["id"]))
                continue
            if size > _MAX_BYTES:
                skipped.append(
                    (child_path, f"exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit", child["id"])
                )
                continue

            target_files.append(
                {
                    "item_id": child["id"],
                    "external_id": child["id"],
                    "path": child_path,
                    "size": size,
                    "version_tag": child.get("cTag") or child.get("eTag"),
                    "modified_at": child.get("lastModifiedDateTime"),
                }
            )

    return target_files, skipped


def fetch_file_content(access_token: str, drive_id: str, item_id: str) -> bytes:
    url = f"{_GRAPH_BASE}/drives/{drive_id}/items/{item_id}/content"
    resp = requests.get(url, headers=_headers(access_token), timeout=60)
    _raise_for_response(resp, "downloading a file")
    return resp.content
