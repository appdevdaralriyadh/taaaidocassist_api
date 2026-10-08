"""
OneDrive automatic sync: a long-lived Microsoft permission kept per folder.

Manual Connect / Sync Now still use the short-lived Graph token the
browser gets at the moment of the click (app/services/onedrive_source.py).
An automatic sync has nobody signed in, so when someone turns automatic
sync on for a folder, the API exchanges *their* current Entra sign-in for
a Microsoft permission that it can renew on its own:

  1. The page sends a token Entra issued for this app itself
     (scope "<ENTRA_CLIENT_ID>/.default", or ENTRA_API_SCOPE when set).
  2. The API trades it, using ENTRA_CLIENT_SECRET, for a Graph token +
     refresh token on behalf of that person (Microsoft's "on-behalf-of"
     exchange; delegated Files.Read.All + offline_access).
  3. The refresh token is encrypted and stored in that connection's
     DarAI_SourceConnections.Secret; the person becomes the folder's
     "automatic sync account". Each automatic sync renews it.
  4. If Microsoft refuses a renewal (password change, sign-in policy such
     as "sign in again every 7 days", account disabled ...), the folder is
     marked "Reconnect needed" and its automatic sync pauses until that
     person -- or anyone -- clicks Reconnect. Documents already synced are
     not touched.
  5. Whenever the automatic sync account opens the OneDrive page, the
     stored permission is quietly renewed from their current sign-in, so
     reconnects are only needed after a long absence.

The automatic sync reads exactly what that person can open in OneDrive /
SharePoint / Teams -- nothing more.

Encryption: TOKEN_ENCRYPTION_KEY (a Fernet key) when set, otherwise a key
derived from JWT_SECRET_KEY. Changing either means folders need Reconnect.
Nothing here ever logs or returns a token.
"""

import base64
import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.config import settings
from app.db.models import SourceConnection

logger = logging.getLogger(__name__)

GRAPH_SCOPES = ["https://graph.microsoft.com/Files.Read.All"]
_PLACEHOLDER_PREFIX = "REPLACE_WITH"
_RENEW_MARGIN_SECONDS = 300

_locks_guard = threading.Lock()
_locks: dict = {}


class OneDriveAuthError(RuntimeError):
    """The stored permission couldn't be created or used."""


class ReconnectNeeded(OneDriveAuthError):
    """Microsoft wants the person to sign in again before automatic syncs can continue."""


# ---------------------------------------------------------------------------
# Is it set up?
# ---------------------------------------------------------------------------


def _set(value: Optional[str]) -> bool:
    return bool((value or "").strip()) and not value.strip().startswith(_PLACEHOLDER_PREFIX)


def problem() -> Optional[str]:
    """Why automatic OneDrive sync isn't available, or None when it is."""
    if not _set(settings.ENTRA_CLIENT_SECRET):
        return (
            "Automatic sync for OneDrive needs ENTRA_CLIENT_SECRET in the API's .env "
            "(a client secret of the app's Entra registration)."
        )
    if not _set(settings.ENTRA_CLIENT_ID) or not _set(settings.ENTRA_TENANT_ID):
        return "ENTRA_CLIENT_ID / ENTRA_TENANT_ID aren't set in the API's .env."
    if not _set(settings.TOKEN_ENCRYPTION_KEY) and not _set(settings.JWT_SECRET_KEY):
        return "Set TOKEN_ENCRYPTION_KEY (or JWT_SECRET_KEY) in the API's .env."
    try:
        _fernet()
    except ValueError:
        return (
            "TOKEN_ENCRYPTION_KEY in the API's .env isn't a valid key -- generate one with: "
            "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return None


def available() -> bool:
    return problem() is None


def api_scope() -> str:
    """The scope the page asks Entra for, to get a token for this app itself."""
    custom = (settings.ENTRA_API_SCOPE or "").strip()
    return custom or f"{settings.ENTRA_CLIENT_ID}/.default"


# ---------------------------------------------------------------------------
# Encryption + storage (DarAI_SourceConnections.Secret / ConfigJson)
# ---------------------------------------------------------------------------


def _fernet() -> Fernet:
    key = (settings.TOKEN_ENCRYPTION_KEY or "").strip()
    if key:
        return Fernet(key.encode())
    derived = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"darai-source-connection-tokens"
    ).derive(settings.JWT_SECRET_KEY.encode())
    return Fernet(base64.urlsafe_b64encode(derived))


def _config(connection: SourceConnection) -> dict:
    try:
        return json.loads(connection.ConfigJson or "{}")
    except ValueError:
        return {}


def _save_config(connection: SourceConnection, config: dict) -> None:
    connection.ConfigJson = json.dumps(config)


def has_permission(connection: SourceConnection) -> bool:
    return connection.SourceType == "onedrive" and bool(connection.Secret)


def needs_reconnect(connection: SourceConnection) -> bool:
    return has_permission(connection) and _config(connection).get("auth_state") == "reconnect"


def owner_user_id(connection: SourceConnection) -> Optional[int]:
    return _config(connection).get("auth_user_id") if has_permission(connection) else None


def reconnect_reason(connection: SourceConnection) -> Optional[str]:
    return _config(connection).get("auth_problem") if needs_reconnect(connection) else None


def _store(connection: SourceConnection, *, refresh_token: str, user_id: Optional[int]) -> None:
    connection.Secret = _fernet().encrypt(
        json.dumps({"rt": refresh_token, "at": datetime.utcnow().isoformat()}).encode()
    ).decode()
    config = _config(connection)
    if user_id is not None:
        config["auth_user_id"] = user_id
    config.pop("auth_state", None)
    config.pop("auth_problem", None)
    config["auth_renewed_at"] = datetime.utcnow().isoformat()
    _save_config(connection, config)


def _load_refresh_token(connection: SourceConnection) -> str:
    try:
        return json.loads(_fernet().decrypt(connection.Secret.encode()))["rt"]
    except (InvalidToken, ValueError, KeyError, TypeError) as exc:
        raise ReconnectNeeded(
            "The saved Microsoft sign-in for this folder can't be read any more (the encryption "
            "key changed) -- click Reconnect."
        ) from exc


def clear(connection: SourceConnection) -> None:
    """Forgets the stored permission (automatic sync turned off). Does not commit."""
    connection.Secret = None
    config = _config(connection)
    for key in ("auth_user_id", "auth_state", "auth_problem", "auth_renewed_at"):
        config.pop(key, None)
    _save_config(connection, config)


def mark_reconnect(connection: SourceConnection, reason: str) -> None:
    """Pauses automatic sync until Reconnect. Does not commit."""
    config = _config(connection)
    config["auth_state"] = "reconnect"
    config["auth_problem"] = reason[:500]
    _save_config(connection, config)


# ---------------------------------------------------------------------------
# Talking to Microsoft
# ---------------------------------------------------------------------------


_app_cache: dict = {}


def _app():
    """One MSAL client per secret (it looks up Microsoft's endpoints once)."""
    import msal  # already a dependency (requirements.txt)

    key = (settings.ENTRA_CLIENT_ID, settings.ENTRA_TENANT_ID, settings.ENTRA_CLIENT_SECRET.strip())
    with _locks_guard:
        app = _app_cache.get(key)
        if app is None:
            _app_cache.clear()
            app = msal.ConfidentialClientApplication(
                client_id=key[0],
                client_credential=key[2],
                authority=f"https://login.microsoftonline.com/{key[1]}",
            )
            _app_cache[key] = app
        return app


def _friendly(result: dict) -> str:
    codes = result.get("error_codes") or []
    text = (result.get("error_description") or result.get("error") or "").split("\r\n")[0]
    if 7000215 in codes or 7000222 in codes:
        return (
            "Microsoft rejected the API's client secret (wrong value, or it has expired) -- check "
            "ENTRA_CLIENT_SECRET in .env. Use the secret's Value, not its Secret ID."
        )
    if 65001 in codes or "consent" in text.lower():
        return (
            "The app doesn't have permission yet -- an admin needs to grant consent for "
            "Files.Read.All and offline_access on the app in Microsoft Entra."
        )
    if 50013 in codes or 500131 in codes:
        return (
            "Microsoft didn't accept the sign-in passed from the page (audience mismatch). If the "
            "app uses an 'Expose an API' scope, set ENTRA_API_SCOPE in .env to it."
        )
    if 50173 in codes or 50076 in codes or 50079 in codes or 50078 in codes or "interaction" in (result.get("error") or ""):
        return "Microsoft wants you to sign in again (a company sign-in rule) -- click Reconnect."
    return f"Microsoft said: {text[:300] or 'unknown error'}"


def grant(connection: SourceConnection, *, api_token: str, user_id: int) -> None:
    """
    Exchanges the person's current sign-in (a token for this app, from the
    page) for a stored, renewable OneDrive permission on this connection;
    the person becomes its automatic sync account. Does not commit.
    """
    if not available():
        raise OneDriveAuthError(problem())
    if not api_token:
        raise OneDriveAuthError("Your sign-in wasn't sent -- reload the page and try again.")
    result = _app().acquire_token_on_behalf_of(api_token, GRAPH_SCOPES)
    if "error" in result:
        logger.warning("OneDrive permission exchange failed: %s %s", result.get("error"), result.get("error_codes"))
        raise OneDriveAuthError(
            "Couldn't set up automatic sync for this folder. " + _friendly(result)
        )
    refresh_token = result.get("refresh_token")
    if not refresh_token:
        raise OneDriveAuthError(
            "Microsoft didn't return a long-lived sign-in -- check that the app has the "
            "'offline_access' permission (with admin consent) in Microsoft Entra."
        )
    _store(connection, refresh_token=refresh_token, user_id=user_id)


def _lock_for(connection_id: int) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(connection_id, threading.Lock())


def token_provider(connection_id: int) -> Callable[[], str]:
    """
    A function returning a current Graph token for this connection's
    stored permission -- renewed when it's within 5 minutes of expiry, and
    the new refresh token saved. Raises ReconnectNeeded (and marks the
    folder) when Microsoft refuses.
    """
    state = {"token": None, "expires": 0.0}

    def get() -> str:
        if state["token"] and time.time() < state["expires"] - _RENEW_MARGIN_SECONDS:
            return state["token"]
        from app.db.session import SessionLocal

        with _lock_for(connection_id):
            db = SessionLocal()
            try:
                connection = db.get(SourceConnection, connection_id)
                if connection is None or not has_permission(connection):
                    raise ReconnectNeeded(
                        "Automatic sync isn't set up for this folder any more -- turn it on again."
                    )
                if needs_reconnect(connection):
                    raise ReconnectNeeded(
                        reconnect_reason(connection) or "This folder needs Reconnect."
                    )
                refresh_token = _load_refresh_token(connection)
                result = _app().acquire_token_by_refresh_token(refresh_token, GRAPH_SCOPES)
                if "error" in result:
                    reason = (
                        "Microsoft asked for a new sign-in for this folder's automatic sync -- "
                        "click Reconnect on the OneDrive page. (" + _friendly(result) + ")"
                    )
                    if result.get("error") in ("invalid_grant", "interaction_required"):
                        mark_reconnect(connection, reason)
                        connection.LastSyncStatus = "error"
                        connection.LastSyncError = reason[:1000]
                        db.commit()
                        raise ReconnectNeeded(reason)
                    raise OneDriveAuthError("Couldn't renew the folder's Microsoft sign-in. " + _friendly(result))
                if result.get("refresh_token"):
                    owner = _config(connection).get("auth_user_id")
                    _store(connection, refresh_token=result["refresh_token"], user_id=owner)
                    db.commit()
                state["token"] = result["access_token"]
                state["expires"] = time.time() + int(result.get("expires_in") or 3600)
                return state["token"]
            finally:
                db.close()

    return get
