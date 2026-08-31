"""
Validates an Entra ID (Microsoft identity platform) token.

Fetches Entra's signing keys from its published JWKS endpoint and verifies
the token's signature, issuer, audience, and expiry — once, on login (spec
§3.1: "Backend validates the Entra token once ... then issues its own
JWT"). The resulting claims are used to look up/create the local
DarAI_Users record.

This module never talks to Entra to *issue* tokens — that happens
client-side, via MSAL.js in the Angular app. It only checks tokens Entra
already issued.

Expects the Angular app to send the MSAL *ID token* (not an access token):
for a public-client SPA with no custom API scope, the ID token's `aud`
claim is the app's own client ID, which is exactly what's validated below.
"""

from functools import lru_cache

import jwt
from fastapi import HTTPException, status
from jwt import PyJWKClient

from app.config import settings

_AUTHORITY = f"https://login.microsoftonline.com/{settings.ENTRA_TENANT_ID}/v2.0"
_JWKS_URL = (
    f"https://login.microsoftonline.com/{settings.ENTRA_TENANT_ID}"
    "/discovery/v2.0/keys"
)


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    # Cached for the process lifetime. PyJWKClient caches individual keys
    # itself; this just avoids re-creating the client object per request.
    return PyJWKClient(_JWKS_URL)


def validate_entra_token(token: str) -> dict:
    """
    Verifies signature, issuer, audience and expiry of an Entra-issued
    token. Returns the decoded claims on success, raises HTTPException(401)
    on any failure (expired token, wrong tenant/app, tampered signature,
    etc.).
    """
    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.ENTRA_CLIENT_ID,
            issuer=_AUTHORITY,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Entra token: {exc}",
        ) from exc
    return claims
