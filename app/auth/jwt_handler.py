"""
Issues and verifies the app's own JWT (HS256), separate from Entra's
tokens. This is the token the Angular app attaches to every API call
after login (spec §3.1) — the raw Entra token is used exactly once, during
/api/auth/login, and never again.
"""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, status

from app.config import settings


def create_app_jwt(*, user_id: int, entra_object_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "entra_oid": entra_object_id,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_app_jwt(token: str) -> dict:
    try:
        return jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired session token: {exc}",
        ) from exc
