"""
Shared FastAPI dependencies.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.jwt_handler import decode_app_jwt
from app.db.models import User
from app.db.session import get_db

_bearer_scheme = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Resolves the app JWT from the Authorization header into a DarAI_Users
    row. Both accounts have identical permissions (spec §3.1) — this only
    identifies *who* is calling, it never gates access by role.
    """
    claims = decode_app_jwt(credentials.credentials)
    user = db.get(User, int(claims["sub"]))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
        )
    return user
