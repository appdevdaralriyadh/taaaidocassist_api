"""
POST /api/auth/login -- exchanges a validated Entra token for the app's
own JWT (spec §3.1). Creates the local DarAI_Users row on first login for
an account; every later login for that same account just re-issues a JWT.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.entra import validate_entra_token
from app.auth.jwt_handler import create_app_jwt
from app.db.models import User
from app.db.session import get_db
from app.schemas import LoginRequest, LoginResponse

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    claims = validate_entra_token(payload.entra_token)

    entra_object_id = claims["oid"]
    display_name = claims.get("name")

    user = db.query(User).filter(User.EntraObjectId == entra_object_id).first()
    if user is None:
        # First successful Entra login for this account -- create the
        # local record. No role is assigned; both accounts are equal
        # (spec §3.1) and access is controlled entirely at the Entra
        # app-registration level.
        user = User(EntraObjectId=entra_object_id, DisplayName=display_name)
        db.add(user)
        db.commit()
        db.refresh(user)
    elif display_name and user.DisplayName != display_name:
        user.DisplayName = display_name
        db.commit()

    token = create_app_jwt(user_id=user.Id, entra_object_id=user.EntraObjectId)
    return LoginResponse(access_token=token, display_name=user.DisplayName)
