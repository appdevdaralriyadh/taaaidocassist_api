"""
Application settings for the Settings page (see app/services/app_settings.py
for the list of settings, their sections, ranges and storage rules).

  GET  /api/settings          sections + every setting's current value
  PUT  /api/settings          save changed values  {"values": {key: value}}
  POST /api/settings/reset    reset some settings  {"keys": [...]}, or all ({})
  GET  /api/settings/changes  recent change log

Values are in the units the page shows (50 = 50%). Who may change settings
is decided by app_settings.can_manage_settings() -- currently any signed-in
user.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.db.models import User
from app.db.session import get_db
from app.schemas import (
    SettingChangeItem,
    SettingsResetRequest,
    SettingsResponse,
    SettingsUpdateRequest,
)
from app.services import app_settings

router = APIRouter()


@router.get("", response_model=SettingsResponse)
def get_settings(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = app_settings.describe(db)
    return SettingsResponse(**result, warnings=[], can_edit=app_settings.can_manage_settings(user))


@router.put("", response_model=SettingsResponse)
def update_settings(
    payload: SettingsUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = app_settings.update(db, payload.values, user)
    return SettingsResponse(**result, can_edit=app_settings.can_manage_settings(user))


@router.post("/reset", response_model=SettingsResponse)
def reset_settings(
    payload: SettingsResetRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    result = app_settings.reset(db, payload.keys, user)
    return SettingsResponse(**result, can_edit=app_settings.can_manage_settings(user))


@router.get("/changes", response_model=list[SettingChangeItem])
def setting_changes(
    limit: int = Query(20, ge=1, le=200),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return [SettingChangeItem(**row) for row in app_settings.recent_changes(db, limit)]
