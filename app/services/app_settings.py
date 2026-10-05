"""
Editable application settings (the Settings page, sections A-E).

Storage is "empty until changed": DarAI_AppSettings holds a row only for a
setting someone has deliberately changed. Every other setting uses its
default from app/config.py. "Reset to defaults" deletes the row; saving a
value equal to the default does the same, so the table never holds
redundant rows. Every change is logged in DarAI_AppSettingChanges.

Values cross the API in the units the Settings page shows them in --
percentages as whole numbers (50 = 50%), "closeness" as a percentage,
days/words as integers, switches as true/false. Internally (and in the
database) they're stored the way app/config.py and the processing code
use them: fractions (0.5), a cosine distance (0.15), integers, "true"/
"false". The conversion lives only here.

Reading: get_effective() returns every setting's current internal value,
cached briefly so an upload doesn't hit the database per threshold; the
cache is cleared whenever settings are saved or reset.

Who may change settings: currently any signed-in user (one user today).
can_manage_settings() is the single place to restrict that later.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings as config_defaults
from app.db.models import AppSetting, AppSettingChange, User

# kind -> how a value is shown/entered on the Settings page
#   'percent'   internal fraction 0-1      <-> whole-number percent
#   'closeness' internal cosine distance   <-> (1 - distance) as percent
#   'int'       integer (words, days)
#   'bool'      on/off switch


@dataclass(frozen=True)
class SettingDef:
    key: str
    section: str
    label: str
    help: str
    kind: str
    min: Optional[float] = None  # in display units
    max: Optional[float] = None  # in display units
    unit: str = ""  # shown after the input: '%', 'words', 'days'
    warn_below: Optional[float] = None  # display units; soft warning, not an error
    warn_text: str = ""


SECTIONS = [
    ("same_name", "Same-name version matching",
     "When an uploaded file has the same name as an existing document (ignoring "
     "years, “new version”, “v2” and similar), how the new file is judged."),
    ("content", "Different-name (content) matching",
     "When an uploaded file has a different name, it replaces an existing document "
     "only if its wording is almost the same."),
    ("review", "Review flag",
     "Uploads that look like a version but don’t meet the rules above are kept as "
     "separate documents and marked REVIEW."),
    ("retention", "History and retention",
     "Deleted documents and previous versions can be restored until they are "
     "permanently deleted. 0 = keep forever."),
    ("advanced", "Advanced",
     "Change only if advised."),
]

DEFINITIONS: list[SettingDef] = [
    SettingDef(
        "VERSION_MATCH_MIN_SIMILARITY", "same_name", "Name-match wording similarity",
        "A file with the same name replaces the existing document if at least this much "
        "of the wording matches both ways.",
        "percent", 30, 95, "%", 40,
        "Below 40%, a rewritten document with a similar name may replace an unrelated one.",
    ),
    SettingDef(
        "BLOCK_OLDER_EDITIONS", "same_name", "Block older editions",
        "A file whose year is earlier than the stored document’s (e.g. 2024 vs 2026) is "
        "not added, so an old edition never replaces a newer one.",
        "bool",
    ),
    SettingDef(
        "CONTENT_MATCH_MIN_SIMILARITY", "content", "Content match – main direction",
        "At least this much of one document’s wording must be found in the other.",
        "percent", 60, 100, "%", 70,
        "Below 70%, different documents that share a lot of wording may be merged.",
    ),
    SettingDef(
        "CONTENT_MATCH_MIN_COVERAGE", "content", "Content match – other direction",
        "…and at least this much the other way. Stops an excerpt, or a document that "
        "merely contains another, from replacing it.",
        "percent", 40, 100, "%", 50,
        "Below 50%, an excerpt or a combined document may replace the full document.",
    ),
    SettingDef(
        "CONTENT_MATCH_MIN_WORDS", "content", "Minimum document length",
        "Shorter files (forms, one-page notices) are never matched by content alone — "
        "there isn’t enough text to judge.",
        "int", 50, 2000, "words", 100,
        "Below 100 words, short forms may be merged with each other.",
    ),
    SettingDef(
        "PROTECT_SIBLING_NAMES", "content", "Protect look-alike documents",
        "When on, documents whose names look like a series (e.g. “NDA Vendor A” and "
        "“NDA Vendor B”) are never merged automatically, even if their wording is nearly "
        "identical -- both are kept and marked REVIEW. Off by default: the same content "
        "under another name replaces the older file, which stays restorable in History.",
        "bool",
    ),
    SettingDef(
        "POSSIBLE_VERSION_MIN_SIMILARITY", "review", "Flag as possible version from",
        "A different-name file with at least this much matching wording (but below the "
        "content match rules) is kept and marked REVIEW.",
        "percent", 30, 95, "%",
    ),
    SettingDef(
        "DELETED_RETENTION_DAYS", "retention", "Keep set-aside copies for",
        "Older copies that an upload replaced automatically wait in the Library's Deleted "
        "tab for this many days, then are permanently deleted. 0 keeps them until deleted "
        "by hand. (Deleting a document yourself is always permanent.)",
        "int", 0, 365, "days",
    ),
    SettingDef(
        "PREVIOUS_VERSION_RETENTION_DAYS", "retention", "Keep previous versions for",
        "After this many days, replaced versions are permanently deleted and can no "
        "longer be restored. 0 keeps them until the document is deleted.",
        "int", 0, 365, "days",
    ),
    SettingDef(
        "VERSION_CANDIDATE_MAX_DISTANCE", "advanced", "Candidate search closeness",
        "How close a passage must be before a document is even compared by wording. "
        "Lower finds more candidates (slower uploads); higher may miss renamed versions.",
        "closeness", 70, 99, "%",
    ),
]

_BY_KEY = {d.key: d for d in DEFINITIONS}

_CACHE_SECONDS = 30
_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"values": None, "loaded_at": 0.0}


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def _default_internal(d: SettingDef) -> Any:
    return getattr(config_defaults, d.key)


def _parse_stored(d: SettingDef, raw: str) -> Any:
    if d.kind == "bool":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if d.kind == "int":
        return int(float(raw))
    return float(raw)


def _to_stored(d: SettingDef, internal: Any) -> str:
    if d.kind == "bool":
        return "true" if internal else "false"
    if d.kind == "int":
        return str(int(internal))
    return repr(round(float(internal), 6))


def to_display(d: SettingDef, internal: Any) -> Any:
    if d.kind == "percent":
        return round(float(internal) * 100, 2)
    if d.kind == "closeness":
        return round((1 - float(internal)) * 100, 2)
    if d.kind == "int":
        return int(internal)
    return bool(internal)


def _with_unit(d: SettingDef, number: float) -> str:
    if d.unit == "%":
        return f"{number:g}%"
    return f"{number:g} {d.unit}" if d.unit else f"{number:g}"


def _from_display(d: SettingDef, display: Any) -> Any:
    if d.kind == "bool":
        if not isinstance(display, bool):
            raise ValueError("must be on or off")
        return display
    try:
        number = float(display)
    except (TypeError, ValueError):
        raise ValueError("must be a number")
    if d.kind == "int" and number != int(number):
        raise ValueError("must be a whole number")
    if d.min is not None and number < d.min:
        raise ValueError(f"must be at least {_with_unit(d, d.min)}")
    if d.max is not None and number > d.max:
        raise ValueError(f"must be at most {_with_unit(d, d.max)}")
    if d.kind == "percent":
        return round(number / 100, 6)
    if d.kind == "closeness":
        return round(1 - number / 100, 6)
    return int(number)


def _same(d: SettingDef, a: Any, b: Any) -> bool:
    if d.kind in ("percent", "closeness"):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _load(db: Session) -> dict[str, Any]:
    values = {d.key: _default_internal(d) for d in DEFINITIONS}
    for row in db.query(AppSetting).all():
        d = _BY_KEY.get(row.SettingKey)
        if d is None:
            continue  # a key no longer used by this version of the app
        try:
            values[d.key] = _parse_stored(d, row.SettingValue)
        except (TypeError, ValueError):
            pass  # unreadable value -> keep the default rather than fail uploads
    return values


def get_effective(db: Session) -> dict[str, Any]:
    """Every setting's current value, in internal units (cached ~30 s)."""
    with _cache_lock:
        cached = _cache["values"]
        if cached is not None and time.monotonic() - _cache["loaded_at"] < _CACHE_SECONDS:
            return dict(cached)
    values = _load(db)
    with _cache_lock:
        _cache["values"] = values
        _cache["loaded_at"] = time.monotonic()
    return dict(values)


def _invalidate() -> None:
    with _cache_lock:
        _cache["values"] = None


def describe(db: Session) -> dict:
    """Everything the Settings page needs: sections, items, current values."""
    rows = {row.SettingKey: row for row in db.query(AppSetting).all()}
    updater_ids = [r.UpdatedBy for r in rows.values() if r.UpdatedBy]
    user_names = (
        {u.Id: u.DisplayName for u in db.query(User).filter(User.Id.in_(updater_ids))}
        if updater_ids
        else {}
    )
    current = _load(db)

    sections = []
    for section_key, title, description in SECTIONS:
        items = []
        for d in DEFINITIONS:
            if d.section != section_key:
                continue
            row = rows.get(d.key)
            default = _default_internal(d)
            items.append(
                {
                    "key": d.key,
                    "label": d.label,
                    "help": d.help,
                    "kind": d.kind,
                    "unit": d.unit,
                    "min": d.min,
                    "max": d.max,
                    "value": to_display(d, current[d.key]),
                    "default_value": to_display(d, default),
                    "is_default": row is None,
                    "updated_at": row.UpdatedAt if row else None,
                    "updated_by": user_names.get(row.UpdatedBy) if row else None,
                    "warn_below": d.warn_below,
                    "warn_text": d.warn_text or None,
                }
            )
        sections.append({"key": section_key, "title": title, "description": description, "items": items})
    return {"sections": sections}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def can_manage_settings(user: User) -> bool:
    # Currently every signed-in user (one user today). Restrict here later --
    # e.g. by user id or an admin flag -- without touching the rest.
    return True


def _check_access(user: User) -> None:
    if not can_manage_settings(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to change settings.",
        )


def _cross_check(values: dict[str, Any]) -> list[str]:
    """Rules between settings (internal units). Returns error messages."""
    errors = []
    main = values["CONTENT_MATCH_MIN_SIMILARITY"]
    coverage = values["CONTENT_MATCH_MIN_COVERAGE"]
    review = values["POSSIBLE_VERSION_MIN_SIMILARITY"]
    if coverage > main:
        errors.append(
            "“Content match – other direction” can’t be higher than “Content match – main "
            f"direction” ({round(coverage * 100)}% vs {round(main * 100)}%)."
        )
    if review >= main:
        errors.append(
            "“Flag as possible version from” must be lower than “Content match – main "
            f"direction” ({round(review * 100)}% vs {round(main * 100)}%)."
        )
    return errors


def _warnings(values: dict[str, Any]) -> list[str]:
    out = []
    for d in DEFINITIONS:
        if d.warn_below is None:
            continue
        shown = to_display(d, values[d.key])
        if shown < d.warn_below:
            out.append(f"{d.label}: {d.warn_text}")
    return out


def update(db: Session, changes: dict[str, Any], user: User) -> dict:
    """
    Applies {key: display_value} changes atomically: every value is checked
    (range + rules between settings) before anything is saved; one bad
    value rejects the whole save. Returns describe() plus `warnings`.
    """
    _check_access(user)

    unknown = [key for key in changes if key not in _BY_KEY]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown setting(s): {', '.join(unknown)}")

    field_errors: dict[str, str] = {}
    new_internal: dict[str, Any] = {}
    for key, display in changes.items():
        d = _BY_KEY[key]
        try:
            new_internal[key] = _from_display(d, display)
        except ValueError as exc:
            field_errors[key] = f"{d.label} {exc}."
    if field_errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "Some values are out of range.", "fields": field_errors},
        )

    current = _load(db)
    proposed = {**current, **new_internal}
    rule_errors = _cross_check(proposed)
    if rule_errors:
        raise HTTPException(status_code=422, detail={"message": " ".join(rule_errors), "fields": {}})

    rows = {row.SettingKey: row for row in db.query(AppSetting).filter(AppSetting.SettingKey.in_(list(new_internal)))}
    for key, value in new_internal.items():
        d = _BY_KEY[key]
        if _same(d, value, current[key]):
            continue  # unchanged
        row = rows.get(key)
        old_stored = row.SettingValue if row else None
        if _same(d, value, _default_internal(d)):
            # back to the default -> no row ("empty until changed")
            if row is not None:
                db.delete(row)
            new_stored = None
        else:
            new_stored = _to_stored(d, value)
            if row is None:
                db.add(AppSetting(SettingKey=key, SettingValue=new_stored, UpdatedBy=user.Id))
            else:
                row.SettingValue = new_stored
                row.UpdatedBy = user.Id
                row.UpdatedAt = func.sysutcdatetime()
        db.add(AppSettingChange(SettingKey=key, OldValue=old_stored, NewValue=new_stored, ChangedBy=user.Id))

    db.commit()
    _invalidate()
    result = describe(db)
    result["warnings"] = _warnings(proposed)
    return result


def reset(db: Session, keys: Optional[list[str]], user: User) -> dict:
    """Resets the given settings (or all, if keys is None/empty) to their defaults."""
    _check_access(user)
    targets = keys or [d.key for d in DEFINITIONS]
    unknown = [key for key in targets if key not in _BY_KEY]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown setting(s): {', '.join(unknown)}")

    for row in db.query(AppSetting).filter(AppSetting.SettingKey.in_(targets)).all():
        db.add(
            AppSettingChange(SettingKey=row.SettingKey, OldValue=row.SettingValue, NewValue=None, ChangedBy=user.Id)
        )
        db.delete(row)
    db.commit()
    _invalidate()
    result = describe(db)
    result["warnings"] = []
    return result


def recent_changes(db: Session, limit: int = 20) -> list[dict]:
    rows = db.query(AppSettingChange).order_by(AppSettingChange.ChangedAt.desc(), AppSettingChange.Id.desc()).limit(limit).all()
    names = {}
    ids = {r.ChangedBy for r in rows if r.ChangedBy}
    if ids:
        names = {u.Id: u.DisplayName for u in db.query(User).filter(User.Id.in_(list(ids)))}

    def shown(d: Optional[SettingDef], stored: Optional[str]) -> Optional[str]:
        if d is None:
            return stored
        internal = _default_internal(d) if stored is None else _parse_stored(d, stored)
        value = to_display(d, internal)
        if d.kind == "bool":
            text = "On" if value else "Off"
        else:
            text = _with_unit(d, value)
        return text + (" (default)" if stored is None else "")

    out = []
    for r in rows:
        d = _BY_KEY.get(r.SettingKey)
        out.append(
            {
                "key": r.SettingKey,
                "label": d.label if d else r.SettingKey,
                "old_value": shown(d, r.OldValue),
                "new_value": shown(d, r.NewValue),
                "changed_at": r.ChangedAt,
                "changed_by": names.get(r.ChangedBy),
            }
        )
    return out
