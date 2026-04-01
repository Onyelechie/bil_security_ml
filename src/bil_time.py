from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

WINNIPEG_TIMEZONE = ZoneInfo("America/Winnipeg")


def ensure_winnipeg(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=WINNIPEG_TIMEZONE)
    return dt.astimezone(WINNIPEG_TIMEZONE)


def now_in_winnipeg() -> datetime:
    return datetime.now(WINNIPEG_TIMEZONE)


def isoformat_winnipeg(dt: datetime) -> str:
    return ensure_winnipeg(dt).isoformat()


def filename_stamp_winnipeg(dt: datetime) -> str:
    local_dt = ensure_winnipeg(dt)
    return f"{local_dt.strftime('%Y%m%dT%H%M%S%f')}{local_dt.strftime('%z')}"
