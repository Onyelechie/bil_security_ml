from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .types import MotionEvent


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass(slots=True)
class ExtractionJob:
    incident_id: str
    camera_id: str
    window_start: datetime
    window_end: datetime
    created_at_utc: datetime
    reason: str  # "quiet" | "max"


@dataclass(slots=True)
class IncidentState:
    incident_id: str
    camera_id: str

    started_at_utc: datetime
    first_motion_at_utc: datetime
    last_motion_at_utc: datetime

    window_start: datetime
    window_end: datetime

    finalize_at_utc: datetime


class IncidentManager:
    """
    Merge many MotionEvents into one 'incident' per camera (no spam).

    - Incident starts only when we see the first *accepted* motion event.
    - While incident is active, ANY motion event (accepted or dropped) extends:
        last_motion_at, window_end, finalize_at (quiet timer).
    - Finalize when:
        (A) quiet: now >= finalize_at
        (B) max:   now - started_at >= max_incident_sec  -> finalize chunk and roll to new incident
    """

    def __init__(
        self,
        *,
        pre_sec: float,
        post_sec: float,
        quiet_sec: float,
        max_incident_sec: float,
    ) -> None:
        self._pre = timedelta(seconds=float(pre_sec))
        self._post = timedelta(seconds=float(post_sec))
        self._quiet = timedelta(seconds=float(quiet_sec))
        self._max = timedelta(seconds=float(max_incident_sec))

        self._states: dict[str, IncidentState] = {}

    @staticmethod
    def _cam(evt: MotionEvent) -> str:
        return evt.camera_id or "unknown"

    def ingest(self, evt: MotionEvent, *, accepted: bool) -> None:
        cam = self._cam(evt)
        now = _utc(evt.received_at_utc)

        st = self._states.get(cam)

        # Only start a new incident on accepted motion.
        if st is None:
            if not accepted:
                return
            incident_id = uuid.uuid4().hex
            first = now
            last = now
            window_start = first - self._pre
            window_end = last + self._post
            finalize_at = last + self._quiet

            self._states[cam] = IncidentState(
                incident_id=incident_id,
                camera_id=cam,
                started_at_utc=now,
                first_motion_at_utc=first,
                last_motion_at_utc=last,
                window_start=window_start,
                window_end=window_end,
                finalize_at_utc=finalize_at,
            )
            return

        # Incident exists: extend it on ANY motion signal.
        st.last_motion_at_utc = now
        st.window_end = now + self._post
        st.finalize_at_utc = now + self._quiet

    def tick(self, now_utc: datetime) -> list[ExtractionJob]:
        """
        Return any finalized jobs (quiet/max). Pure function for easy testing.
        """
        now = _utc(now_utc)
        jobs: list[ExtractionJob] = []

        for cam in list(self._states.keys()):
            st = self._states[cam]

            # Quiet finalize
            if now >= st.finalize_at_utc:
                jobs.append(
                    ExtractionJob(
                        incident_id=st.incident_id,
                        camera_id=st.camera_id,
                        window_start=st.window_start,
                        window_end=st.window_end,
                        created_at_utc=now,
                        reason="quiet",
                    )
                )
                del self._states[cam]
                continue

            # Max-chunk finalize
            if now - st.started_at_utc >= self._max:
                jobs.append(
                    ExtractionJob(
                        incident_id=st.incident_id,
                        camera_id=st.camera_id,
                        window_start=st.window_start,
                        window_end=st.window_end,
                        created_at_utc=now,
                        reason="max",
                    )
                )
                # Roll to a new incident that starts at the last seen motion time
                # (keeps continuous motion from spamming, but avoids infinite waiting).
                new_start = st.last_motion_at_utc
                new_id = uuid.uuid4().hex
                self._states[cam] = IncidentState(
                    incident_id=new_id,
                    camera_id=cam,
                    started_at_utc=new_start,
                    first_motion_at_utc=new_start,
                    last_motion_at_utc=new_start,
                    window_start=new_start - self._pre,
                    window_end=new_start + self._post,
                    finalize_at_utc=new_start + self._quiet,
                )

        return jobs

    def active_incidents(self) -> int:
        return len(self._states)
