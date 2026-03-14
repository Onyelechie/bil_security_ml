from datetime import datetime, timedelta, timezone

from edge_agent.triggers.incident_manager import IncidentManager
from edge_agent.triggers.types import MotionEvent


def _evt(ts: datetime, cam: str) -> MotionEvent:
    return MotionEvent(
        received_at_utc=ts,
        site_id="site",
        edge_pc_id="edge",
        camera_id=cam,
        source="tcp",
    )


def test_incident_starts_only_on_accepted():
    mgr = IncidentManager(pre_sec=2, post_sec=6, quiet_sec=2, max_incident_sec=20)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    mgr.ingest(_evt(t0, "1"), accepted=False)
    assert mgr.active_incidents() == 0

    mgr.ingest(_evt(t0, "1"), accepted=True)
    assert mgr.active_incidents() == 1


def test_incident_extends_on_dropped_and_finalizes_after_quiet():
    mgr = IncidentManager(pre_sec=2, post_sec=6, quiet_sec=2, max_incident_sec=20)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    mgr.ingest(_evt(t0, "1"), accepted=True)
    # cooldown/merge might drop this, but it should still extend incident
    mgr.ingest(_evt(t0 + timedelta(seconds=1), "1"), accepted=False)

    # Not quiet yet
    jobs = mgr.tick(
        t0 + timedelta(seconds=2)
    )  # last_motion=+1, quiet=2 => finalize at +3
    assert jobs == []

    # Quiet reached: finalize
    jobs = mgr.tick(t0 + timedelta(seconds=3))
    assert len(jobs) == 1
    job = jobs[0]
    assert job.camera_id == "1"
    assert job.window_start == (t0 - timedelta(seconds=2))
    assert job.window_end == (t0 + timedelta(seconds=1 + 6))
    assert job.reason == "quiet"
    assert mgr.active_incidents() == 0


def test_max_incident_chunks_and_rolls_over():
    mgr = IncidentManager(pre_sec=2, post_sec=6, quiet_sec=2, max_incident_sec=5)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    mgr.ingest(_evt(t0, "1"), accepted=True)
    # keep extending
    mgr.ingest(_evt(t0 + timedelta(seconds=4), "1"), accepted=False)

    # hit max at +5
    jobs = mgr.tick(t0 + timedelta(seconds=5))
    assert len(jobs) == 1
    assert jobs[0].reason == "max"
    # rollover should keep an active incident
    assert mgr.active_incidents() == 1
