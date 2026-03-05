from datetime import datetime, timedelta, timezone

from edge_agent.triggers.incident_manager import IncidentManager
from edge_agent.triggers.types import MotionEvent


def evt(ts, cam):
    return MotionEvent(
        received_at_utc=ts,
        site_id="site",
        edge_pc_id="edge",
        camera_id=cam,
        source="tcp",
    )


def test_dropped_motion_does_not_start_incident():
    mgr = IncidentManager(pre_sec=2, post_sec=6, quiet_sec=2, max_incident_sec=20)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    mgr.ingest(evt(t0, "1"), accepted=False)
    assert mgr.active_incidents() == 0


def test_two_cameras_keep_separate_incidents_and_finalize_independently():
    mgr = IncidentManager(pre_sec=2, post_sec=6, quiet_sec=2, max_incident_sec=20)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    mgr.ingest(evt(t0, "1"), accepted=True)
    mgr.ingest(evt(t0, "2"), accepted=True)
    assert mgr.active_incidents() == 2

    # extend cam1 only
    mgr.ingest(evt(t0 + timedelta(seconds=1), "1"), accepted=False)

    # At t0+3: cam2 should finalize (quiet), cam1 should finalize too (quiet from last=+1 => finalize at +3)
    jobs = mgr.tick(t0 + timedelta(seconds=3))
    cams = sorted([j.camera_id for j in jobs])
    assert cams == ["1", "2"]
    assert mgr.active_incidents() == 0
