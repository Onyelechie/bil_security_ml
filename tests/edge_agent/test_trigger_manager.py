from datetime import datetime, timedelta, timezone

from edge_agent.triggers.trigger_manager import TriggerManager
from edge_agent.triggers.types import MotionEvent


def _evt(ts: datetime, cam: str) -> MotionEvent:
    return MotionEvent(
        received_at_utc=ts,
        site_id="site",
        edge_pc_id="edge",
        camera_id=cam,
        source="tcp",
    )


def test_merge_window_drops_duplicates():
    mgr = TriggerManager(cooldown_sec=10, merge_window_sec=2.0)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    assert mgr.accept(_evt(t0, "1")) is True
    # Within merge window => drop
    assert mgr.accept(_evt(t0 + timedelta(seconds=1), "1")) is False


def test_cooldown_drops_until_elapsed():
    # Turn off merge window to isolate cooldown behavior
    mgr = TriggerManager(cooldown_sec=10, merge_window_sec=0.0)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    assert mgr.accept(_evt(t0, "1")) is True
    assert mgr.accept(_evt(t0 + timedelta(seconds=5), "1")) is False  # still cooldown
    assert mgr.accept(_evt(t0 + timedelta(seconds=11), "1")) is True  # cooldown passed


def test_different_cameras_independent():
    mgr = TriggerManager(cooldown_sec=10, merge_window_sec=2.0)
    t0 = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)

    assert mgr.accept(_evt(t0, "1")) is True
    # Camera 2 should not be blocked by camera 1 rules
    assert mgr.accept(_evt(t0 + timedelta(seconds=1), "2")) is True
