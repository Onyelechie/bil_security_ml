import asyncio
import runpy
import sys
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
import uvicorn

from edge_agent.config import EdgeSettings
from edge_agent.main import consume_extraction_results, run
from edge_agent.video.ring_buffer import FrameItem
from edge_agent.video.window_extractor import WindowResult, WindowStatus


def test_run_print_config_does_not_crash(capfd):
    cfg = EdgeSettings()
    code = run(argv=["--print-config"], cfg=cfg)
    assert code == 0

    out, _ = capfd.readouterr()
    assert "edge_pc_id" in out


def test_run_http_serve_uses_uvicorn_config(monkeypatch):
    captured = {}

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    class FakeServer:
        def __init__(self, config):
            captured["config"] = config
            self.started = False
            self.should_exit = True

        def run(self):
            return None

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr("edge_agent.main.threading.Thread", DummyThread)
    monkeypatch.setattr("edge_agent.edge_api.create_app", lambda cfg, sender: object())

    cfg = EdgeSettings(
        edge_http_host="127.0.0.1", edge_http_port=9999, log_level="INFO"
    )
    code = run(argv=["--http-serve"], cfg=cfg)

    assert code == 0
    config = captured["config"]
    assert config.host == "127.0.0.1"
    assert config.port == 9999
    assert config.log_level == "info"


def test_run_http_serve_does_not_set_status_before_startup(monkeypatch):
    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    class FakeSender:
        last_instance = None

        def __init__(self, settings):
            self.settings = settings
            self.statuses = []
            FakeSender.last_instance = self

        def set_status(self, status: str) -> None:
            self.statuses.append(status)

        def send_heartbeat(self, *args, **kwargs):
            return True

        def retry_queued_alerts(self):
            return None

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = True

        def run(self):
            return None

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr("edge_agent.main.threading.Thread", DummyThread)
    monkeypatch.setattr("edge_agent.main.ServerSender", FakeSender)
    monkeypatch.setattr("edge_agent.edge_api.create_app", lambda cfg, sender: object())

    cfg = EdgeSettings(
        edge_http_host="127.0.0.1", edge_http_port=9999, log_level="INFO"
    )
    code = run(argv=["--http-serve"], cfg=cfg)

    assert code == 0
    assert FakeSender.last_instance is not None
    assert FakeSender.last_instance.statuses == []


def test_run_http_serve_times_out_when_not_started(monkeypatch):
    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def join(self, timeout=None):
            return None

    class FakeServer:
        last_instance = None

        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = False
            FakeServer.last_instance = self

        def run(self):
            return None

    ticks = iter([0.0, 0.1, 10.1, 10.2])

    def fake_monotonic():
        return next(ticks)

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr("edge_agent.main.threading.Thread", DummyThread)
    monkeypatch.setattr("edge_agent.edge_api.create_app", lambda cfg, sender: object())
    monkeypatch.setattr("edge_agent.main.time.monotonic", fake_monotonic)
    monkeypatch.setattr("edge_agent.main.time.sleep", lambda _s: None)

    cfg = EdgeSettings(
        edge_http_host="127.0.0.1", edge_http_port=9999, log_level="INFO"
    )
    code = run(argv=["--http-serve"], cfg=cfg)

    assert code == 1
    assert FakeServer.last_instance is not None
    assert FakeServer.last_instance.should_exit is True


def test_run_returns_1_on_unexpected_exception(monkeypatch):
    import edge_agent.main as m

    def mock_build_parser():
        raise RuntimeError("boom")

    monkeypatch.setattr(m, "build_parser", mock_build_parser)
    code = m.run(argv=[])
    assert code == 1


def test_run_starts_retry_thread(monkeypatch):
    import edge_agent.main as m

    created = []

    class DummyThread:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    monkeypatch.setattr("edge_agent.main.threading.Thread", DummyThread)
    cfg = EdgeSettings(retry_interval_sec=123)

    code = run(argv=[], cfg=cfg)

    assert code == 0
    assert any(
        kw.get("target") == m.retry_loop
        and len(kw.get("args", ())) >= 2
        and kw.get("args", ())[1] == 123
        for kw in created
    )


def test_retry_loop_handles_exception():
    import edge_agent.main as m

    stop_event = threading.Event()

    class FakeSender:
        def __init__(self):
            self.calls = 0

        def retry_queued_alerts(self):
            self.calls += 1
            stop_event.set()
            raise RuntimeError("boom")

    sender = FakeSender()

    m.retry_loop(sender, 0, stop_event)

    assert sender.calls == 1


def test_module_entrypoint_exits_cleanly(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["edge_agent"])

    sys.modules.pop("edge_agent.main", None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("edge_agent.main", run_name="__main__")

    assert exc.value.code == 0


@pytest.mark.asyncio
async def test_consume_extraction_results_processes_ready(monkeypatch):
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("edge_agent.main.asyncio.to_thread", fake_to_thread)

    worker = MagicMock()
    worker.results = asyncio.Queue()

    pipeline = MagicMock()

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    selected = [FrameItem(ts=ts, frame=np.zeros((10, 10), dtype=np.uint8))]

    res = WindowResult(
        incident_id="inc1",
        camera_id="cam-1",
        window_start=ts,
        window_end=ts,
        selected=selected,
        status=WindowStatus.READY,
        reason="ok",
    )

    task = asyncio.create_task(consume_extraction_results(worker, pipeline))
    try:
        await worker.results.put(res)
        await asyncio.sleep(0)
        pipeline.process_frames.assert_called_once_with("cam-1", selected)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_consume_extraction_results_processes_partial(monkeypatch):
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("edge_agent.main.asyncio.to_thread", fake_to_thread)

    worker = MagicMock()
    worker.results = asyncio.Queue()

    pipeline = MagicMock()

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    selected = [FrameItem(ts=ts, frame=np.zeros((10, 10), dtype=np.uint8))]

    res = WindowResult(
        incident_id="inc2",
        camera_id="cam-2",
        window_start=ts,
        window_end=ts,
        selected=selected,
        status=WindowStatus.PARTIAL,
        reason="timeout",
    )

    task = asyncio.create_task(consume_extraction_results(worker, pipeline))
    try:
        await worker.results.put(res)
        await asyncio.sleep(0)
        pipeline.process_frames.assert_called_once_with("cam-2", selected)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_consume_extraction_results_skips_dropped(monkeypatch):
    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr("edge_agent.main.asyncio.to_thread", fake_to_thread)

    worker = MagicMock()
    worker.results = asyncio.Queue()

    pipeline = MagicMock()

    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    res = WindowResult(
        incident_id="inc3",
        camera_id="cam-3",
        window_start=ts,
        window_end=ts,
        selected=[],
        status=WindowStatus.DROPPED,
        reason="no_frames",
    )

    task = asyncio.create_task(consume_extraction_results(worker, pipeline))
    try:
        await worker.results.put(res)
        await asyncio.sleep(0)
        pipeline.process_frames.assert_not_called()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_run_mode_builds_evaluator_pipeline_and_local_trigger(monkeypatch):
    created = {}

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    class FakeReader:
        def __init__(self, cfg, ring):
            pass

        async def start(self):
            return None

        async def stop(self):
            return None

    class _Queue:
        async def get(self):
            # Simulate an empty queue without bubbling KeyboardInterrupt into asyncio.
            await asyncio.sleep(10)

    class FakeTcpTrigger:
        def __init__(self, cfg):
            self.queue = _Queue()

        async def start(self):
            created["tcp_started"] = True

        async def stop(self):
            created["tcp_stopped"] = True

    class FakeTriggerManager:
        def __init__(self, *args, **kwargs):
            pass

        def accept(self, evt):
            return True

    class FakeIncidentManager:
        def __init__(self, *args, **kwargs):
            self._tick_count = 0

        def ingest(self, evt, *, accepted):
            return None

        def tick(self, now):
            self._tick_count += 1
            if self._tick_count == 1:
                raise KeyboardInterrupt()
            return []

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.results = asyncio.Queue()

        async def start(self):
            created["worker_started"] = True

        async def stop(self):
            created["worker_stopped"] = True

        async def enqueue(self, job):
            return None

    class FakeLocalMotionTrigger:
        def __init__(self, cfg, ring, mgr, on_motion=None, queue_max=1000):
            self.on_motion = on_motion

        async def start(self):
            created["local_started"] = True

        async def stop(self):
            created["local_stopped"] = True

    class FakeEvaluator:
        def __init__(
            self,
            model_name,
            weights_path,
            person_conf=None,
            vehicle_conf=None,
            allowed_classes=None,
        ):
            created["model_name"] = model_name
            created["weights_path"] = weights_path
            created["person_conf"] = person_conf
            created["vehicle_conf"] = vehicle_conf
            created["allowed_classes"] = allowed_classes

    class FakePipeline:
        def __init__(
            self,
            evaluator,
            sender,
            image_output_dir="storage/ws_alert_images",
            save_images=True,
        ):
            created["pipeline_created"] = True
            created["pipeline_sender"] = sender
            created["pipeline_evaluator"] = evaluator

        def process_frames(self, camera_id, frames, *, frame_timestamps=None):
            return None

    monkeypatch.setattr("edge_agent.main.threading.Thread", DummyThread)
    monkeypatch.setattr("edge_agent.video.rtsp_reader.RtspReader", FakeReader)
    monkeypatch.setattr(
        "edge_agent.triggers.tcp_trigger.TcpMotionTrigger", FakeTcpTrigger
    )
    monkeypatch.setattr(
        "edge_agent.triggers.trigger_manager.TriggerManager", FakeTriggerManager
    )
    monkeypatch.setattr(
        "edge_agent.triggers.incident_manager.IncidentManager", FakeIncidentManager
    )
    monkeypatch.setattr(
        "edge_agent.video.extraction_worker.ExtractionWorker", FakeWorker
    )
    monkeypatch.setattr(
        "edge_agent.triggers.local_motion_trigger.LocalMotionTrigger",
        FakeLocalMotionTrigger,
    )
    monkeypatch.setattr("edge_agent.ml_evaluator.MLEvaluator", FakeEvaluator)
    monkeypatch.setattr("edge_agent.pipeline_runner.PipelineRunner", FakePipeline)

    cfg = EdgeSettings(
        detector_model="YOLOv8-Nano",
        detector_weights="custom.pt",
        rtsp_url_low="rtsp://demo/stream",
        incident_tick_interval_sec=0.01,
        enable_tcp_motion=True,
        enable_local_motion=True,
    )

    code = run(argv=["--run"], cfg=cfg)

    assert code == 0
    assert created["model_name"] == "YOLOv8-Nano"
    assert created["weights_path"] == "custom.pt"
    assert created["pipeline_created"] is True
    assert created["tcp_started"] is True
    assert created["local_started"] is True
    assert created["person_conf"] == cfg.detector_person_conf
    assert created["vehicle_conf"] == cfg.detector_vehicle_conf
    assert created["allowed_classes"] == cfg.detector_allowed_classes
