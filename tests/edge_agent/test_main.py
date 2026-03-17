import runpy
import sys

import pytest
import uvicorn

from edge_agent.config import EdgeSettings
from edge_agent.main import run


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
    """
    Test that --http-serve does not set status before app startup.
    """

    class DummyThread:
        def __init__(self, *args, **kwargs):
            self._target = kwargs.get("target")
            self._run_target = self._target and self._target.__name__ == "_run_server"

        def start(self):
            if self._run_target:
                self._target()
            return None

        def join(self, timeout=None):
            return None

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

    def fake_run(app, **kwargs):
        return None

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.started = False
            self.should_exit = True

        def run(self):
            return None

    monkeypatch.setattr(uvicorn, "run", fake_run)
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


def test_module_entrypoint_exits_cleanly(monkeypatch):
    """
    Covers:
    - the default "Nothing to do..." branch in run()
    - the __main__ guard
    """
    monkeypatch.setattr(sys, "argv", ["edge_agent"])

    # Ensure runpy executes a fresh copy (avoid RuntimeWarning)
    sys.modules.pop("edge_agent.main", None)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("edge_agent.main", run_name="__main__")

    assert exc.value.code == 0
