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


def test_run_http_serve_calls_uvicorn(monkeypatch):
    called = {}

    def fake_run(app, **kwargs):
        called.update(kwargs)
        return None

    monkeypatch.setattr(uvicorn, "run", fake_run)

    cfg = EdgeSettings(
        edge_http_host="127.0.0.1", edge_http_port=9999, log_level="INFO"
    )
    code = run(argv=["--http-serve"], cfg=cfg)

    assert code == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 9999
    assert called["log_level"] == "info"


def test_run_http_serve_sets_status_online(monkeypatch):
    """
    Test that when --http-serve is used,
    the ServerSender's set_status is called with 'online'.
    """

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
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

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr("edge_agent.main.threading.Thread", DummyThread)
    monkeypatch.setattr("edge_agent.main.ServerSender", FakeSender)
    monkeypatch.setattr("edge_agent.edge_api.create_app", lambda cfg, sender: object())

    cfg = EdgeSettings(
        edge_http_host="127.0.0.1", edge_http_port=9999, log_level="INFO"
    )
    code = run(argv=["--http-serve"], cfg=cfg)

    assert code == 0
    assert FakeSender.last_instance is not None
    assert "online" in FakeSender.last_instance.statuses


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
