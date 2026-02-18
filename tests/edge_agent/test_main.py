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
    assert "site_id" in out


def test_run_http_serve_calls_uvicorn(monkeypatch):
    called = {}

    def fake_run(app, host, port, log_level):
        called["host"] = host
        called["port"] = port
        called["log_level"] = log_level
        return None

    monkeypatch.setattr(uvicorn, "run", fake_run)

    cfg = EdgeSettings(edge_http_host="127.0.0.1", edge_http_port=9999, log_level="INFO")
    code = run(argv=["--http-serve"], cfg=cfg)

    assert code == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 9999
    assert called["log_level"] == "info"


def test_run_returns_1_on_unexpected_exception(monkeypatch):
    import edge_agent.main as m

    monkeypatch.setattr(m, "build_parser", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
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

