# These imports should work if PYTHONPATH is set to include /src
from edge_agent.config import EdgeSettings
from edge_agent.main import run


def test_edge_agent_imports_and_settings():
    """
    Sanity test:
    - EdgeSettings can be constructed
    - Basic field types are correct (pydantic validation works)
    """
    cfg = EdgeSettings()

    # site_id should always be a string (from defaults or .env)
    assert isinstance(cfg.site_id, str)

    # tcp_port should always be an integer (pydantic should coerce/validate)
    assert isinstance(cfg.tcp_port, int)


def test_run_does_not_crash(monkeypatch):
    """
    run() uses argparse (which reads sys.argv).
    In pytest, sys.argv includes pytest arguments, so we patch it to keep run() clean.
    """
    monkeypatch.setattr("sys.argv", ["edge_agent"])

    # run() should exit cleanly with code 0 for PR1 skeleton
    code = run()
    assert code == 0
