from __future__ import annotations

import argparse
import logging

from .config import EdgeSettings
from .logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser.
    """
    parser = argparse.ArgumentParser(description="BIL Security ML - Edge Agent (Area B)")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved configuration and exit.",
    )
    return parser


def run() -> int:
    """
    Edge Agent entrypoint (PR1 skeleton).

    PR1 goals:
    - prove imports work
    - prove config loads
    - prove logging works
    - prove CLI works
    - exit cleanly

    Future PRs will:
    - start TCP listener
    - start RTSP reader + ring buffer
    - run inference workers
    - send heartbeats / poll updates
    """
    args = build_parser().parse_args()

    # Load settings from environment / .env
    cfg = EdgeSettings()

    # Setup logging using configured level
    configure_logging(cfg.log_level)

    logger.info("Edge Agent starting (PR1 skeleton)")
    logger.info(
        "Resolved config: site_id=%s tcp=%s:%s server=%s",
        cfg.site_id, cfg.tcp_host, cfg.tcp_port, cfg.server_base_url
    )

    if args.print_config:
        # PR1 has no secrets; later we should avoid printing API keys.
        print(cfg.model_dump())
        return 0

    # PR1 doesn't start any real services yet.
    logger.info("PR1: No runtime behavior yet. TCP listener will be added in PR2.")
    return 0


if __name__ == "__main__":
    # Allows: python -m edge_agent.main
    raise SystemExit(run())
