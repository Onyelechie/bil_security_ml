from __future__ import annotations

import argparse
import logging

from .config import EdgeSettings
from .logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIL Security ML - Edge Agent (Area B)")
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit.")
    parser.add_argument("--http-serve", action="store_true", help="Start Edge HTTP API server (/health, /heartbeat).")
    return parser


def run(argv: list[str] | None = None, cfg: EdgeSettings | None = None) -> int:
    """
    Edge Agent entrypoint.

    PR1: config/logging/CLI skeleton
    PR2: add Edge HTTP API for install/debug (/health, /heartbeat)
    """
    try:
        args = build_parser().parse_args(argv)

        # Load settings from environment / .env
        cfg = cfg or EdgeSettings()

        # Setup logging using configured level
        configure_logging(cfg.log_level)

        logger.info("Edge Agent starting")
        logger.info(
            "Resolved config: site_id=%s tcp=%s:%s server=%s",
            cfg.site_id, cfg.tcp_host, cfg.tcp_port, cfg.server_base_url
        )

        if args.print_config:
            print(cfg.model_dump())
            return 0

        if args.http_serve:
            import uvicorn
            from .edge_api import create_app

            app = create_app(cfg)

            logger.info("Starting Edge HTTP API at http://%s:%s", cfg.edge_http_host, cfg.edge_http_port)
            uvicorn.run(
                app,
                host=cfg.edge_http_host,
                port=cfg.edge_http_port,
                log_level=cfg.log_level.lower(),
            )
            return 0

        logger.info("Nothing to do. Use --print-config or --http-serve.")
        return 0

    except Exception:
        # Log unexpected exceptions so the edge service is diagnosable.
        logger.exception("Edge Agent crashed due to an unexpected error")
        if cfg.debug or cfg.log_level.upper() == "DEBUG":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
