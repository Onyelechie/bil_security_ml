from __future__ import annotations

import argparse
import logging

from .config import EdgeSettings
from .logging import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BIL Security ML - Edge Agent (Area B)")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved configuration and exit.",
    )
    parser.add_argument(
        "--http-serve",
        action="store_true",
        help="Start Edge HTTP API server (/health, /heartbeat).",
    )
    parser.add_argument(
        "--tcp-listen",
        action="store_true",
        help="Start TCP motion listener and print parsed motion events.",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full edge pipeline (placeholder for now).",
    )

    parser.add_argument(
        "--rtsp-test",
        action="store_true",
        help="Start RTSP reader and print ring buffer size.",
    )

    parser.add_argument(
        "--motion-test",
        action="store_true",
        help="Run RTSP reader + local motion trigger (live test).",
    )

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
            cfg.site_id,
            cfg.tcp_host,
            cfg.tcp_port,
            cfg.server_base_url,
        )

        if args.print_config:
            print(cfg.model_dump())
            return 0

        if args.http_serve:
            import uvicorn
            from .edge_api import create_app

            app = create_app(cfg)

            logger.info(
                "Starting Edge HTTP API at http://%s:%s",
                cfg.edge_http_host,
                cfg.edge_http_port,
            )
            uvicorn.run(
                app,
                host=cfg.edge_http_host,
                port=cfg.edge_http_port,
                log_level=cfg.log_level.lower(),
            )
            return 0

        if args.tcp_listen:
            import asyncio

            from .triggers.tcp_trigger import TcpMotionTrigger
            from .triggers.trigger_manager import TriggerManager

            # Create the manager once (config-driven, not hardcoded)
            mgr = TriggerManager(
                cooldown_sec=cfg.trigger_cooldown_sec,
                merge_window_sec=cfg.trigger_merge_window_sec,
            )

            async def _main() -> None:
                trigger = TcpMotionTrigger(cfg)
                await trigger.start()
                try:
                    while True:
                        try:
                            evt = await trigger.queue.get()
                        except asyncio.CancelledError:
                            break

                        accepted = mgr.accept(evt)
                        if accepted:
                            logger.info(
                                "MOTION(accepted): source=%s camera_id=%s camera_name=%s policy=%s user=%s",
                                evt.source,
                                evt.camera_id,
                                evt.camera_name,
                                evt.policy_name,
                                evt.user_string,
                            )
                        else:
                            logger.debug(
                                "MOTION(dropped): source=%s camera_id=%s",
                                evt.source,
                                evt.camera_id,
                            )
                finally:
                    await trigger.stop()

            try:
                asyncio.run(_main())
            except KeyboardInterrupt:
                logger.info("TCP listener stopped (Ctrl+C).")
            return 0

        if args.rtsp_test:
            import asyncio
            from .video.ring_buffer import RingBuffer
            from .video.rtsp_reader import RtspReader

            async def _rtsp_main() -> None:
                ring = RingBuffer(seconds=cfg.ring_buffer_seconds)
                reader = RtspReader(cfg, ring)
                await reader.start()
                try:
                    while True:
                        logger.info("RingBuffer frames=%d", ring.size())
                        await asyncio.sleep(2)
                finally:
                    await reader.stop()

            asyncio.run(_rtsp_main())
            return 0

        if args.run:
            logger.info("Full edge pipeline not implemented yet. (Step 1 placeholder)")
            return 0

        if args.motion_test:
            import asyncio

            from .triggers.local_motion_trigger import LocalMotionTrigger
            from .triggers.trigger_manager import TriggerManager

            async def _motion_main() -> None:
                ring = RingBuffer(seconds=cfg.ring_buffer_seconds)

                mgr = TriggerManager(
                    cooldown_sec=cfg.trigger_cooldown_sec,
                    merge_window_sec=cfg.trigger_merge_window_sec,
                )

                reader = RtspReader(cfg, ring)
                local = LocalMotionTrigger(cfg, ring, mgr)

                await reader.start()
                task_local = asyncio.create_task(local.run(), name="local-motion")

                try:
                    while True:
                        logger.info("RTSP live: ring_frames=%d", ring.size())
                        await asyncio.sleep(2)
                finally:
                    task_local.cancel()
                    await reader.stop()

            try:
                asyncio.run(_motion_main())
            except KeyboardInterrupt:
                logger.info("Motion test stopped (Ctrl+C).")
            return 0

        logger.info("Nothing to do. Use --print-config, --http-serve, --tcp-listen, or --run.")
        return 0

    except Exception:
        # Log unexpected exceptions so the edge service is diagnosable.
        logger.exception("Edge Agent crashed due to an unexpected error")

        debug_mode = bool(getattr(cfg, "debug", False)) or (getattr(cfg, "log_level", "").upper() == "DEBUG")
        if debug_mode:
            raise

        return 1


if __name__ == "__main__":
    raise SystemExit(run())
