from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time

from .sender import ServerSender
from datetime import datetime, timezone

from .config import EdgeSettings
from .logging import configure_logging

logger = logging.getLogger(__name__)


def heartbeat_loop(sender: ServerSender, interval_sec: int):
    """
    Thread target for sending heartbeats at regular intervals.
    """
    started_monotonic = time.monotonic()
    while True:
        sender.send_heartbeat(started_monotonic)
        time.sleep(interval_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BIL Security ML - Edge Agent (Area B)"
    )
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
        help="Run the PR6 pipeline (incident merge + window extraction).",
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
    try:
        args = build_parser().parse_args(argv)
        cfg = cfg or EdgeSettings()
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

        # Start heartbeat thread (runs regardless of mode to ensure server knows we're alive)
        sender = ServerSender(cfg)
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(sender, cfg.heartbeat_interval_sec),
            daemon=True
        )
        heartbeat_thread.start()

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
            from .triggers.trigger_manager import TriggerManager

            mgr = TriggerManager(
                cooldown_sec=cfg.trigger_cooldown_sec,
                merge_window_sec=cfg.trigger_merge_window_sec,
            )

            async def _main() -> None:
                trigger = TcpMotionTrigger(cfg)
                await trigger.start()
                try:
                    while True:
                        evt = await trigger.queue.get()
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

            try:
                asyncio.run(_rtsp_main())
            except KeyboardInterrupt:
                logger.info("RTSP test stopped (Ctrl+C).")
            return 0

        if args.motion_test:
            from .triggers.local_motion_trigger import LocalMotionTrigger
            from .triggers.trigger_manager import TriggerManager

            if not cfg.rtsp_url_low:
                logger.warning(
                    "Motion test requires RTSP_URL_LOW. Set it in env/.env and retry."
                )
                return 0

            async def _motion_main() -> None:
                ring = RingBuffer(seconds=cfg.ring_buffer_seconds)
                reader = RtspReader(cfg, ring)

                mgr = TriggerManager(
                    cooldown_sec=cfg.trigger_cooldown_sec,
                    merge_window_sec=cfg.trigger_merge_window_sec,
                )
                incidents = IncidentManager(
                    pre_sec=cfg.window_pre_sec,
                    post_sec=cfg.window_post_sec,
                    quiet_sec=cfg.incident_quiet_sec,
                    max_incident_sec=cfg.incident_max_sec,
                )

                # Single-ring provider for now (multi-cam ready)
                def ring_provider(camera_id: str) -> RingBuffer | None:
                    return ring

                worker = ExtractionWorker(
                    ring_provider=ring_provider,
                    target_fps=cfg.window_target_fps,
                    max_frames=cfg.window_max_frames,
                    wait_grace_sec=cfg.window_wait_grace_sec,
                )

                def on_motion(evt, accepted: bool) -> None:
                    incidents.ingest(evt, accepted=accepted)

                motion = LocalMotionTrigger(cfg, ring, mgr, on_motion=on_motion)

                await reader.start()
                await worker.start()
                await motion.start()

                try:
                    last_tick = datetime.now(timezone.utc)
                    while True:
                        # tick incidents even if no new events
                        now = datetime.now(timezone.utc)
                        if (
                            now - last_tick
                        ).total_seconds() >= cfg.incident_tick_interval_sec:
                            for job in incidents.tick(now):
                                await worker.enqueue(job)
                            last_tick = now

                        ring_frames = ring.size()
                        state = "connecting" if ring_frames == 0 else "streaming"
                        logger.info(
                            "RTSP live: state=%s ring_frames=%d active_incidents=%d",
                            state,
                            ring_frames,
                            incidents.active_incidents(),
                        )
                        await asyncio.sleep(2.0)

                finally:
                    await motion.stop()
                    await worker.stop()
                    await reader.stop()

            try:
                asyncio.run(_motion_main())
            except KeyboardInterrupt:
                logger.info("Motion test stopped (Ctrl+C).")
            return 0

        if args.run:
            from .triggers.incident_manager import IncidentManager
            from .triggers.tcp_trigger import TcpMotionTrigger
            from .triggers.trigger_manager import TriggerManager
            from .video.extraction_worker import ExtractionWorker
            from .video.ring_buffer import RingBuffer
            from .video.rtsp_reader import RtspReader

            async def _run_main() -> None:
                ring = RingBuffer(seconds=cfg.ring_buffer_seconds)
                reader = RtspReader(cfg, ring)

                trigger = TcpMotionTrigger(cfg)
                mgr = TriggerManager(
                    cooldown_sec=cfg.trigger_cooldown_sec,
                    merge_window_sec=cfg.trigger_merge_window_sec,
                )

                incidents = IncidentManager(
                    pre_sec=cfg.window_pre_sec,
                    post_sec=cfg.window_post_sec,
                    quiet_sec=cfg.incident_quiet_sec,
                    max_incident_sec=cfg.incident_max_sec,
                )

                # Single-ring provider for now (multi-cam ready: switch to dict[camera_id, RingBuffer])
                def ring_provider(camera_id: str) -> RingBuffer | None:
                    return ring

                worker = ExtractionWorker(
                    ring_provider=ring_provider,
                    target_fps=cfg.window_target_fps,
                    max_frames=cfg.window_max_frames,
                    wait_grace_sec=cfg.window_wait_grace_sec,
                )

                if not cfg.rtsp_url_low:
                    logger.warning(
                        "RTSP_URL_LOW is not set. Windows will likely be DROPPED (no frames)."
                    )

                await reader.start()
                await trigger.start()
                await worker.start()

                # Warn if window could exceed ring size
                max_window_s = (
                    cfg.window_pre_sec + cfg.incident_max_sec + cfg.window_post_sec
                )
                if max_window_s > cfg.ring_buffer_seconds:
                    logger.warning(
                        (
                            "Configured window span (%.1fs) > ring_buffer_seconds (%ds). "
                            "Expect PARTIAL windows unless ring buffer is increased."
                        ),
                        max_window_s,
                        cfg.ring_buffer_seconds,
                    )

                last_tick = datetime.now(timezone.utc)

                try:
                    while True:
                        # Pull motion events with a short timeout so we can tick even when quiet
                        try:
                            evt = await asyncio.wait_for(
                                trigger.queue.get(),
                                timeout=cfg.incident_tick_interval_sec,
                            )
                            accepted = mgr.accept(evt)
                            incidents.ingest(evt, accepted=accepted)

                            if accepted:
                                logger.info(
                                    "INCIDENT_MOTION(accepted): source=%s camera_id=%s policy=%s user=%s",
                                    evt.source,
                                    evt.camera_id,
                                    evt.policy_name,
                                    evt.user_string,
                                )
                        except asyncio.TimeoutError:
                            pass

                        now = datetime.now(timezone.utc)
                        if (
                            now - last_tick
                        ).total_seconds() >= cfg.incident_tick_interval_sec:
                            jobs = incidents.tick(now)
                            for job in jobs:
                                logger.info(
                                    "INCIDENT_FINALIZE(enqueue): incident=%s camera=%s reason=%s span=%.1fs",
                                    job.incident_id,
                                    job.camera_id,
                                    job.reason,
                                    (job.window_end - job.window_start).total_seconds(),
                                )
                                await worker.enqueue(job)
                            last_tick = now

                finally:
                    await worker.stop()
                    await trigger.stop()
                    await reader.stop()

            try:
                asyncio.run(_run_main())
            except KeyboardInterrupt:
                logger.info("--run stopped (Ctrl+C).")
            return 0

        logger.info(
            "Nothing to do. Use --print-config, --http-serve, --tcp-listen, --run, --motion-test."
        )
        return 0

    except Exception:
        logger.exception("Edge Agent crashed due to an unexpected error")
        debug_mode = bool(getattr(cfg, "debug", False)) or (
            getattr(cfg, "log_level", "").upper() == "DEBUG"
        )
        if debug_mode:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
