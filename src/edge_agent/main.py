from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone

from .config import EdgeSettings
from .logging import configure_logging
from .runtime_state import EdgeRuntimeState
from .sender import ServerSender

logger = logging.getLogger(__name__)


async def consume_extraction_results(worker, pipeline) -> None:
    """
    Consume ExtractionWorker results and forward usable windows into the
    blocking ML + alert pipeline without blocking the main asyncio loop.
    """
    from .video.window_extractor import WindowStatus

    while True:
        res = await worker.results.get()

        if res.status == WindowStatus.DROPPED:
            logger.warning(
                "WINDOW(skip): incident=%s camera=%s status=%s reason=%s",
                res.incident_id,
                res.camera_id,
                res.status.value,
                res.reason,
            )
            continue

        if not res.selected:
            logger.warning(
                "WINDOW(skip): incident=%s camera=%s reason=empty_selection",
                res.incident_id,
                res.camera_id,
            )
            continue

        try:
            await asyncio.to_thread(
                pipeline.process_frames,
                res.camera_id,
                list(res.selected),
            )
        except Exception:
            logger.exception(
                "Pipeline processing failed: incident=%s camera=%s",
                res.incident_id,
                res.camera_id,
            )


def heartbeat_loop(
    sender: ServerSender, interval_sec: int, stop_event: threading.Event
):
    """
    Thread target for sending heartbeats at regular intervals.
    """
    started_monotonic = time.monotonic()
    while not stop_event.is_set():
        sender.send_heartbeat(started_monotonic)
        stop_event.wait(interval_sec)


def retry_loop(sender: ServerSender, interval_sec: int, stop_event: threading.Event):
    """
    Thread target for retrying queued alerts.
    """
    while not stop_event.is_set():
        try:
            sender.retry_queued_alerts()
        except Exception:
            logger.exception("Retry loop encountered an unexpected error")
        stop_event.wait(interval_sec)


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
        "--sample-video",
        type=str,
        help="Run a CCTV sample video directly through YOLO + alert sending, without RTSP.",
    )
    parser.add_argument(
        "--tcp-listen",
        action="store_true",
        help="Start TCP motion listener and print parsed motion events.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the unified edge pipeline (RTSP + optional motion sources + alerts).",
    )
    parser.add_argument(
        "--rtsp-test",
        action="store_true",
        help="Start RTSP reader and print ring buffer size.",
    )
    parser.add_argument(
        "--motion-test",
        action="store_true",
        help="Run RTSP reader + local motion trigger (debug mode, no alerts).",
    )
    return parser


def start_edge_http_server(
    cfg: EdgeSettings, sender: ServerSender, runtime_state: EdgeRuntimeState
):
    import uvicorn

    from .edge_api import create_app

    app = create_app(cfg, sender, runtime_state)

    logger.info(
        "Starting Edge HTTP API at http://%s:%s",
        cfg.edge_http_host,
        cfg.edge_http_port,
    )

    config = uvicorn.Config(
        app,
        host=cfg.edge_http_host,
        port=cfg.edge_http_port,
        log_level=cfg.log_level.lower(),
    )
    server = uvicorn.Server(config)

    def _run_server() -> None:
        server.run()

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    startup_deadline = time.monotonic() + 10.0
    while not server.started and not server.should_exit:
        if time.monotonic() >= startup_deadline:
            logger.error("Edge HTTP API failed to start within 10s; exiting.")
            server.should_exit = True
            server_thread.join(timeout=1)
            raise RuntimeError("Edge HTTP API failed to start")
        time.sleep(0.05)

    return server, server_thread


def join_interruptible_thread(t: threading.Thread, poll: float = 0.2) -> None:
    while t.is_alive():
        t.join(timeout=poll)


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

        sender = ServerSender(cfg)
        runtime_state = EdgeRuntimeState()
        runtime_state.update(sender_status=sender.get_status())
        stop_event = threading.Event()

        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(sender, cfg.heartbeat_interval_sec, stop_event),
            daemon=True,
        )
        heartbeat_thread.start()

        retry_thread = threading.Thread(
            target=retry_loop,
            args=(sender, cfg.retry_interval_sec, stop_event),
            daemon=True,
        )
        retry_thread.start()

        def _shutdown(code: int) -> int:
            stop_event.set()
            heartbeat_thread.join(timeout=1)
            retry_thread.join(timeout=1)
            return code

        if args.http_serve:
            try:
                server, server_thread = start_edge_http_server(
                    cfg, sender, runtime_state
                )
            except KeyboardInterrupt:
                return _shutdown(0)
            except Exception:
                return _shutdown(1)

            if server.started:
                sender.set_status("online")
                runtime_state.update(sender_status=sender.get_status())

            try:
                join_interruptible_thread(server_thread)
            except KeyboardInterrupt:
                server.should_exit = True
                join_interruptible_thread(server_thread)

            if server.started:
                sender.set_status("shutting_down")
                runtime_state.update(sender_status=sender.get_status())

            return _shutdown(0)

        if args.tcp_listen:

            async def _main() -> None:
                from .triggers.tcp_trigger import TcpMotionTrigger
                from .triggers.trigger_manager import TriggerManager

                mgr = TriggerManager(
                    cooldown_sec=cfg.trigger_cooldown_sec,
                    merge_window_sec=cfg.trigger_merge_window_sec,
                )
                trigger = TcpMotionTrigger(cfg)
                await trigger.start()
                sender.set_status("online")
                runtime_state.update(sender_status=sender.get_status())
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
            return _shutdown(0)

        if args.rtsp_test:

            async def _rtsp_main() -> None:
                from .video.ring_buffer import RingBuffer
                from .video.rtsp_reader import RtspReader

                ring = RingBuffer(seconds=cfg.ring_buffer_seconds)
                reader = RtspReader(cfg, ring)
                await reader.start()
                sender.set_status("online")
                runtime_state.update(sender_status=sender.get_status())
                try:
                    while True:
                        logger.info("RingBuffer frames=%d", ring.size())
                        runtime_state.update(
                            pipeline_mode="rtsp_test",
                            stream_state=(
                                "streaming" if ring.size() > 0 else "connecting"
                            ),
                            ring_buffer_frames=ring.size(),
                            latest_frame_item=ring.latest_item(),
                            sender_status=sender.get_status(),
                        )
                        await asyncio.sleep(2)
                finally:
                    await reader.stop()

            try:
                asyncio.run(_rtsp_main())
            except KeyboardInterrupt:
                logger.info("RTSP test stopped (Ctrl+C).")
            return _shutdown(0)

        if args.motion_test:
            if not cfg.rtsp_url_low:
                logger.warning(
                    "Motion test requires RTSP_URL_LOW. Set it in env/.env and retry."
                )
                return _shutdown(0)

            async def _motion_main() -> None:
                from .triggers.incident_manager import IncidentManager
                from .triggers.local_motion_trigger import LocalMotionTrigger
                from .triggers.trigger_manager import TriggerManager
                from .video.extraction_worker import ExtractionWorker
                from .video.ring_buffer import RingBuffer
                from .video.rtsp_reader import RtspReader

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

                def ring_provider(camera_id: str):
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
                sender.set_status("online")
                runtime_state.update(sender_status=sender.get_status())

                try:
                    last_tick = datetime.now(timezone.utc)
                    while True:
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

                        runtime_state.update(
                            pipeline_mode="motion_test",
                            stream_state=state,
                            ring_buffer_frames=ring_frames,
                            latest_frame_item=ring.latest_item(),
                            sender_status=sender.get_status(),
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
            return _shutdown(0)

        if args.sample_video:
            from .ml_evaluator import MLEvaluator
            from .pipeline_runner import PipelineRunner
            from .sample_video_runner import run_sample_video

            evaluator = MLEvaluator(
                model_name=cfg.detector_model,
                weights_path=cfg.detector_weights,
                person_conf=cfg.detector_person_conf,
                vehicle_conf=cfg.detector_vehicle_conf,
                allowed_classes=cfg.detector_allowed_classes,
                include_polygons=cfg.motion_include_polygons,
                exclude_polygons=cfg.motion_exclude_polygons,
            )

            image_output_dir = (
                cfg.shared_storage_root.strip() or "storage/ws_alert_images"
            )

            pipeline = PipelineRunner(
                evaluator=evaluator,
                sender=sender,
                image_output_dir=image_output_dir,
            )

            sender.set_status("online")
            runtime_state.update(sender_status=sender.get_status())

            run_sample_video(
                video_path=args.sample_video,
                pipeline=pipeline,
                sender=sender,
                camera_id=cfg.sample_camera_id,
                window_sec=cfg.sample_window_sec,
                stride_sec=cfg.sample_stride_sec,
                target_fps=cfg.sample_target_fps,
                max_frames=cfg.sample_max_frames,
            )

            sender.set_status("shutting_down")
            runtime_state.update(sender_status=sender.get_status())
            return _shutdown(0)

        if args.run:

            async def _run_main() -> None:
                from .ml_evaluator import MLEvaluator
                from .pipeline_runner import PipelineRunner
                from .triggers.incident_manager import IncidentManager
                from .triggers.local_motion_trigger import LocalMotionTrigger
                from .triggers.tcp_trigger import TcpMotionTrigger
                from .triggers.trigger_manager import TriggerManager
                from .video.extraction_worker import ExtractionWorker
                from .video.ring_buffer import RingBuffer
                from .video.rtsp_reader import RtspReader

                if not cfg.enable_tcp_motion and not cfg.enable_local_motion:
                    logger.warning(
                        "No motion source enabled. Set ENABLE_TCP_MOTION and/or ENABLE_LOCAL_MOTION."
                    )
                    return

                ring = RingBuffer(seconds=cfg.ring_buffer_seconds)
                reader = RtspReader(cfg, ring)
                server = None
                server_thread = None

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

                def ring_provider(camera_id: str):
                    return ring

                worker = ExtractionWorker(
                    ring_provider=ring_provider,
                    target_fps=cfg.window_target_fps,
                    max_frames=cfg.window_max_frames,
                    wait_grace_sec=cfg.window_wait_grace_sec,
                )

                evaluator = MLEvaluator(
                    model_name=cfg.detector_model,
                    weights_path=cfg.detector_weights,
                    person_conf=cfg.detector_person_conf,
                    vehicle_conf=cfg.detector_vehicle_conf,
                    allowed_classes=cfg.detector_allowed_classes,
                    include_polygons=cfg.motion_include_polygons,
                    exclude_polygons=cfg.motion_exclude_polygons,
                )

                runtime_state.update(
                    pipeline_mode="running",
                    stream_state="connecting",
                    ring_buffer_frames=0,
                    sender_status=sender.get_status(),
                    latest_frame_item=None,
                    last_error=None,
                )

                image_output_dir = (
                    cfg.shared_storage_root.strip() or "storage/ws_alert_images"
                )

                pipeline = PipelineRunner(
                    evaluator=evaluator,
                    sender=sender,
                    image_output_dir=image_output_dir,
                )

                pipeline_task = asyncio.create_task(
                    consume_extraction_results(worker, pipeline),
                    name="pipeline-consumer",
                )

                tcp_trigger = TcpMotionTrigger(cfg) if cfg.enable_tcp_motion else None

                def on_motion(evt, accepted: bool) -> None:
                    incidents.ingest(evt, accepted=accepted)

                local_trigger = (
                    LocalMotionTrigger(cfg, ring, mgr, on_motion=on_motion)
                    if cfg.enable_local_motion
                    else None
                )

                if not cfg.rtsp_url_low:
                    logger.warning(
                        "RTSP_URL_LOW is not set. Windows will likely be DROPPED (no frames)."
                    )

                logger.info(
                    "Unified run enabled: tcp_motion=%s local_motion=%s",
                    cfg.enable_tcp_motion,
                    cfg.enable_local_motion,
                )

                await reader.start()
                await worker.start()

                try:
                    server, server_thread = start_edge_http_server(
                        cfg, sender, runtime_state
                    )

                    if tcp_trigger is not None:
                        await tcp_trigger.start()

                    if local_trigger is not None:
                        await local_trigger.start()

                    sender.set_status("online")
                    runtime_state.update(sender_status=sender.get_status())

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

                    while True:
                        if tcp_trigger is not None:
                            try:
                                evt = await asyncio.wait_for(
                                    tcp_trigger.queue.get(),
                                    timeout=cfg.incident_tick_interval_sec,
                                )
                                accepted = mgr.accept(evt)
                                incidents.ingest(evt, accepted=accepted)

                                if accepted:
                                    logger.info(
                                        "TCP MOTION(accepted): camera_id=%s policy=%s user=%s",
                                        evt.camera_id,
                                        evt.policy_name,
                                        evt.user_string,
                                    )
                                else:
                                    logger.debug(
                                        "TCP MOTION(dropped): camera_id=%s",
                                        evt.camera_id,
                                    )
                            except asyncio.TimeoutError:
                                pass
                        else:
                            await asyncio.sleep(cfg.incident_tick_interval_sec)

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

                        latest_item = ring.latest_item()
                        ring_frames = ring.size()

                        runtime_state.update(
                            stream_state=(
                                "streaming" if ring_frames > 0 else "connecting"
                            ),
                            ring_buffer_frames=ring_frames,
                            latest_frame_item=latest_item,
                            sender_status=sender.get_status(),
                        )

                finally:
                    pipeline_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await pipeline_task

                    if local_trigger is not None:
                        await local_trigger.stop()

                    if tcp_trigger is not None:
                        await tcp_trigger.stop()

                    await worker.stop()
                    await reader.stop()

                    if server is not None:
                        server.should_exit = True
                    if server_thread is not None:
                        join_interruptible_thread(server_thread)

                    sender.set_status("shutting_down")
                    runtime_state.update(
                        pipeline_mode="idle",
                        stream_state="stopped",
                        ring_buffer_frames=0,
                        sender_status=sender.get_status(),
                        latest_frame_item=None,
                    )

            try:
                asyncio.run(_run_main())
            except KeyboardInterrupt:
                logger.info("--run stopped (Ctrl+C).")
            return _shutdown(0)

        logger.info(
            "Nothing to do. Use --print-config, --http-serve, --sample-video, --tcp-listen, --run, --motion-test."
        )
        return _shutdown(0)

    except Exception:
        logger.exception("Edge Agent crashed due to an unexpected error")
        if "stop_event" in locals():
            stop_event.set()
            heartbeat_thread.join(timeout=1)
            retry_thread.join(timeout=1)

        debug_mode = bool(getattr(cfg, "debug", False)) or (
            getattr(cfg, "log_level", "").upper() == "DEBUG"
        )
        if debug_mode:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
