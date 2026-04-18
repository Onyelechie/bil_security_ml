from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone

from .config import EdgeSettings
from .logging import configure_logging
from .runtime_state import EdgeRuntimeState
from .sender import ServerSender
from .settings_store import load_effective_settings

logger = logging.getLogger(__name__)
ShutdownFn = Callable[[int], int]


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


def _start_uvicorn_server(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    *,
    startup_log_message: str | None = None,
    startup_error_message: str,
    startup_error_log_message: str | None = None,
):
    import uvicorn

    from .edge_api import create_app

    if startup_log_message:
        logger.info(startup_log_message, cfg.edge_http_host, cfg.edge_http_port)

    app = create_app(cfg, sender, runtime_state)
    config = uvicorn.Config(
        app,
        host=cfg.edge_http_host,
        port=cfg.edge_http_port,
        log_level=cfg.log_level.lower(),
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run_server() -> None:
        server.run()

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    startup_deadline = time.monotonic() + 10.0
    while not server.started and not server.should_exit:
        if time.monotonic() >= startup_deadline:
            if startup_error_log_message:
                logger.error(startup_error_log_message)
            server.should_exit = True
            server_thread.join(timeout=1)
            raise RuntimeError(startup_error_message)
        time.sleep(0.05)

    return server, server_thread


def _start_console_server(
    cfg: EdgeSettings, sender: ServerSender, runtime_state: EdgeRuntimeState
):
    return _start_uvicorn_server(
        cfg,
        sender,
        runtime_state,
        startup_error_message="Edge console failed to start within 10s",
    )


def start_edge_http_server(
    cfg: EdgeSettings, sender: ServerSender, runtime_state: EdgeRuntimeState
):
    return _start_uvicorn_server(
        cfg,
        sender,
        runtime_state,
        startup_log_message="Starting Edge HTTP API at http://%s:%s",
        startup_error_message="Edge HTTP API failed to start",
        startup_error_log_message="Edge HTTP API failed to start within 10s; exiting.",
    )


def _stop_console_server(server, server_thread) -> None:
    if server is None or server_thread is None:
        return
    server.should_exit = True
    while server_thread.is_alive():
        server_thread.join(timeout=0.2)


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

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--http-serve",
        action="store_true",
        help="Start Edge HTTP API server (/health, /heartbeat).",
    )
    mode.add_argument(
        "--sample-video",
        type=str,
        help="Run a CCTV sample video directly through YOLO + alert sending, without RTSP.",
    )
    mode.add_argument(
        "--tcp-listen",
        action="store_true",
        help="Start TCP motion listener and print parsed motion events.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run the unified edge pipeline (RTSP + optional motion sources + alerts).",
    )
    mode.add_argument(
        "--rtsp-test",
        action="store_true",
        help="Start RTSP reader and print ring buffer size.",
    )
    mode.add_argument(
        "--motion-test",
        action="store_true",
        help="Run RTSP reader + local motion trigger (debug mode, no alerts).",
    )

    return parser


def join_interruptible_thread(t: threading.Thread, poll: float = 0.2) -> None:
    while t.is_alive():
        t.join(timeout=poll)


def _handle_http_serve_mode(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    shutdown: ShutdownFn,
) -> int:
    try:
        server, server_thread = start_edge_http_server(cfg, sender, runtime_state)
    except KeyboardInterrupt:
        return shutdown(0)
    except Exception:
        return shutdown(1)

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

    return shutdown(0)


def _handle_sample_video_mode(
    video_path: str,
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    shutdown: ShutdownFn,
) -> int:
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

    image_output_dir = cfg.shared_storage_root.strip() or "storage/ws_alert_images"

    pipeline = PipelineRunner(
        evaluator=evaluator,
        sender=sender,
        image_output_dir=image_output_dir,
    )

    sender.set_status("online")
    runtime_state.update(sender_status=sender.get_status())

    run_sample_video(
        video_path=video_path,
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
    return shutdown(0)


def _handle_run_mode(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    shutdown: ShutdownFn,
) -> int:
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

        active_cfg = cfg
        restart_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def request_restart() -> dict:
            loop.call_soon_threadsafe(restart_event.set)
            return {
                "accepted": True,
                "message": "Restart requested. The edge pipeline will rebuild shortly.",
            }

        runtime_state.set_restart_pipeline_fn(request_restart)

        try:
            while True:
                current_cfg = active_cfg

                if (
                    not current_cfg.enable_tcp_motion
                    and not current_cfg.enable_local_motion
                ):
                    logger.warning(
                        "No motion source enabled. Set ENABLE_TCP_MOTION and/or ENABLE_LOCAL_MOTION."
                    )
                    return

                sender.settings = current_cfg
                sender.queue_dir = current_cfg.offline_queue_dir

                ring = RingBuffer(seconds=current_cfg.ring_buffer_seconds)

                def on_preview_frame(item):
                    runtime_state.update(
                        latest_frame_item=item,
                        stream_state="streaming",
                    )

                reader = RtspReader(
                    current_cfg,
                    ring,
                    on_frame=on_preview_frame,
                )

                mgr = TriggerManager(
                    cooldown_sec=current_cfg.trigger_cooldown_sec,
                    merge_window_sec=current_cfg.trigger_merge_window_sec,
                )

                incidents = IncidentManager(
                    pre_sec=current_cfg.window_pre_sec,
                    post_sec=current_cfg.window_post_sec,
                    quiet_sec=current_cfg.incident_quiet_sec,
                    max_incident_sec=current_cfg.incident_max_sec,
                )

                def ring_provider(camera_id: str):
                    return ring

                worker = ExtractionWorker(
                    ring_provider=ring_provider,
                    target_fps=current_cfg.window_target_fps,
                    max_frames=current_cfg.window_max_frames,
                    wait_grace_sec=current_cfg.window_wait_grace_sec,
                )

                evaluator = MLEvaluator(
                    model_name=current_cfg.detector_model,
                    weights_path=current_cfg.detector_weights,
                    person_conf=current_cfg.detector_person_conf,
                    vehicle_conf=current_cfg.detector_vehicle_conf,
                    allowed_classes=current_cfg.detector_allowed_classes,
                    include_polygons=current_cfg.motion_include_polygons,
                    exclude_polygons=current_cfg.motion_exclude_polygons,
                )

                image_output_dir = (
                    current_cfg.shared_storage_root.strip() or "storage/ws_alert_images"
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

                tcp_trigger = (
                    TcpMotionTrigger(current_cfg)
                    if current_cfg.enable_tcp_motion
                    else None
                )

                def on_motion(evt, accepted: bool) -> None:
                    incidents.ingest(evt, accepted=accepted)
                    if accepted:
                        runtime_state.update(last_motion_at=datetime.now(timezone.utc))

                local_trigger = (
                    LocalMotionTrigger(current_cfg, ring, mgr, on_motion=on_motion)
                    if current_cfg.enable_local_motion
                    else None
                )

                if local_trigger is not None:
                    runtime_state.set_apply_settings_fn(
                        local_trigger.apply_runtime_settings
                    )
                else:
                    runtime_state.set_apply_settings_fn(None)

                if not current_cfg.rtsp_url_low:
                    logger.warning(
                        "RTSP_URL_LOW is not set. Windows will likely be DROPPED (no frames)."
                    )

                logger.info(
                    "Unified run enabled: tcp_motion=%s local_motion=%s",
                    current_cfg.enable_tcp_motion,
                    current_cfg.enable_local_motion,
                )

                runtime_state.update(
                    pipeline_mode="starting",
                    stream_state="connecting",
                    ring_buffer_frames=0,
                    latest_frame_item=None,
                    sender_status=sender.get_status(),
                    last_error=None,
                )

                await reader.start()
                await worker.start()

                if tcp_trigger is not None:
                    await tcp_trigger.start()

                if local_trigger is not None:
                    await local_trigger.start()

                sender.set_status("online")
                runtime_state.update(
                    pipeline_mode="running",
                    sender_status=sender.get_status(),
                )

                max_window_s = (
                    current_cfg.window_pre_sec
                    + current_cfg.incident_max_sec
                    + current_cfg.window_post_sec
                )
                if max_window_s > current_cfg.ring_buffer_seconds:
                    logger.warning(
                        (
                            "Configured window span (%.1fs) > ring_buffer_seconds (%ds). "
                            "Expect PARTIAL windows unless ring buffer is increased."
                        ),
                        max_window_s,
                        current_cfg.ring_buffer_seconds,
                    )

                last_tick = datetime.now(timezone.utc)
                should_restart = False

                try:
                    while True:
                        if restart_event.is_set():
                            restart_event.clear()
                            sender.set_status("restarting")
                            runtime_state.update(
                                pipeline_mode="restarting",
                                sender_status=sender.get_status(),
                            )
                            logger.info("Restart requested from edge console")
                            should_restart = True
                            break

                        if tcp_trigger is not None:
                            try:
                                evt = await asyncio.wait_for(
                                    tcp_trigger.queue.get(),
                                    timeout=current_cfg.incident_tick_interval_sec,
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
                                    runtime_state.update(
                                        last_motion_at=datetime.now(timezone.utc)
                                    )
                                else:
                                    logger.debug(
                                        "TCP MOTION(dropped): camera_id=%s",
                                        evt.camera_id,
                                    )
                            except asyncio.TimeoutError:
                                pass
                        else:
                            await asyncio.sleep(current_cfg.incident_tick_interval_sec)

                        now = datetime.now(timezone.utc)
                        if (
                            now - last_tick
                        ).total_seconds() >= current_cfg.incident_tick_interval_sec:
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

                        latest_item = reader.latest_item() or ring.latest_item()
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
                    runtime_state.set_apply_settings_fn(None)

                    pipeline_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await pipeline_task

                    if local_trigger is not None:
                        await local_trigger.stop()

                    if tcp_trigger is not None:
                        await tcp_trigger.stop()

                    await worker.stop()
                    await reader.stop()

                if should_restart:
                    active_cfg = load_effective_settings()
                    logger.info(
                        "Reloaded edge settings from local saved state after restart request"
                    )
                    continue

                sender.set_status("shutting_down")
                runtime_state.update(
                    pipeline_mode="idle",
                    stream_state="stopped",
                    ring_buffer_frames=0,
                    sender_status=sender.get_status(),
                    latest_frame_item=None,
                )
                break

        finally:
            runtime_state.set_apply_settings_fn(None)
            runtime_state.set_restart_pipeline_fn(None)

    console_server = None
    console_thread = None
    try:
        console_server, console_thread = _start_console_server(
            cfg, sender, runtime_state
        )
        logger.info(
            "Edge console available at http://%s:%s",
            cfg.edge_http_host,
            cfg.edge_http_port,
        )
        asyncio.run(_run_main())
    except KeyboardInterrupt:
        logger.info("--run stopped (Ctrl+C).")
    finally:
        _stop_console_server(console_server, console_thread)

    return shutdown(0)


def _handle_tcp_listen_mode(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    shutdown: ShutdownFn,
) -> int:
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
    return shutdown(0)


def _handle_rtsp_test_mode(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    shutdown: ShutdownFn,
) -> int:
    async def _rtsp_main() -> None:
        from .video.ring_buffer import RingBuffer
        from .video.rtsp_reader import RtspReader

        ring = RingBuffer(seconds=cfg.ring_buffer_seconds)

        def on_preview_frame(item):
            runtime_state.update(
                latest_frame_item=item,
                stream_state="streaming",
            )

        reader = RtspReader(cfg, ring, on_frame=on_preview_frame)
        await reader.start()
        sender.set_status("online")
        runtime_state.update(sender_status=sender.get_status())
        try:
            while True:
                logger.info("RingBuffer frames=%d", ring.size())
                latest_item = reader.latest_item() or ring.latest_item()

                runtime_state.update(
                    pipeline_mode="rtsp_test",
                    stream_state=(
                        "streaming" if latest_item is not None else "connecting"
                    ),
                    ring_buffer_frames=ring.size(),
                    latest_frame_item=latest_item,
                    sender_status=sender.get_status(),
                )
                await asyncio.sleep(2)
        finally:
            await reader.stop()

    try:
        asyncio.run(_rtsp_main())
    except KeyboardInterrupt:
        logger.info("RTSP test stopped (Ctrl+C).")
    return shutdown(0)


def _handle_motion_test_mode(
    cfg: EdgeSettings,
    sender: ServerSender,
    runtime_state: EdgeRuntimeState,
    shutdown: ShutdownFn,
) -> int:
    if not cfg.rtsp_url_low:
        logger.warning(
            "Motion test requires RTSP_URL_LOW. Set it in env/.env and retry."
        )
        return shutdown(0)

    async def _motion_main() -> None:
        from .triggers.incident_manager import IncidentManager
        from .triggers.local_motion_trigger import LocalMotionTrigger
        from .triggers.trigger_manager import TriggerManager
        from .video.extraction_worker import ExtractionWorker
        from .video.ring_buffer import RingBuffer
        from .video.rtsp_reader import RtspReader

        ring = RingBuffer(seconds=cfg.ring_buffer_seconds)

        def on_preview_frame(item):
            runtime_state.update(
                latest_frame_item=item,
                stream_state="streaming",
            )

        reader = RtspReader(cfg, ring, on_frame=on_preview_frame)

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
                if (now - last_tick).total_seconds() >= cfg.incident_tick_interval_sec:
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

                latest_item = reader.latest_item() or ring.latest_item()

                runtime_state.update(
                    pipeline_mode="motion_test",
                    stream_state=state,
                    ring_buffer_frames=ring_frames,
                    latest_frame_item=latest_item,
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
    return shutdown(0)


def run(argv: list[str] | None = None, cfg: EdgeSettings | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        cfg = cfg or load_effective_settings()
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
            return _handle_http_serve_mode(cfg, sender, runtime_state, _shutdown)

        if args.tcp_listen:
            return _handle_tcp_listen_mode(cfg, sender, runtime_state, _shutdown)

        if args.rtsp_test:
            return _handle_rtsp_test_mode(cfg, sender, runtime_state, _shutdown)

        if args.motion_test:
            return _handle_motion_test_mode(cfg, sender, runtime_state, _shutdown)

        if args.sample_video:
            return _handle_sample_video_mode(
                args.sample_video,
                cfg,
                sender,
                runtime_state,
                _shutdown,
            )

        if args.run:
            return _handle_run_mode(cfg, sender, runtime_state, _shutdown)

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
