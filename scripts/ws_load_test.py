#!/usr/bin/env python3
"""Simple WebSocket load test for /ws/alerts.

Example:
  python scripts/ws_load_test.py --clients 100 --messages-per-client 20
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class WorkerResult:
    sent: int = 0
    ack: int = 0
    error_frames: int = 0
    bad_frames: int = 0
    exceptions: int = 0
    latencies_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = []


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a WebSocket alert ingestion load test.")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/alerts", help="WebSocket endpoint URL")
    p.add_argument("--clients", type=int, default=50, help="Number of concurrent client connections")
    p.add_argument("--messages-per-client", type=int, default=10, help="Messages sent by each client")
    p.add_argument("--stagger-ms", type=int, default=0, help="Delay between client starts in milliseconds")
    p.add_argument("--connect-timeout", type=float, default=10.0, help="Connect timeout in seconds")
    p.add_argument("--recv-timeout", type=float, default=10.0, help="Receive timeout in seconds")
    p.add_argument("--max-size", type=int, default=2_000_000, help="Max inbound frame size in bytes")
    return p


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


async def _worker(
    idx: int,
    args: argparse.Namespace,
    websockets_mod,
) -> WorkerResult:
    result = WorkerResult()
    try:
        async with websockets_mod.connect(
            args.url,
            open_timeout=args.connect_timeout,
            max_size=args.max_size,
        ) as ws:
            # Expect initial "connected" frame from server.
            await asyncio.wait_for(ws.recv(), timeout=args.recv_timeout)

            for msg_idx in range(args.messages_per_client):
                payload = {
                    "site_id": f"load_site_{idx}",
                    "camera_id": f"cam_{msg_idx}",
                    "edge_pc_id": f"edge-load-{idx}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detections": [{"class": "person", "confidence": 0.95}],
                    "image_path": None,
                }

                t0 = time.perf_counter()
                await ws.send(json.dumps(payload))
                raw = await asyncio.wait_for(ws.recv(), timeout=args.recv_timeout)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                result.sent += 1
                result.latencies_ms.append(latency_ms)

                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    result.bad_frames += 1
                    continue

                frame_type = frame.get("type")
                if frame_type == "ack":
                    result.ack += 1
                elif frame_type == "error":
                    result.error_frames += 1
                else:
                    result.bad_frames += 1
    except Exception:
        result.exceptions += 1

    return result


async def _run(args: argparse.Namespace) -> int:
    if args.clients < 1:
        raise ValueError("--clients must be >= 1")
    if args.messages_per_client < 1:
        raise ValueError("--messages-per-client must be >= 1")
    if args.stagger_ms < 0:
        raise ValueError("--stagger-ms must be >= 0")

    websockets_mod = importlib.import_module("websockets")

    started = time.perf_counter()
    tasks: list[asyncio.Task] = []
    for idx in range(args.clients):
        tasks.append(asyncio.create_task(_worker(idx, args, websockets_mod)))
        if args.stagger_ms:
            await asyncio.sleep(args.stagger_ms / 1000.0)

    results = await asyncio.gather(*tasks)
    duration_s = time.perf_counter() - started

    total_sent = sum(r.sent for r in results)
    total_ack = sum(r.ack for r in results)
    total_errors = sum(r.error_frames for r in results)
    total_bad = sum(r.bad_frames for r in results)
    total_exceptions = sum(r.exceptions for r in results)
    latencies = [lat for r in results for lat in r.latencies_ms]
    expected = args.clients * args.messages_per_client

    print(f"target_messages={expected}")
    print(f"sent_messages={total_sent}")
    print(f"ack_messages={total_ack}")
    print(f"error_frames={total_errors}")
    print(f"bad_frames={total_bad}")
    print(f"worker_exceptions={total_exceptions}")
    print(f"duration_s={duration_s:.2f}")
    if duration_s > 0:
        print(f"throughput_msgs_per_s={total_sent / duration_s:.2f}")

    if latencies:
        print(f"latency_ms_p50={_percentile(latencies, 50):.2f}")
        print(f"latency_ms_p95={_percentile(latencies, 95):.2f}")
        print(f"latency_ms_p99={_percentile(latencies, 99):.2f}")
        print(f"latency_ms_max={max(latencies):.2f}")
        print(f"latency_ms_mean={statistics.mean(latencies):.2f}")

    # Non-zero exit for meaningful failure conditions.
    if total_ack != expected or total_exceptions > 0:
        return 1
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except ModuleNotFoundError as exc:
        if exc.name == "websockets":
            print("Missing dependency: websockets. Install with `pip install websockets`.")
            return 2
        raise
    except ValueError as exc:
        print(f"Invalid argument: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

