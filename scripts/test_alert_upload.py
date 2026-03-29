"""CLI test uploader for signed alert flow.

Usage examples:
    python scripts/test_alert_upload.py --image "/path/to/img.jpg"

The script will:
 - wait for the server to become healthy
 - provision a test device key (enroll or rotate) via admin API
 - post a signed heartbeat to register the site/edge
 - upload the provided image to /api/alerts/upload with a valid signature
 - optionally set site retention when --set-retention is provided
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import requests

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import \
        Ed25519PrivateKey
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: cryptography is required for request signing. "
        "Install with `pip install cryptography`."
    ) from exc


BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
DEFAULT_IMAGE_PATH = (
    r"C:\Users\ebere\Documents\bil_security_ml\storage\alert_images\test_site\\"
    r"test_site_cam_01_test-edge-1_20260322T025707090673Z_person-0_95.jpg"
)
REQUEST_TIMEOUT_SEC = 10


def load_local_env_file() -> None:
    """Best-effort .env loader so script works when launched without exported env vars."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def wait_for_server(base: str, attempts: int = 10, delay: float = 1.0) -> bool:
    for _ in range(attempts):
        try:
            r = requests.get(base + "/", timeout=REQUEST_TIMEOUT_SEC)
            print("Health:", r.status_code)
            if r.ok:
                return True
        except Exception as e:
            print("Waiting for server...", e)
        time.sleep(delay)
    return False


def build_ed25519_keypair_b64() -> tuple[str, str, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.generate()
    private_raw = private_key.private_bytes_raw()
    public_raw = private_key.public_key().public_bytes_raw()
    private_b64 = base64.b64encode(private_raw).decode("ascii")
    public_b64 = base64.b64encode(public_raw).decode("ascii")
    return private_b64, public_b64, private_key


def sign_bytes_b64(private_key: Ed25519PrivateKey, payload: bytes) -> str:
    signature = private_key.sign(payload)
    return base64.b64encode(signature).decode("ascii")


def get_admin_token(base: str, username: str, password: str) -> str:
    form = {"username": username, "password": password}
    r = requests.post(base + "/api/auth/token", data=form, timeout=REQUEST_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to get admin token: {r.status_code} {r.text}")
    body = r.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError("Auth response missing access_token")
    return token


def private_key_from_b64(private_key_b64: str) -> Ed25519PrivateKey:
    raw = base64.b64decode(private_key_b64, validate=True)
    if len(raw) != 32:
        raise ValueError("DEVICE_PRIVATE_KEY_B64 must decode to 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(raw)


def public_key_b64_from_private(private_key: Ed25519PrivateKey) -> str:
    return base64.b64encode(private_key.public_key().public_bytes_raw()).decode("ascii")


def enroll_or_rotate_device(
    base: str, admin_token: str, device_id: str, public_key_b64: str
) -> None:
    headers = {
        "Authorization": f"Bearer {admin_token}",
        "Content-Type": "application/json",
    }
    payload = {"device_id": device_id, "public_key_b64": public_key_b64}

    r = requests.post(
        base + "/api/devices/enroll",
        headers=headers,
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if r.status_code == 201:
        print("Device enrolled:", device_id)
        return
    if r.status_code != 409:
        raise RuntimeError(f"Failed to enroll device: {r.status_code} {r.text}")

    rotate_payload = {"public_key_b64": public_key_b64}
    rr = requests.post(
        base + f"/api/devices/{device_id}/rotate",
        headers=headers,
        data=json.dumps(rotate_payload),
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if rr.status_code != 200:
        raise RuntimeError(
            f"Failed to rotate existing device key: {rr.status_code} {rr.text}"
        )
    print("Device key rotated:", device_id)


def post_signed_heartbeat(
    base: str,
    site_name: str,
    edge_pc_id: str,
    private_key: Ed25519PrivateKey,
) -> None:
    hb = {
        "edge_pc_id": edge_pc_id,
        "site_name": site_name,
        "status": "online",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    body_bytes = json.dumps(hb, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    signature_b64 = sign_bytes_b64(private_key, body_bytes)
    headers = {
        "Content-Type": "application/json",
        "X-Device-Id": edge_pc_id,
        "X-Device-Signature": signature_b64,
    }

    print("Posting signed heartbeat...", hb)
    r = requests.post(
        base + "/api/heartbeat",
        data=body_bytes,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    print(r.status_code, r.text)
    if r.status_code != 201:
        raise RuntimeError("Heartbeat failed; cannot continue")


def upload_signed_alert(
    base: str,
    image_path: str,
    site_id: str,
    camera_id: str,
    edge_pc_id: str,
    detections: list,
    private_key: Ed25519PrivateKey,
) -> None:
    if not image_path:
        raise ValueError("image_path is required; pass --image /path/to/file.jpg")

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    sha = hashlib.sha256(image_bytes).hexdigest()
    canonical = f"{site_id}|{camera_id}|{edge_pc_id}|{ts}|{sha}".encode("utf-8")
    signature_b64 = sign_bytes_b64(private_key, canonical)

    headers = {
        "X-Device-Id": edge_pc_id,
        "X-Device-Signature": signature_b64,
    }
    files = {
        "image": (os.path.basename(image_path), image_bytes, "application/octet-stream")
    }
    data = {
        "site_id": site_id,
        "camera_id": camera_id,
        "edge_pc_id": edge_pc_id,
        "timestamp": ts,
        "detections": json.dumps(detections),
    }

    print("Uploading signed alert image...")
    r = requests.post(
        base + "/api/alerts/upload",
        files=files,
        data=data,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    print("Upload response:", r.status_code, r.text)
    if r.status_code != 201:
        raise RuntimeError("Alert upload failed")


def get_site_settings(base: str, site_name: str) -> None:
    print("GET site settings")
    r = requests.get(
        base + f"/api/sites/{site_name}/settings", timeout=REQUEST_TIMEOUT_SEC
    )
    print(r.status_code, r.text)


def put_site_settings(base: str, site_name: str, image_retention_hours: int) -> None:
    print(f"PUT site settings => {image_retention_hours} hours")
    r = requests.put(
        base + f"/api/sites/{site_name}/settings",
        json={"image_retention_hours": image_retention_hours},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    print(r.status_code, r.text)


def get_server_info(base: str) -> None:
    print("GET server-info")
    r = requests.get(base + "/api/server-info", timeout=REQUEST_TIMEOUT_SEC)
    print(r.status_code, r.text)


def parse_detections(raw: str | None) -> list:
    if not raw:
        return [{"class": "person", "confidence": 0.95}]
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise ValueError("Invalid --detections JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("--detections must decode to a JSON list")
    return parsed


def main() -> None:
    load_local_env_file()
    parser = argparse.ArgumentParser(description="Signed test alert uploader")
    parser.add_argument(
        "--image", "-i", required=False, help="Path to image file to upload"
    )
    parser.add_argument(
        "--site",
        default=(os.getenv("SITE_ID") or "test_site"),
        help="Site id/name",
    )
    parser.add_argument(
        "--camera", default="cam_01", help="Camera id (default: cam_01)"
    )
    parser.add_argument(
        "--edge",
        default=(os.getenv("EDGE_PC_ID") or "test-edge-1"),
        help="Edge PC/device id",
    )
    parser.add_argument(
        "--base", default=BASE, help=f"Server base URL (default: {BASE})"
    )
    parser.add_argument(
        "--set-retention",
        type=int,
        help="If provided, PUT site settings to this retention (hours)",
    )
    parser.add_argument(
        "--detections",
        help=(
            'Detections JSON string, e.g. "[{\\"class\\":\\"person\\",\\"confidence\\":0.95}]"'
        ),
    )
    parser.add_argument(
        "--admin-user",
        default=os.getenv("ADMIN_USER", "admin"),
        help="Admin username for /api/auth/token",
    )
    parser.add_argument(
        "--admin-password",
        default=(
            os.getenv("ADMIN_PASSWORD")
            or os.getenv("SECRET_KEY")
            or "your-secret-key-here"
        ),
        help="Admin password for /api/auth/token",
    )
    parser.add_argument(
        "--device-private-key-b64",
        default=os.getenv("DEVICE_PRIVATE_KEY_B64"),
        help="Existing device private key (base64). If provided with --no-provision, admin auth is skipped.",
    )
    parser.add_argument(
        "--no-provision",
        action="store_true",
        help="Skip /api/auth + /api/devices provisioning and use an existing enrolled device key.",
    )

    args = parser.parse_args()

    image_to_use = DEFAULT_IMAGE_PATH if DEFAULT_IMAGE_PATH else args.image
    if not image_to_use:
        raise SystemExit("No image provided. Use --image /path/to/file.jpg")
    if not os.path.isfile(image_to_use):
        raise SystemExit(f"Image not found: {image_to_use}")

    if not wait_for_server(args.base):
        raise SystemExit("Server did not become healthy in time")

    detections = parse_detections(args.detections)
    provision = not args.no_provision

    if args.device_private_key_b64:
        private_key = private_key_from_b64(args.device_private_key_b64)
        public_key_b64 = public_key_b64_from_private(private_key)
    else:
        _, public_key_b64, private_key = build_ed25519_keypair_b64()
        provision = True

    if provision:
        try:
            token = get_admin_token(args.base, args.admin_user, args.admin_password)
        except RuntimeError as exc:
            raise SystemExit(
                "Admin authentication failed. Pass a valid --admin-password, or run with "
                "--no-provision and --device-private-key-b64 for an already-enrolled device.\n"
                f"Details: {exc}"
            ) from exc
        enroll_or_rotate_device(args.base, token, args.edge, public_key_b64)

    post_signed_heartbeat(args.base, args.site, args.edge, private_key)
    upload_signed_alert(
        args.base,
        image_to_use,
        args.site,
        args.camera,
        args.edge,
        detections,
        private_key,
    )

    get_site_settings(args.base, args.site)
    if args.set_retention:
        put_site_settings(args.base, args.site, args.set_retention)
    get_server_info(args.base)

    print("Test script completed")


if __name__ == "__main__":
    main()
