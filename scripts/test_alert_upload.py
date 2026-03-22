"""CLI test uploader for alerts.

Usage examples:
    python scripts/test_alert_upload.py --image "/path/to/img.jpg"

The script will:
 - wait for the server to become healthy
 - post a heartbeat to register the site/edge
 - upload the provided image to /api/alerts/upload
 - optionally set site retention when --set-retention is provided
"""

import argparse
import json
import os
import time
import datetime

import requests


BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# If you want to hardcode a path here, set it to an absolute path string.
# If left as an empty string, the script will use the `--image` argument.
DEFAULT_IMAGE_PATH = r"C:\Users\ebere\Documents\bil_security_ml\storage\alert_images\test_site\WhatsApp Image 2025-05-13 at 00.20.37.jpeg"



def wait_for_server(base: str, attempts: int = 6, delay: float = 1.0):
    for i in range(attempts):
        try:
            r = requests.get(base + "/")
            print("Health:", r.status_code)
            return True
        except Exception as e:
            print("Waiting for server...", e)
            time.sleep(delay)
    return False


def post_heartbeat(base: str, site_name: str, edge_pc_id: str):
    hb = {
        "edge_pc_id": edge_pc_id,
        "site_name": site_name,
        "status": "online",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    print("Posting heartbeat...", hb)
    r = requests.post(base + "/api/heartbeat", json=hb)
    print(r.status_code, r.text)


def upload_alert(
    base: str,
    image_path: str | None,
    site_id: str,
    camera_id: str,
    edge_pc_id: str,
    detections: list,
):
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Prepare files object for requests
    if not image_path:
        raise ValueError("image_path is required; pass --image /path/to/file.jpg")
    f = None
    fname = os.path.basename(image_path)
    f = open(image_path, "rb")
    files = {"image": (fname, f, "application/octet-stream")}

    data = {
        "site_id": site_id,
        "camera_id": camera_id,
        "edge_pc_id": edge_pc_id,
        "timestamp": ts,
        "detections": json.dumps(detections),
    }

    print("Uploading alert image...")
    try:
        r = requests.post(base + "/api/alerts/upload", files=files, data=data)
        print("Upload response:", r.status_code, r.text)
    finally:
        # close file if we opened one
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


def get_site_settings(base: str, site_name: str):
    print("GET site settings")
    r = requests.get(base + f"/api/sites/{site_name}/settings")
    print(r.status_code, r.text)


def put_site_settings(base: str, site_name: str, image_retention_hours: int):
    print(f"PUT site settings => {image_retention_hours} hours")
    r = requests.put(base + f"/api/sites/{site_name}/settings", json={"image_retention_hours": image_retention_hours})
    print(r.status_code, r.text)


def get_server_info(base: str):
    print("GET server-info")
    r = requests.get(base + "/api/server-info")
    print(r.status_code, r.text)


def main():
    parser = argparse.ArgumentParser(description="Test alert uploader")
    parser.add_argument("--image", "-i", required=False, help="Path to image file to upload (required unless DEFAULT_IMAGE_PATH is set)")
    parser.add_argument("--site", default="test_site", help="Site id/name (default: test_site)")
    parser.add_argument("--camera", default="cam_01", help="Camera id (default: cam_01)")
    parser.add_argument("--edge", default="test-edge-1", help="Edge PC id (default: test-edge-1)")
    parser.add_argument("--base", default=BASE, help=f"Server base URL (default: {BASE})")
    parser.add_argument("--set-retention", type=int, help="If provided, PUT site settings to this retention (hours)")
    parser.add_argument(
        "--detections",
        help='Detections JSON string, e.g. "[{\"class\":\"person\",\"confidence\":0.95}]"',
    )

    args = parser.parse_args()

    base = args.base

    if not wait_for_server(base):
        print("Server did not start, aborting test")
        raise SystemExit(1)

    post_heartbeat(base, args.site, args.edge)

    # choose image: prefer DEFAULT_IMAGE_PATH when set, otherwise use CLI arg
    image_to_use = DEFAULT_IMAGE_PATH if DEFAULT_IMAGE_PATH else args.image

    if not image_to_use:
        print("Error: no image provided. Either set DEFAULT_IMAGE_PATH in the script or pass --image /path/to/file.jpg")
        raise SystemExit(2)

    if args.detections:
        try:
            dets = json.loads(args.detections)
        except Exception:
            print("Invalid detections JSON provided on command line; falling back to default")
            dets = [{"class": "person", "confidence": 0.95}]
    else:
        dets = [{"class": "person", "confidence": 0.95}]

    upload_alert(base, image_to_use, args.site, args.camera, args.edge, dets)

    get_site_settings(base, args.site)

    if args.set_retention:
        put_site_settings(base, args.site, args.set_retention)

    get_server_info(base)

    print("Test script completed")


if __name__ == "__main__":
    main()
