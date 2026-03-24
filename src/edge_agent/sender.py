from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from bil_time import isoformat_winnipeg, now_in_winnipeg

from .config import EdgeSettings
from .signing import sign_message_b64

logger = logging.getLogger(__name__)


class ServerSender:
    """
    Responsible for sending alerts and heartbeats to the central server.
    """

    def __init__(self, settings: EdgeSettings):
        self.settings = settings
        self._session = requests.Session()
        self._status = "starting"
        self._status_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self.queue_dir = settings.offline_queue_dir
        try:
            os.makedirs(self.queue_dir, exist_ok=True)
        except Exception as e:
            logger.error(
                "Failed to create offline queue dir '%s': %s", self.queue_dir, e
            )

    def set_status(self, status: str) -> None:
        """Set the agent's status for heartbeats. This is thread-safe."""
        with self._status_lock:
            if self._status != status:
                self._status = status
                logger.info("Agent status for heartbeats set to '%s'", status)

    def get_status(self) -> str:
        """Get the agent's current status. This is thread-safe."""
        with self._status_lock:
            return self._status

    def _resolved_device_id(self) -> str:
        return (self.settings.device_id or self.settings.edge_pc_id).strip()

    def _signed_headers(self, message: bytes, *, edge_pc_id: str) -> dict[str, str] | None:
        device_id = self._resolved_device_id()
        private_key_b64 = self.settings.device_private_key_b64.strip()
        if not device_id:
            logger.error("Cannot send request without a configured device_id or edge_pc_id")
            return None
        if not private_key_b64:
            logger.error("Cannot send request without DEVICE_PRIVATE_KEY_B64 configured")
            return None
        if device_id != edge_pc_id:
            logger.error(
                "Configured device_id '%s' does not match edge_pc_id '%s'",
                device_id,
                edge_pc_id,
            )
            return None
        try:
            signature = sign_message_b64(private_key_b64, message)
        except Exception as exc:
            logger.error("Failed to sign request: %s", exc)
            return None
        return {
            "Content-Type": "application/json",
            "X-Device-Id": device_id,
            "X-Device-Signature": signature,
        }

    def send_alert(
        self,
        *,
        camera_id: str,
        detections: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None,
        image_path: Optional[str] = None,
    ) -> bool:
        """
        Build and send an alert to the central server, conforming to the AlertCreate schema.

        :param camera_id: The ID of the camera that generated the alert.
        :param detections: A list of detection dictionaries, e.g., [{"class": "person", "confidence": 0.95}].
        :param timestamp: The timestamp of the alert. If None, the current time is used.
        :param image_path: Optional path to an associated image.
        :return: True if the alert was sent successfully, False otherwise.
        """
        if not self._validate_detections(detections):
            logger.error("Invalid detections payload; skipping alert send")
            return False

        url = f"{self.settings.server_base_url}/api/alerts"
        payload: Dict[str, Any] = {
            "site_id": self.settings.site_id,
            "edge_pc_id": self.settings.edge_pc_id,
            "camera_id": camera_id,
            "timestamp": isoformat_winnipeg(timestamp or now_in_winnipeg()),
            "detections": detections,
        }
        if image_path:
            payload["image_path"] = image_path
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = self._signed_headers(body, edge_pc_id=self.settings.edge_pc_id)
        if headers is None:
            return False

        try:
            with self._session_lock:
                resp = self._session.post(url, data=body, headers=headers, timeout=5)
            resp.raise_for_status()
            logger.info(
                "Sent alert to server: camera_id=%s detections=%d",
                camera_id,
                len(detections),
            )
            return True
        except requests.RequestException as e:
            status = None
            if getattr(e, "response", None) is not None:
                status = e.response.status_code

            if status is not None and 400 <= status < 500:
                logger.error(
                    "Failed to send alert (client error %s); dropping: %s",
                    status,
                    e,
                )
                return False

            logger.error("Failed to send alert: %s", e)
            self._save_to_queue(self._queue_payload(payload))
            return False

    def _save_to_queue(self, payload: Dict[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"alert_{timestamp}.json"
        path = os.path.join(self.queue_dir, filename)
        tmp_path = f"{path}.tmp"

        try:
            with self._queue_lock:
                with open(tmp_path, "w") as f:
                    json.dump(payload, f)
                os.replace(tmp_path, path)
            logger.warning("Saved alert to offline queue: %s", path)
        except Exception as e:
            logger.error("Failed to save alert locally: %s", e)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as cleanup_err:
                logger.warning(
                    "Failed to remove temp queue file '%s': %s", tmp_path, cleanup_err
                )

    @staticmethod
    def _queue_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove fields that are not safe to replay later.
        For now, drop image_path because it points to an edge-local path.
        """
        payload = dict(payload)
        payload.pop("image_path", None)
        return payload

    def retry_queued_alerts(self) -> None:
        """Attempt to resend any alerts that were saved to the offline queue."""
        with self._queue_lock:
            files = sorted(
                f
                for f in os.listdir(self.queue_dir)
                if f.startswith("alert_") and f.endswith(".json")
            )
        if files:
            logger.info("Retrying %d queued alerts", len(files))
        for filename in files:
            path = os.path.join(self.queue_dir, filename)
            try:
                with self._queue_lock:
                    with open(path, "r") as f:
                        payload = json.load(f)

                url = f"{self.settings.server_base_url}/api/alerts"

                with self._session_lock:
                    resp = self._session.post(url, json=payload, timeout=5)

                resp.raise_for_status()

                with self._queue_lock:
                    os.remove(path)
                logger.info(
                    "Successfully resent queued alert and removed file: %s", path
                )

            except Exception as e:
                logger.error("Error processing queued alert file %s: %s", path, e)
                break  # Stop processing further files if one fails to avoid rapid retries

    @staticmethod
    def _validate_detections(detections: List[Dict[str, Any]]) -> bool:
        """Validate the detections payload to ensure it conforms to expected schema."""
        if not isinstance(detections, list) or not detections:
            return False
        for detection in detections:
            if not isinstance(detection, dict):
                return False
            if "class" not in detection or "confidence" not in detection:
                return False
            if not isinstance(detection["class"], str) or not detection["class"]:
                return False
            if not isinstance(detection["confidence"], (int, float)):
                return False
        return True

    def send_heartbeat(self, started_monotonic: Optional[float] = None) -> bool:
        """
        Send a heartbeat to the central server to indicate that the edge agent is alive.
        If `started_monotonic` is provided, include uptime_seconds in the payload.
        """
        url = f"{self.settings.server_base_url}/api/heartbeat"
        current_status = self.get_status()
        payload: Dict[str, Any] = {
            "edge_pc_id": self.settings.edge_pc_id,
            "site_name": self.settings.site_name,
            "site_id": self.settings.site_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": current_status,
        }
        if started_monotonic is not None:
            payload["uptime_seconds"] = int(time.monotonic() - started_monotonic)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = self._signed_headers(body, edge_pc_id=self.settings.edge_pc_id)
        if headers is None:
            return False

        try:
            with self._session_lock:
                resp = self._session.post(url, data=body, headers=headers, timeout=5)
            resp.raise_for_status()
            logger.info("Sent heartbeat to server (status: %s).", current_status)
            return True
        except requests.RequestException as e:
            logger.error("Failed to send heartbeat (status: %s): %s", current_status, e)
            return False
