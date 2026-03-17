from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .config import EdgeSettings

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
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "detections": detections,
        }
        if image_path:
            payload["image_path"] = image_path

        try:
            with self._session_lock:
                resp = self._session.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info(
                "Sent alert to server: camera_id=%s detections=%d",
                camera_id,
                len(detections),
            )
            return True
        except requests.RequestException as e:
            logger.error("Failed to send alert: %s", e)
            return False

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

        try:
            with self._session_lock:
                resp = self._session.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info("Sent heartbeat to server (status: %s).", current_status)
            return True
        except requests.RequestException as e:
            logger.error("Failed to send heartbeat (status: %s): %s", current_status, e)
            return False
