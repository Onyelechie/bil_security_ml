from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

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

    def send_alert(self, payload: Dict[str, Any]) -> bool:
        """
        Send an alert to the central server.
        """
        url = f"{self.settings.server_base_url}/api/alerts"
        try:
            resp = self._session.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info("Sent alert to server: %s", payload)
            return True
        except requests.RequestException as e:
            logger.error("Failed to send alert: %s", e)
            return False

    def send_heartbeat(self, started_monotonic: Optional[float] = None) -> bool:
        """
        Send a heartbeat to the central server to indicate that the edge agent is alive.
        If `started_monotonic` is provided, include uptime_seconds in the payload.
        """
        url = f"{self.settings.server_base_url}/api/heartbeat"
        payload: Dict[str, Any] = {
            "edge_pc_id": self.settings.edge_pc_id,
            "site_name": self.settings.site_name,
            "site_id": self.settings.site_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "online",
        }
        if started_monotonic is not None:
            payload["uptime_seconds"] = int(time.monotonic() - started_monotonic)

        try:
            resp = self._session.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info("Sent heartbeat to server.")
            return True
        except requests.RequestException as e:
            logger.error("Failed to send heartbeat: %s", e)
            return False