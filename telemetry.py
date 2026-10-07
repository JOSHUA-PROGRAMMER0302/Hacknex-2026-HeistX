"""
Sentinel-Edge Telemetry Client
──────────────────────────────
Async WebSocket sender with automatic reconnection.
Runs in a background thread so the main CV loop never blocks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from typing import Any, Dict

import websockets
import websockets.exceptions

from config import TrackerConfig

logger = logging.getLogger("sentinel.telemetry")


class TelemetryClient:
    """
    Thread-safe, fire-and-forget telemetry sender.

    Usage:
        client = TelemetryClient(cfg)
        client.start()           # spawns background thread
        client.send(payload)     # non-blocking enqueue
        client.stop()            # graceful shutdown
    """

    def __init__(self, cfg: TrackerConfig):
        self._uri = cfg.websocket_uri
        self._reconnect_delay = cfg.ws_reconnect_delay_sec
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=256)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Public API ───────────────────────────────────────────────────

    def start(self):
        """Launch the background sender loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry")
        self._thread.start()
        logger.info("Telemetry thread started → %s", self._uri)

    def stop(self):
        """Signal the background thread to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
            logger.info("Telemetry thread stopped.")

    def send(self, payload: Dict[str, Any]):
        """
        Enqueue a JSON-serialisable payload.  Non-blocking; drops oldest
        message if the internal queue is full (back-pressure).
        """
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Shed oldest to keep the queue moving
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)

    # ── Internal Async Loop ──────────────────────────────────────────

    def _run(self):
        """Entry point for the background thread."""
        asyncio.run(self._async_loop())

    async def _async_loop(self):
        """
        Persistent connection loop with reconnection back-off.
        Drains the send queue and ships payloads as fast as the WS allows.
        """
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    self._uri,
                    open_timeout=5,
                    close_timeout=3,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.info("WebSocket connected to %s", self._uri)
                    while not self._stop_event.is_set():
                        try:
                            payload = self._queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        try:
                            await ws.send(json.dumps(payload))
                        except websockets.exceptions.ConnectionClosed:
                            # Re-enqueue and reconnect
                            self.send(payload)
                            break

            except (OSError, websockets.exceptions.WebSocketException) as exc:
                logger.warning(
                    "WebSocket connection failed (%s). Retrying in %.1fs…",
                    exc, self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
            except Exception:
                logger.exception("Unexpected telemetry error")
                await asyncio.sleep(self._reconnect_delay)
