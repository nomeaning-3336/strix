"""Bounded, non-blocking telemetry delivery on a daemon worker thread.

Telemetry is a beacon, never something a scan waits on. HTTP delivery runs on a
single daemon thread fed by a bounded queue: enqueue is ``put_nowait`` (drops the
event when the queue is full rather than back-pressuring the caller), so an
unreachable telemetry endpoint can never stall the asyncio event loop that all
agents share.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class TelemetryDispatcher:
    """Fire-and-forget delivery: submit() never blocks the caller."""

    def __init__(self, maxsize: int = 1024, *, name: str = "strix-telemetry") -> None:
        self._queue: queue.Queue[tuple[Callable[[], None], str]] = queue.Queue(
            maxsize=maxsize
        )
        self._name = name
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Idle = every accepted submission has finished delivering. Counting
        # accepted/completed (rather than inspecting queue emptiness + busy)
        # closes the race where a worker has dequeued an item but not yet marked
        # itself busy, which would otherwise let wait_until_idle return early.
        self._accepted = 0
        self._completed = 0

    def submit(self, fn: Callable[[], None], *, label: str) -> bool:
        """Enqueue one delivery. Never blocks; drops the event on overflow."""
        if fn is None:
            return False
        try:
            self._queue.put_nowait((fn, label))
        except queue.Full:
            logger.debug("telemetry queue full; dropping event %s", label)
            return False
        with self._lock:
            self._accepted += 1
        self._ensure_worker()
        return True

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread = thread
            thread.start()

    def _run(self) -> None:
        while True:
            fn, label = self._queue.get()
            try:
                fn()
            except Exception:  # noqa: BLE001 - telemetry must never propagate
                logger.debug("telemetry delivery failed for %s", label, exc_info=True)
            finally:
                with self._lock:
                    self._completed += 1

    def wait_until_idle(self, timeout: float = 5.0) -> bool:
        """Block (caller's choice) until accepted deliveries have drained."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._completed >= self._accepted:
                    return True
            time.sleep(0.01)
        return False


dispatcher = TelemetryDispatcher()
