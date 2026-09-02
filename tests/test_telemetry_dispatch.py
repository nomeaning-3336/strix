"""Tests for the non-blocking telemetry dispatcher (#677).

Telemetry HTTP delivery must never block the asyncio event loop that all agents
share: enqueue is put_nowait, delivery happens on a daemon worker thread, and a
full/blocked endpoint drops events instead of back-pressuring the scan.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

from strix.telemetry import posthog, scarf
from strix.telemetry.dispatch import TelemetryDispatcher, dispatcher


if TYPE_CHECKING:
    import pytest


def _settings(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(telemetry=SimpleNamespace(enabled=enabled))


def test_submit_runs_delivery_on_worker() -> None:
    d = TelemetryDispatcher(maxsize=16)
    done = threading.Event()
    ran: list[str] = []

    def deliver() -> None:
        ran.append("ok")
        done.set()

    assert d.submit(deliver, label="test") is True
    assert done.wait(2.0) is True
    assert ran == ["ok"]


def test_submit_never_blocks_while_worker_is_busy() -> None:
    d = TelemetryDispatcher(maxsize=16)
    gate = threading.Event()
    started = threading.Event()

    def blocker() -> None:
        started.set()
        gate.wait(5)

    d.submit(blocker, label="blocker")
    assert started.wait(2.0) is True

    t0 = time.monotonic()
    ok = d.submit(lambda: None, label="queued")
    elapsed = time.monotonic() - t0

    gate.set()
    assert ok is True
    assert elapsed < 0.2  # enqueue did not wait for the busy worker
    assert d.wait_until_idle(2.0) is True


def test_overflow_drops_instead_of_blocking() -> None:
    d = TelemetryDispatcher(maxsize=2)
    gate = threading.Event()
    started = threading.Event()

    def blocker() -> None:
        started.set()
        gate.wait(5)

    d.submit(blocker, label="blocker")
    assert started.wait(2.0) is True

    # Fill the bounded queue...
    assert d.submit(lambda: None, label="a") is True
    assert d.submit(lambda: None, label="b") is True
    # ...and the next enqueue is dropped, not blocked on.
    t0 = time.monotonic()
    assert d.submit(lambda: None, label="overflow") is False
    assert time.monotonic() - t0 < 0.2

    gate.set()
    assert d.wait_until_idle(2.0) is True


def test_posthog_disabled_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def record(payload: object) -> None:
        calls.append(payload)

    monkeypatch.setattr(posthog, "load_settings", lambda: _settings(enabled=False))
    monkeypatch.setattr(posthog, "_http_post", record)

    posthog.finding("high", cwe="CWE-79")

    assert calls == []


def test_posthog_finding_enqueues_and_delivers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def record(payload: dict[str, object]) -> None:
        calls.append(payload)

    monkeypatch.setattr(posthog, "load_settings", lambda: _settings(enabled=True))
    monkeypatch.setattr(posthog, "_http_post", record)

    posthog.finding("high", cwe="CWE-79")

    assert dispatcher.wait_until_idle(2.0) is True
    assert len(calls) == 1
    assert calls[0]["event"] == "finding_reported"
    assert calls[0]["properties"]["severity"] == "high"
    assert calls[0]["properties"]["cwe"] == "cwe-79"


def test_scarf_finding_delivers_on_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    def record(event: str, properties: object) -> None:
        calls.append((event, properties))

    monkeypatch.setattr(scarf, "load_settings", lambda: _settings(enabled=True))
    monkeypatch.setattr(scarf, "_http_post", record)

    scarf.finding("medium")

    assert dispatcher.wait_until_idle(2.0) is True
    assert len(calls) == 1
    assert calls[0][0] == "finding_reported"
