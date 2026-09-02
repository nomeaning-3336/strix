"""Tests for pending-message restoration on session-write failure (#1161).

If ``session.add_items`` throws after the mailbox was drained, the drained
messages must be restored to the FRONT of the mailbox (order preserved against
concurrent arrivals), the pending count rebuilt, and the agent woken — the
caller gets ``(0, [])`` so nothing is treated as delivered that was not
persisted. This protects inter-agent delivery and the hot-reload machinery.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from strix.core.agents import AgentCoordinator, AgentRuntime


class FailOnceSession:
    """add_items fails the first `fail_for` calls, then succeeds."""

    def __init__(self, fail_for: int = 1) -> None:
        self.fail_for = fail_for
        self.items: list[Any] = []
        self.writes = 0

    async def add_items(self, items: list[Any]) -> None:
        self.writes += 1
        if self.writes <= self.fail_for:
            raise RuntimeError("sqlite write failed")
        self.items.extend(items)


class GatedSession:
    """add_items parks until released, then raises (or records)."""

    def __init__(self, *, raise_on_release: bool = True) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.raise_on_release = raise_on_release
        self.items: list[Any] = []

    async def add_items(self, items: list[Any]) -> None:
        self.started.set()
        await self.release.wait()
        if self.raise_on_release:
            raise RuntimeError("sqlite write failed")
        self.items.extend(items)


def _msg(content: str = "hello", from_="user", mtype="instruction") -> dict[str, Any]:
    return {"from": from_, "content": content, "type": mtype}


def _coord(session: Any | None, msgs: list[dict[str, Any]], agent: str = "a1"):
    coord = AgentCoordinator()
    coord.names[agent] = "Agent A"
    coord.statuses[agent] = "waiting"
    runtime = AgentRuntime()
    runtime.session = session
    runtime.mailbox = list(msgs)
    coord.runtimes[agent] = runtime
    if msgs:
        coord.pending_counts[agent] = len(msgs)
    return coord, runtime


@pytest.mark.asyncio
async def test_failed_write_restores_message() -> None:
    coord, runtime = _coord(FailOnceSession(fail_for=1), [_msg()])

    result = await coord.consume_pending("a1", include_items=True)

    assert result == (0, [])
    assert runtime.mailbox == [_msg()]
    assert coord.pending_counts["a1"] == 1


@pytest.mark.asyncio
async def test_next_consume_retries_successfully() -> None:
    session = FailOnceSession(fail_for=1)
    coord, runtime = _coord(session, [_msg("orig")])

    assert await coord.consume_pending("a1") == (0, [])
    assert runtime.mailbox == [_msg("orig")]

    result = await coord.consume_pending("a1", include_items=True)
    assert result[0] == 1
    # Delivered items are the SDK-session form, not the raw message dicts.
    assert result[1] == [{"role": "user", "content": "orig"}]
    assert runtime.mailbox == []
    assert coord.pending_counts["a1"] == 0
    assert len(session.items) == 1


@pytest.mark.asyncio
async def test_concurrent_arrival_keeps_order_original_then_new() -> None:
    session = GatedSession(raise_on_release=True)
    coord, runtime = _coord(session, [_msg("original")])
    new_msg = _msg("new-arrival")

    consumer = asyncio.create_task(coord.consume_pending("a1"))
    await session.started.wait()  # original drained, write in flight
    await coord.send("a1", new_msg)  # concurrent arrival
    session.release.set()  # ...then the write fails
    assert await consumer == (0, [])

    assert [m["content"] for m in runtime.mailbox] == ["original", "new-arrival"]
    assert coord.pending_counts["a1"] == 2


@pytest.mark.asyncio
async def test_pending_count_reconstructed_after_restore() -> None:
    coord, runtime = _coord(FailOnceSession(fail_for=1), [_msg("a"), _msg("b")])
    coord.pending_counts["a1"] = 3  # simulate an extra concurrent notification

    await coord.consume_pending("a1")

    assert coord.pending_counts["a1"] == 2  # rebuilt from the restored mailbox
    assert len(runtime.mailbox) == 2


@pytest.mark.asyncio
async def test_wake_event_set_after_restore() -> None:
    coord, runtime = _coord(FailOnceSession(fail_for=1), [_msg()])
    assert runtime.wake.is_set() is False

    await coord.consume_pending("a1")

    assert runtime.wake.is_set() is True


@pytest.mark.asyncio
async def test_include_items_false_does_not_lose_message() -> None:
    coord, runtime = _coord(FailOnceSession(fail_for=1), [_msg()])

    assert await coord.consume_pending("a1", include_items=False) == (0, [])
    assert runtime.mailbox == [_msg()]
    assert coord.pending_counts["a1"] == 1


@pytest.mark.asyncio
async def test_normal_successful_consume_unchanged() -> None:
    session = FailOnceSession(fail_for=0)
    coord, runtime = _coord(session, [_msg("ok")])

    assert await coord.consume_pending("a1") == (1, [])
    assert runtime.mailbox == []
    assert coord.pending_counts["a1"] == 0
    assert len(session.items) == 1


@pytest.mark.asyncio
async def test_repeated_failures_do_not_duplicate_messages() -> None:
    coord, runtime = _coord(FailOnceSession(fail_for=99), [_msg("once")])

    for _ in range(3):
        assert await coord.consume_pending("a1") == (0, [])
        assert len(runtime.mailbox) == 1
        assert coord.pending_counts["a1"] == 1


@pytest.mark.asyncio
async def test_no_session_restores_instead_of_losing() -> None:
    # A caller with include_items=False must never receive a positive delivery
    # count when nothing was persisted: with no session the messages are
    # restored, exactly like a failed write.
    coord, runtime = _coord(None, [_msg("keep-me")])

    assert await coord.consume_pending("a1", include_items=False) == (0, [])
    assert runtime.mailbox == [_msg("keep-me")]
    assert coord.pending_counts["a1"] == 1


@pytest.mark.asyncio
async def test_hot_reload_control_message_survives_write_failure() -> None:
    # A reload/model-control instruction (e.g. an agent-models change note or a
    # steering message) must survive the same persistence failure unchanged.
    control = _msg("agent-models changed; adopt on next boundary", mtype="instruction")
    session = FailOnceSession(fail_for=1)
    coord, runtime = _coord(session, [control])

    assert await coord.consume_pending("a1") == (0, [])
    assert runtime.mailbox[0] == control

    await coord.consume_pending("a1", include_items=True)
    assert session.items[0] == {"role": "user", "content": control["content"]}
    assert control["type"] == "instruction"  # raw message untouched by the round trip
