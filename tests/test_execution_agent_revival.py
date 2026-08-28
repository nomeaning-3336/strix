"""Bounded revival of agents that died, driven by the scan recovering around them.

A failed agent is gated behind a user message and the user can only realistically
reach the root, so without this a child that dies stays dead for the rest of the
scan. Drawn from a run where a provider outage failed eighteen agents in thirteen
seconds and only the root ever came back.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from agents import RunConfig, Runner
from agents.exceptions import AgentsException
from openai import APIError, BadRequestError

from strix.core import execution
from strix.core.agents import AgentCoordinator


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class _FakeStream:
    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc
        self._events: list[Any] = []
        self.run_loop_exception: BaseException | None = None

    async def stream_events(self) -> Any:
        if self._exc is not None:
            raise self._exc
        for event in self._events:
            yield event


async def _run_cycle(
    monkeypatch: pytest.MonkeyPatch,
    coordinator: AgentCoordinator,
    agent_id: str,
    stream: _FakeStream,
) -> Any:
    monkeypatch.setattr(execution, "_TRANSIENT_MODEL_RETRY_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(execution, "_TRANSIENT_MODEL_RETRY_MAX_DELAY_S", 0.0)
    monkeypatch.setattr(Runner, "run_streamed", lambda *_a, **_k: stream)
    return await execution._run_cycle(
        object(),
        coordinator,
        agent_id,
        input_data="task",
        run_config=cast("RunConfig", object()),
        context={},
        max_turns=5,
        session=None,
        interactive=True,
        event_sink=None,
        hooks=None,
    )


async def _fleet(children: int = 3) -> AgentCoordinator:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    for index in range(children):
        await coordinator.register(f"child{index}", f"hunter {index}", parent_id="root")
    return coordinator


@pytest.mark.asyncio
async def test_a_landed_turn_revives_the_agents_that_died(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported scenario: the whole fleet is down and only the root is reachable."""
    coordinator = await _fleet()
    for agent_id in ("root", "child0", "child1", "child2"):
        await _run_cycle(
            monkeypatch, coordinator, agent_id, _FakeStream(exc=AgentsException("provider said no"))
        )
    # Every one of them is gated behind a user message it will never get.
    assert all(coordinator.runtimes[f"child{index}"].user_wake_required for index in range(3))

    # Whatever broke is fixed and the user messages the root, all they can reach.
    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())

    for index in range(3):
        child = f"child{index}"
        assert coordinator.statuses[child] == "waiting", f"{child} was left for dead"
        assert coordinator.runtimes[child].user_wake_required is False
        assert coordinator.pending_counts[child] == 1
        assert child not in coordinator.errors
        assert coordinator.runtimes[child].mailbox[0]["content"] == execution.AGENT_REVIVED_MESSAGE


@pytest.mark.parametrize(
    "exc",
    [
        AgentsException("sdk gave up"),
        APIError("provider rejected the request", _request(), body=None),
        BadRequestError("bad", response=httpx.Response(400, request=_request()), body=None),
        RuntimeError("something nobody anticipated"),
    ],
    ids=["agents", "api", "bad-request", "unknown"],
)
@pytest.mark.asyncio
async def test_revival_does_not_care_how_the_agent_died(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    """Cause of death is never inspected - only whether the scan works now."""
    coordinator = await _fleet(children=1)
    await _run_cycle(monkeypatch, coordinator, "child0", _FakeStream(exc=exc))
    assert coordinator.statuses["child0"] in {"failed", "crashed"}

    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())

    assert coordinator.statuses["child0"] == "waiting"


@pytest.mark.asyncio
async def test_an_agent_that_keeps_dying_drains_its_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken agent must not fail and revive forever."""
    coordinator = await _fleet(children=1)

    for attempt in range(execution._MAX_AGENT_REVIVALS):
        await coordinator.set_status("child0", "failed", error="down again")
        await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())
        assert coordinator.statuses["child0"] == "waiting", f"revival {attempt} was refused"

    await coordinator.set_status("child0", "failed", error="down again")
    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())

    assert coordinator.statuses["child0"] == "failed"
    assert coordinator.revival_counts["child0"] == execution._MAX_AGENT_REVIVALS


@pytest.mark.asyncio
async def test_a_turn_of_its_own_refunds_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = await _fleet(children=1)
    await coordinator.set_status("child0", "failed", error="down")
    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())
    assert coordinator.revival_counts["child0"] == 1

    # The revived agent picks its task back up and completes a turn.
    await _run_cycle(monkeypatch, coordinator, "child0", _FakeStream())

    assert "child0" not in coordinator.revival_counts


@pytest.mark.asyncio
async def test_an_agent_that_dies_after_the_recovery_is_still_revived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering must not strand anyone: revival is driven by live status, not a flag.

    A marker set at failure time can be cleared by a recovery that raced ahead of
    the agent recording its own status, leaving it permanently unrevivable.
    """
    coordinator = await _fleet(children=1)

    # A turn lands while nothing is down at all.
    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())
    # Only afterwards does the child die.
    await _run_cycle(monkeypatch, coordinator, "child0", _FakeStream(exc=AgentsException("late")))
    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())

    assert coordinator.statuses["child0"] == "waiting"


@pytest.mark.asyncio
async def test_healthy_agents_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = await _fleet(children=2)
    await coordinator.set_status("child0", "waiting")
    await coordinator.set_status("child1", "completed")

    await _run_cycle(monkeypatch, coordinator, "root", _FakeStream())

    assert coordinator.statuses["child0"] == "waiting"
    assert coordinator.statuses["child1"] == "completed"
    assert coordinator.pending_counts["child0"] == 0
    assert coordinator.revival_counts == {}


@pytest.mark.asyncio
async def test_revival_is_a_no_op_with_nothing_down() -> None:
    coordinator = await _fleet(children=1)

    assert coordinator.has_failed_agents is False
    assert await coordinator.revive_failed_agents("recovered", limit=3) == []


@pytest.mark.asyncio
async def test_the_budget_survives_a_resume() -> None:
    """A snapshot that drops the budget hands a broken agent unlimited revivals."""
    coordinator = await _fleet(children=1)
    await coordinator.set_status("child0", "failed", error="down")
    await coordinator.revive_failed_agents("recovered", limit=3)
    snapshot = await coordinator.snapshot()

    restored = AgentCoordinator()
    await restored.restore(snapshot)

    assert restored.revival_counts["child0"] == 1
