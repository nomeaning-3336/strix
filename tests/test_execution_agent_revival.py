"""A user's message revives every agent that died, not just the one addressed.

A failed agent parks until a human tells it to go again, and the user can only
realistically reach the root. Drawn from a run where a provider outage failed
eighteen agents in thirteen seconds and only the root ever came back.
"""

from __future__ import annotations

import pytest

from strix.core.agents import AgentCoordinator


async def _fleet(children: int = 3) -> AgentCoordinator:
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    for index in range(children):
        await coordinator.register(f"child{index}", f"hunter {index}", parent_id="root")
    return coordinator


def _user_message(text: str = "credits are back, keep going") -> dict[str, str]:
    return {"from": "user", "content": text, "type": "instruction"}


@pytest.mark.asyncio
async def test_a_user_message_revives_the_whole_fleet() -> None:
    """The reported scenario: everything is down and only the root is reachable."""
    coordinator = await _fleet()
    for agent_id in ("root", "child0", "child1", "child2"):
        await coordinator.set_status(agent_id, "failed", error="provider said no")
    assert all(coordinator.runtimes[f"child{i}"].user_wake_required for i in range(3))

    await coordinator.send("root", _user_message())

    for index in range(3):
        child = f"child{index}"
        assert coordinator.statuses[child] == "waiting", f"{child} was left for dead"
        assert coordinator.runtimes[child].user_wake_required is False
        assert coordinator.pending_counts[child] == 1
        assert child not in coordinator.errors
    # The addressed agent is woken by the user's own message, not a notice.
    assert coordinator.runtimes["root"].user_wake_required is False
    assert coordinator.runtimes["root"].mailbox == [_user_message()]


@pytest.mark.asyncio
async def test_the_revived_agent_wakes_on_the_notice() -> None:
    """An empty mailbox parks the agent straight back where it was."""
    coordinator = await _fleet(children=1)
    await coordinator.set_status("child0", "failed", error="down")

    await coordinator.send("root", _user_message())

    assert await coordinator.wait_for_message("child0", timeout=0.1) is True
    count, _items = await coordinator.consume_pending("child0")
    assert count == 1


@pytest.mark.asyncio
async def test_revival_is_not_rationed() -> None:
    """A human deciding to try again is the only limit there needs to be."""
    coordinator = await _fleet(children=1)

    for attempt in range(6):
        await coordinator.set_status("child0", "failed", error="down again")
        await coordinator.send("root", _user_message())
        assert coordinator.statuses["child0"] == "waiting", f"attempt {attempt} was refused"


@pytest.mark.parametrize("status", ["failed", "crashed"])
@pytest.mark.asyncio
async def test_revival_does_not_care_how_the_agent_died(status: str) -> None:
    coordinator = await _fleet(children=1)
    await coordinator.set_status("child0", status, error="whatever it was")

    await coordinator.send("root", _user_message())

    assert coordinator.statuses["child0"] == "waiting"


@pytest.mark.parametrize("status", ["running", "waiting", "completed", "stopped", "budget_paused"])
@pytest.mark.asyncio
async def test_agents_that_did_not_die_are_left_alone(status: str) -> None:
    """A deliberate stop - budget, user request - is not a failure to recover from."""
    coordinator = await _fleet(children=1)
    await coordinator.set_status("child0", status)

    await coordinator.send("root", _user_message())

    assert coordinator.statuses["child0"] == status
    assert coordinator.pending_counts["child0"] == 0
    assert coordinator.runtimes["child0"].mailbox == []


@pytest.mark.asyncio
async def test_messaging_a_child_revives_its_failed_siblings() -> None:
    """A person is present either way; which agent they addressed is incidental."""
    coordinator = await _fleet(children=2)
    await coordinator.set_status("child0", "failed", error="down")
    await coordinator.set_status("child1", "failed", error="down")

    await coordinator.send("child0", _user_message("look at this instead"))

    assert coordinator.statuses["child1"] == "waiting"
    assert coordinator.runtimes["child0"].user_wake_required is False


@pytest.mark.asyncio
async def test_an_agent_message_revives_nobody() -> None:
    """Only a human clears the gate; agents talking among themselves must not."""
    coordinator = await _fleet(children=2)
    await coordinator.set_status("child1", "failed", error="down")

    await coordinator.send("child0", {"from": "root", "content": "status?"})

    assert coordinator.statuses["child1"] == "failed"
    assert coordinator.runtimes["child1"].user_wake_required is True


@pytest.mark.asyncio
async def test_a_message_to_an_unknown_agent_still_revives_the_fleet() -> None:
    """The user spoke; a stale target id should not swallow that."""
    coordinator = await _fleet(children=1)
    await coordinator.set_status("child0", "failed", error="down")

    assert await coordinator.send("gone", _user_message()) is False

    assert coordinator.statuses["child0"] == "waiting"
