"""Tests for parent/child coordination once a non-interactive child has finished.

A non-interactive agent's loop returns after its terminal state, so nothing will
ever read a message sent to it afterwards. Messaging it must say so instead of
reporting delivery, waiting on it must return at once, and its completion report
must carry the identities of the reports it actually filed - with their current
lifecycle state, so a parent cannot treat a later-retracted finding as active.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from agents.tool_context import ToolContext

from strix.core.agents import AgentCoordinator
from strix.core.sessions import open_agent_session
from strix.report.state import ReportState, set_global_report_state
from strix.tools.agents_graph.tools import agent_finish, send_message_to_agent, wait_for_agents


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def report_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ReportState]:
    monkeypatch.chdir(tmp_path)
    state = ReportState(run_name="test-run")
    set_global_report_state(state)
    yield state
    set_global_report_state(None)


async def _graph(*, interactive: bool, tmp_path: Path | None = None) -> AgentCoordinator:
    """A two-agent graph shaped like the real runner's.

    An interactive run mirrors _start_child_runner, which attaches a session
    together with resumable=interactive; a non-interactive run attaches neither
    (its loops return when terminal). Pass ``tmp_path`` for an interactive graph
    so the sessions are real and delivery can be counted.
    """
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    await coordinator.register("child", "Validator", parent_id="root")
    for aid in ("root", "child"):
        session = (
            open_agent_session(aid, tmp_path / f"{aid}.db")
            if interactive and tmp_path is not None
            else None
        )
        await coordinator.attach_runtime(aid, session=session, resumable=interactive)
    return coordinator


async def _call(
    tool: Any, coordinator: AgentCoordinator, agent_id: str, args: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    ctx = ToolContext(
        context={"coordinator": coordinator, "agent_id": agent_id, **extra},
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments="{}",
    )
    raw: str = await tool.on_invoke_tool(ctx, json.dumps(args))
    return cast("dict[str, Any]", json.loads(raw))


# --- send_message_to_agent -------------------------------------------------------


@pytest.mark.asyncio
async def test_message_to_finished_non_interactive_child_is_not_delivered() -> None:
    coordinator = await _graph(interactive=False)
    await coordinator.set_status("child", "completed")

    result = await _call(
        send_message_to_agent,
        coordinator,
        "root",
        {"target_agent_id": "child", "message": "did you file it?", "message_type": "query"},
    )

    assert result["success"] is False
    assert result["delivery_status"] == "not_delivered"
    assert result["target_status"] == "completed"
    assert "list_reports" in result["error"]
    assert coordinator.pending_counts.get("child", 0) == 0
    assert coordinator.runtimes["child"].mailbox == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["stopped", "failed", "crashed"])
async def test_every_terminal_non_interactive_status_is_unreachable(status: str) -> None:
    coordinator = await _graph(interactive=False)
    await coordinator.set_status("child", status)

    assert await coordinator.send("child", {"from": "root", "content": "hi"}) is False
    assert await coordinator.reachability("child") == (False, status)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["running", "waiting"])
async def test_message_to_live_child_is_delivered(status: str) -> None:
    coordinator = await _graph(interactive=False)
    await coordinator.set_status("child", status)

    result = await _call(
        send_message_to_agent,
        coordinator,
        "root",
        {"target_agent_id": "child", "message": "one more thing"},
    )

    assert result["success"] is True
    assert coordinator.pending_counts["child"] == 1


@pytest.mark.asyncio
async def test_interactive_run_can_still_wake_a_completed_child(tmp_path: Path) -> None:
    """An interactive loop parks after a terminal state, so a message resumes it."""
    coordinator = await _graph(interactive=True, tmp_path=tmp_path)
    await coordinator.set_status("child", "completed")

    assert await coordinator.reachability("child") == (True, "completed")
    result = await _call(
        send_message_to_agent,
        coordinator,
        "root",
        {"target_agent_id": "child", "message": "one more thing"},
    )

    assert result["success"] is True
    assert coordinator.pending_counts["child"] == 1


@pytest.mark.asyncio
async def test_unknown_target_is_reported_as_not_found() -> None:
    coordinator = await _graph(interactive=False)

    result = await _call(
        send_message_to_agent,
        coordinator,
        "root",
        {"target_agent_id": "ghost", "message": "hello"},
    )

    assert result["success"] is False
    assert result["target_status"] is None
    assert "not found" in result["error"]


# --- reachability across a resume -------------------------------------------------


@pytest.mark.asyncio
async def test_completed_child_stays_unreachable_across_snapshot_restore() -> None:
    """The 1edafd3 bug must not come back after ``--resume``.

    ``resumable`` is runtime-only: snapshot() does not persist it and restore()
    recreates runtimes with the dataclass default (True). A non-interactive
    resume also skips terminal children in respawn_subagents, so a completed
    child comes back with the flag claiming "reachable" and no loop behind it.
    """
    coordinator = await _graph(interactive=False)
    await coordinator.set_status("child", "completed")
    assert await coordinator.reachability("child") == (False, "completed")

    snapshot = await coordinator.snapshot()
    resumed = AgentCoordinator()
    await resumed.restore(snapshot)

    # The flag did not survive; the missing session is what has to catch this.
    assert resumed.runtimes["child"].resumable is True
    assert resumed.runtimes["child"].session is None
    assert await resumed.reachability("child") == (False, "completed")
    assert await resumed.send("child", {"from": "root", "content": "hi"}) is False


@pytest.mark.asyncio
async def test_terminal_agent_without_a_session_is_unreachable() -> None:
    """Registered then terminal with no session: nothing can read the mailbox.

    Also covers the rarer case where child creation registers an agent but
    runner/session setup fails before a usable loop exists.
    """
    coordinator = AgentCoordinator()
    await coordinator.register("root", "strix", parent_id=None)
    await coordinator.register("child", "Validator", parent_id="root")
    await coordinator.set_status("child", "completed")

    runtime = coordinator.runtimes["child"]
    assert runtime.resumable is True  # the flag alone would claim reachable
    assert runtime.session is None
    assert await coordinator.reachability("child") == (False, "completed")


@pytest.mark.asyncio
async def test_terminal_child_with_a_session_stays_reachable(tmp_path: Path) -> None:
    """Positive control: the session check must not make reachability too broad.

    An interactive resumed child is actually respawned, so it has both the flag
    and a session, and a follow-up message must still reach it.
    """
    coordinator = await _graph(interactive=True, tmp_path=tmp_path)
    await coordinator.set_status("child", "completed")

    assert await coordinator.reachability("child") == (True, "completed")
    assert await coordinator.send("child", {"from": "root", "content": "hi"}) is True


# --- wait_for_agents -------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_returns_at_once_when_no_child_can_answer() -> None:
    coordinator = await _graph(interactive=False)
    await coordinator.set_status("child", "completed")
    # The completion report was already consumed in an earlier turn.

    result = await _call(
        wait_for_agents,
        coordinator,
        "root",
        {"reason": "waiting for validator", "timeout_seconds": 240},
    )

    assert result["wait_outcome"] == "no_active_agents"
    assert result["agents"] == [{"agent_id": "child", "name": "Validator", "status": "completed"}]
    assert coordinator.statuses["root"] == "running"


@pytest.mark.asyncio
async def test_wait_delivers_a_pending_report_before_checking_liveness(tmp_path: Path) -> None:
    # This fork only counts a delivery once the message is durably persisted to
    # the target's SDK session (see AgentCoordinator.consume_pending), so the
    # pending-report path needs a real session attached.
    coordinator = await _graph(interactive=False)
    await coordinator.attach_runtime("root", session=open_agent_session("root", tmp_path / "a.db"))
    await coordinator.send("root", {"from": "child", "type": "completion", "content": "done"})
    await coordinator.set_status("child", "completed")

    result = await _call(wait_for_agents, coordinator, "root", {"timeout_seconds": 5})

    assert result["wait_outcome"] == "message_arrived"
    assert result["pending_messages"] == 1


@pytest.mark.asyncio
async def test_wait_does_not_shortcut_while_a_child_is_running() -> None:
    """A live child keeps the parent parked; only nobody-can-answer shortcuts.

    The fork reports a timed-out wait with live children as ``supervision_tick``
    (a health snapshot) rather than a bare ``timeout``, so the property under
    test here is that the wait did NOT take the no_active_agents shortcut.
    """
    coordinator = await _graph(interactive=False)

    result = await _call(wait_for_agents, coordinator, "root", {"timeout_seconds": 1})

    assert result["wait_outcome"] in {"timeout", "supervision_tick"}
    assert result["wait_outcome"] != "no_active_agents"


@pytest.mark.asyncio
async def test_interactive_wait_parks_even_without_active_children() -> None:
    # In an interactive run a finished child can be woken later, so parking is
    # legitimate; the run loop's own auto-resume bounds the wait.
    coordinator = await _graph(interactive=True)
    await coordinator.set_status("child", "completed")

    result = await _call(
        wait_for_agents, coordinator, "root", {"timeout_seconds": 5}, interactive=True
    )

    assert result["wait_outcome"] == "waiting"


# --- agent_finish ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_finish_lists_the_reports_the_child_filed(report_state: ReportState) -> None:
    coordinator = await _graph(interactive=False)
    mine = report_state.add_vulnerability_report(
        title="IDOR on /api/audits", severity="high", agent_id="child", agent_name="Validator"
    )
    report_state.add_vulnerability_report(title="Root's own", severity="low", agent_id="root")

    result = await _call(
        agent_finish,
        coordinator,
        "child",
        {"result_summary": "confirmed", "findings": ["IDOR confirmed"]},
        parent_id="root",
    )

    assert result["filed_report_ids"] == [mine]
    delivered = coordinator.runtimes["root"].mailbox
    assert len(delivered) == 1
    assert delivered[0]["filed_report_ids"] == [mine]
    body = delivered[0]["content"]
    # The state rides along with the id so a parent can tell active from archived.
    assert f"- {mine} [verified] [HIGH] IDOR on /api/audits" in body
    assert "Root's own" not in body


@pytest.mark.asyncio
async def test_agent_finish_reports_a_retracted_finding_as_retracted(
    report_state: ReportState,
) -> None:
    """A parent must not treat a retracted child finding as active."""
    coordinator = await _graph(interactive=False)
    mine = report_state.add_vulnerability_report(
        title="Claimed auth bypass", severity="high", agent_id="child", agent_name="Validator"
    )
    report_state.set_finding_state(
        mine, "retracted", reason="the mechanic does not exist", changed_by_agent_id="child"
    )

    await _call(
        agent_finish,
        coordinator,
        "child",
        {"result_summary": "ruled out"},
        parent_id="root",
    )

    delivered = coordinator.runtimes["root"].mailbox
    body = delivered[0]["content"]
    assert f"- {mine} [retracted] [HIGH] Claimed auth bypass" in body
    assert delivered[0]["filed_reports"] == [{"id": mine, "state": "retracted"}]


@pytest.mark.asyncio
async def test_agent_finish_states_explicitly_when_nothing_was_filed(
    report_state: ReportState,
) -> None:
    coordinator = await _graph(interactive=False)
    report_state.add_vulnerability_report(title="Someone else's", severity="low", agent_id="root")

    result = await _call(
        agent_finish,
        coordinator,
        "child",
        {"result_summary": "nothing exploitable", "findings": ["ruled out X"]},
        parent_id="root",
    )

    assert result["filed_report_ids"] == []
    assert result["filed_reports"] == []
    body = coordinator.runtimes["root"].mailbox[0]["content"]
    assert "Vulnerability reports filed by this agent" in body
    assert body.index("filed by this agent") < body.index("- (none)")


@pytest.mark.asyncio
async def test_agent_finish_without_report_state_still_completes() -> None:
    set_global_report_state(None)
    coordinator = await _graph(interactive=False)

    result = await _call(
        agent_finish, coordinator, "child", {"result_summary": "done"}, parent_id="root"
    )

    assert result["success"] is True
    assert result["filed_report_ids"] == []
    assert result["filed_reports"] == []
