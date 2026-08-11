"""Run-scoped safety enforcement."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from strix.config.settings import SafetySettings
from strix.safety.runtime import SafetyRuntime
from strix.safety.types import SafetyApprovalCallback, SafetyApprovalRequest, SafetyDecision


if TYPE_CHECKING:
    from pathlib import Path


_SNAPSHOT_HISTORY: list[dict[str, Any]] = [
    {
        "type": "function_call",
        "name": "exec_command",
        "call_id": "snapshot-1",
        "arguments": '{"cmd":"agent-browser snapshot -i"}',
    },
    {
        "type": "function_call_output",
        "call_id": "snapshot-1",
        "output": '@e3 [button type="submit"] "Search"',
    },
]


class _InspectionRunner:
    async def run(self, *, evidence_dir: str, script: str) -> str:
        return f"unused: {evidence_dir} {script}"


class _StubReviewer:
    """Stands in for the model review so a decision's source can be asserted."""

    def __init__(
        self,
        on_review: Any = None,
        decision: SafetyDecision | None = None,
    ) -> None:
        self.on_review = on_review
        self.decision = decision
        self.calls = 0
        self.human_approval_available: list[bool] = []

    async def review(
        self,
        bundle: Any,
        *,
        human_approval_available: bool = False,
    ) -> SafetyDecision:
        self.calls += 1
        self.human_approval_available.append(human_approval_available)
        if self.on_review is not None:
            await self.on_review()
        if self.decision is not None:
            return SafetyDecision(
                allowed=self.decision.allowed,
                source=self.decision.source,
                reason=self.decision.reason,
                categories=self.decision.categories,
                case_id=bundle.case_id,
                risk=self.decision.risk,
                deferred=self.decision.deferred,
            )
        return SafetyDecision(
            allowed=True,
            source="reviewer",
            reason="allowed",
            case_id=bundle.case_id,
        )


def _runtime(
    tmp_path: Path,
    mode: str,
    approval_callback: SafetyApprovalCallback | None = None,
) -> SafetyRuntime:
    return SafetyRuntime(
        scan_id="scan-1",
        mode=mode,  # type: ignore[arg-type]
        scope={},
        user_instruction="",
        settings=SafetySettings(),
        run_dir=tmp_path,
        sandbox_image="image",
        inspection_runner=_InspectionRunner(),
        approval_callback=approval_callback,
    )


def _deferred() -> SafetyDecision:
    return SafetyDecision(
        allowed=False,
        source="reviewer",
        reason="the endpoint effect is ambiguous",
        categories=("ambiguous_effect",),
        risk="medium",
        deferred=True,
    )


def _ctx(
    *,
    agent_id: str = "agent-1",
    turn_input: list[dict[str, Any]] | None = None,
    coordinator: Any = None,
) -> Any:
    context = {"agent_id": agent_id, "sandbox_session": object()}
    if coordinator is not None:
        context["coordinator"] = coordinator
    return SimpleNamespace(
        context=context,
        tool_call_id="call-1",
        turn_input=turn_input or [],
    )


@pytest.mark.asyncio
async def test_known_read_command_executes_without_model_review(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def invoke(_ctx: Any, raw_input: str) -> str:
        seen.append(json.loads(raw_input))
        return "ok"

    result = await _runtime(tmp_path, "guarded").invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "ls /workspace"},
        invoke_tool=invoke,
    )

    assert result == "ok"
    assert seen == [{"cmd": "ls /workspace"}]


@pytest.mark.asyncio
async def test_browser_sessions_are_disjoint_per_agent(tmp_path: Path) -> None:
    seen: list[str] = []

    async def invoke(_ctx: Any, raw_input: str) -> str:
        seen.append(json.loads(raw_input)["cmd"])
        return "snapshot"

    runtime = _runtime(tmp_path, "guarded")
    for agent_id in ("agent-1", "agent-2"):
        result = await runtime.invoke_exec(
            ctx=_ctx(agent_id=agent_id),
            arguments={"cmd": "agent-browser snapshot -i"},
            invoke_tool=invoke,
        )
        assert result == "snapshot"

    assert seen == [
        "AGENT_BROWSER_SESSION=strix-scan-1-agent-1 agent-browser snapshot -i",
        "AGENT_BROWSER_SESSION=strix-scan-1-agent-2 agent-browser snapshot -i",
    ]


@pytest.mark.asyncio
async def test_active_browser_action_reaches_the_reviewer(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, "guarded")
    reviewer = _StubReviewer()
    runtime._reviewer = reviewer

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "clicked"

    result = await runtime.invoke_exec(
        ctx=_ctx(turn_input=_SNAPSHOT_HISTORY),
        arguments={"cmd": "agent-browser click @e3"},
        invoke_tool=invoke,
    )

    assert result == "clicked"
    assert reviewer.calls == 1


@pytest.mark.asyncio
async def test_passive_browser_read_keeps_the_fast_path(tmp_path: Path) -> None:
    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "snapshot"

    runtime = _runtime(tmp_path, "guarded")
    reviewer = _StubReviewer()
    runtime._reviewer = reviewer
    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "agent-browser snapshot -i"},
        invoke_tool=invoke,
    )

    assert result == "snapshot"
    assert reviewer.calls == 0


@pytest.mark.asyncio
async def test_guarded_repeat_request_fails_closed(tmp_path: Path) -> None:
    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "bad"

    result = await _runtime(tmp_path, "guarded").invoke_mutating_tool(
        ctx=_ctx(),
        tool_name="repeat_request",
        raw_input="{}",
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert "effective method" in payload["safety"]["reason"]


class _Sandbox:
    async def read(self, path: Path) -> io.BytesIO:
        if path.as_posix() == "/workspace/app.py":
            return io.BytesIO(b"print(1)\n")
        raise FileNotFoundError(path)


def _script_ctx() -> Any:
    return SimpleNamespace(
        context={"agent_id": "agent-1", "sandbox_session": _Sandbox()},
        tool_call_id="call-1",
        turn_input=[],
    )


@pytest.mark.asyncio
async def test_write_stdin_is_blocked_in_guarded_mode(tmp_path: Path) -> None:
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await _runtime(tmp_path, "guarded").invoke_write_stdin(
        ctx=_ctx(),
        arguments={"session_id": "s", "chars": "rm -rf /workspace/app\n"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert "write_stdin is blocked" in payload["safety"]["reason"]
    assert invoked is False


@pytest.mark.asyncio
async def test_write_stdin_allows_an_interrupt(tmp_path: Path) -> None:
    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "interrupted"

    result = await _runtime(tmp_path, "guarded").invoke_write_stdin(
        ctx=_ctx(),
        arguments={"session_id": "s", "chars": "\x03"},
        invoke_tool=invoke,
    )

    assert result == "interrupted"


@pytest.mark.asyncio
async def test_write_stdin_is_untouched_when_safety_is_off(tmp_path: Path) -> None:
    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "typed"

    result = await _runtime(tmp_path, "off").invoke_write_stdin(
        ctx=_ctx(),
        arguments={"session_id": "s", "chars": "anything\n"},
        invoke_tool=invoke,
    )

    assert result == "typed"


@pytest.mark.asyncio
async def test_mutating_request_is_evidence_for_the_reviewer(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, "guarded")
    reviewer = _StubReviewer()
    runtime._reviewer = reviewer

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "reviewed"

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "curl -X DELETE https://example.test/v1/users/1042"},
        invoke_tool=invoke,
    )

    assert result == "reviewed"
    assert reviewer.calls == 1


@pytest.mark.asyncio
async def test_deferred_action_waits_for_human_and_executes_the_original_call(
    tmp_path: Path,
) -> None:
    requests: list[SafetyApprovalRequest] = []
    approval_started = asyncio.Event()
    release_approval = asyncio.Event()

    async def approve(request: SafetyApprovalRequest) -> bool:
        requests.append(request)
        approval_started.set()
        await release_approval.wait()
        return True

    runtime = _runtime(tmp_path, "guarded", approve)
    reviewer = _StubReviewer(decision=_deferred())
    runtime._reviewer = reviewer
    arguments = {"cmd": "nmap -sV example.test", "workdir": "/workspace"}
    original_arguments = dict(arguments)
    invoked: list[dict[str, Any]] = []

    async def invoke(_ctx: Any, raw_input: str) -> str:
        invoked.append(json.loads(raw_input))
        return "scanned"

    pending = asyncio.create_task(
        runtime.invoke_exec(ctx=_ctx(), arguments=arguments, invoke_tool=invoke)
    )
    await approval_started.wait()

    assert pending.done() is False
    arguments["cmd"] = "rm -rf /workspace"
    release_approval.set()

    assert await pending == "scanned"
    assert invoked == [original_arguments]
    assert reviewer.human_approval_available == [True]
    assert len(requests) == 1
    request = requests[0]
    assert request.agent_id == "agent-1"
    assert request.request_id == request.case_id
    assert request.tool_call_id == "call-1"
    assert request.tool_name == "exec_command"
    assert request.action == '{"cmd":"nmap -sV example.test","workdir":"/workspace"}'
    assert request.digest == hashlib.sha256(request.action.encode()).hexdigest()
    assert request.reason == "the endpoint effect is ambiguous"
    assert request.categories == ("ambiguous_effect",)
    assert request.risk == "medium"

    audit_path = tmp_path / ".state" / "safety-audit.jsonl"
    entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(
        entry["decision_source"] == "reviewer"
        and entry["summary"].get("approval", {}).get("status") == "requested"
        for entry in entries
    )
    assert any(
        entry["decision_source"] == "human"
        and entry["summary"].get("approval", {}).get("status") == "approved"
        for entry in entries
    )


@pytest.mark.asyncio
async def test_approval_cannot_execute_after_requesting_agent_stops(tmp_path: Path) -> None:
    class Coordinator:
        def __init__(self) -> None:
            self.calls = 0

        async def graph_snapshot(self) -> tuple[dict[str, str], dict[str, str], dict, dict]:
            self.calls += 1
            status = "running" if self.calls == 1 else "stopped"
            return {}, {"agent-1": status}, {}, {}

    async def approve(_request: SafetyApprovalRequest) -> bool:
        return True

    runtime = _runtime(tmp_path, "guarded", approve)
    runtime._reviewer = _StubReviewer(decision=_deferred())
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await runtime.invoke_exec(
        ctx=_ctx(coordinator=Coordinator()),
        arguments={"cmd": "nmap example.test"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["safety"]["source"] == "system"
    assert payload["safety"]["categories"][-1] == "agent_inactive"
    assert invoked is False


@pytest.mark.asyncio
async def test_oversized_action_cannot_be_deferred_to_human(tmp_path: Path) -> None:
    approval_called = False

    async def approve(_request: SafetyApprovalRequest) -> bool:
        nonlocal approval_called
        approval_called = True
        return True

    runtime = _runtime(tmp_path, "guarded", approve)
    runtime._reviewer = _StubReviewer(decision=_deferred())
    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "nmap " + "a" * 600},
        invoke_tool=_noop_invoke,
    )

    payload = json.loads(result)
    assert payload["safety"]["categories"] == ["approval_action_too_large"]
    assert approval_called is False


@pytest.mark.asyncio
async def test_human_denial_returns_a_human_sourced_block(tmp_path: Path) -> None:
    async def deny(_request: SafetyApprovalRequest) -> bool:
        return False

    runtime = _runtime(tmp_path, "guarded", deny)
    runtime._reviewer = _StubReviewer(decision=_deferred())
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "nmap -sV example.test"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["safety"]["source"] == "human"
    assert payload["safety"]["risk"] == "medium"
    assert "Human denied" in payload["safety"]["reason"]
    assert invoked is False


@pytest.mark.asyncio
async def test_lifecycle_cancellation_is_not_recorded_as_human_denial(tmp_path: Path) -> None:
    async def cancel(_request: SafetyApprovalRequest) -> str:
        return "cancelled"

    runtime = _runtime(tmp_path, "guarded", cancel)  # type: ignore[arg-type]
    runtime._reviewer = _StubReviewer(decision=_deferred())

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "nmap example.test"},
        invoke_tool=_noop_invoke,
    )

    payload = json.loads(result)
    assert payload["safety"]["source"] == "system"
    assert payload["safety"]["categories"][-1] == "approval_cancelled"
    assert "cancelled" in payload["safety"]["reason"]


@pytest.mark.asyncio
async def test_defer_without_an_approval_channel_blocks(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, "guarded")
    runtime._reviewer = _StubReviewer(decision=_deferred())

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "nmap -sV example.test"},
        invoke_tool=_noop_invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["safety"]["source"] == "reviewer"
    assert "no human approval channel" in payload["safety"]["reason"]


@pytest.mark.parametrize("command", ["rm -rf /workspace", ""])
@pytest.mark.asyncio
async def test_deterministic_and_incomplete_blocks_never_request_approval(
    tmp_path: Path,
    command: str,
) -> None:
    approval_calls = 0

    async def approve(_request: SafetyApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    runtime = _runtime(tmp_path, "guarded", approve)
    reviewer = _StubReviewer(decision=_deferred())
    runtime._reviewer = reviewer

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": command},
        invoke_tool=_noop_invoke,
    )

    assert json.loads(result)["status"] == "blocked"
    assert reviewer.calls == 0
    assert approval_calls == 0


@pytest.mark.parametrize(
    "decision",
    [
        SafetyDecision(
            allowed=False,
            source="review_error",
            reason="provider failed",
            categories=("review_error",),
        ),
        SafetyDecision(
            allowed=False,
            source="reviewer",
            reason="confidently destructive",
            categories=("destructive_effect",),
            risk="high",
        ),
    ],
)
@pytest.mark.asyncio
async def test_reviewer_errors_and_confident_blocks_never_request_approval(
    tmp_path: Path,
    decision: SafetyDecision,
) -> None:
    approval_calls = 0

    async def approve(_request: SafetyApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    runtime = _runtime(tmp_path, "guarded", approve)
    runtime._reviewer = _StubReviewer(decision=decision)

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "nmap -sV example.test"},
        invoke_tool=_noop_invoke,
    )

    assert json.loads(result)["status"] == "blocked"
    assert approval_calls == 0


@pytest.mark.asyncio
async def test_review_does_not_hold_the_workspace_lock(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, "guarded")
    concurrent = 0
    peak = 0

    async def on_review() -> None:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1

    runtime._reviewer = _StubReviewer(on_review)

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "ok"

    await asyncio.gather(
        *[
            runtime.invoke_exec(
                ctx=_ctx(),
                arguments={"cmd": f"nmap -sV host{index}"},
                invoke_tool=invoke,
            )
            for index in range(3)
        ]
    )

    assert peak == 3


@pytest.mark.asyncio
async def test_workspace_change_during_review_invalidates_the_decision(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, "guarded")

    async def on_review() -> None:
        runtime._workspace_epoch += 1

    runtime._reviewer = _StubReviewer(on_review)
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await runtime.invoke_exec(
        ctx=_script_ctx(),
        arguments={"cmd": "python /workspace/app.py"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["safety"]["categories"] == ["stale_evidence"]
    assert invoked is False


@pytest.mark.asyncio
async def test_workspace_change_during_human_approval_invalidates_the_decision(
    tmp_path: Path,
) -> None:
    runtime: SafetyRuntime

    async def approve(_request: SafetyApprovalRequest) -> bool:
        runtime._workspace_epoch += 1
        return True

    runtime = _runtime(tmp_path, "guarded", approve)
    runtime._reviewer = _StubReviewer(decision=_deferred())
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await runtime.invoke_exec(
        ctx=_script_ctx(),
        arguments={"cmd": "python /workspace/app.py"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["safety"]["categories"] == ["stale_evidence"]
    assert invoked is False


@pytest.mark.asyncio
async def test_unchanged_workspace_executes_after_review(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, "guarded")
    runtime._reviewer = _StubReviewer()

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "ran"

    result = await runtime.invoke_exec(
        ctx=_script_ctx(),
        arguments={"cmd": "python /workspace/app.py"},
        invoke_tool=invoke,
    )

    assert result == "ran"


@pytest.mark.asyncio
async def test_mutating_tool_is_untouched_when_safety_is_off(tmp_path: Path) -> None:
    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "patched"

    result = await _runtime(tmp_path, "off").invoke_mutating_tool(
        ctx=_ctx(),
        tool_name="apply_patch",
        raw_input="{}",
        invoke_tool=invoke,
    )

    assert result == "patched"


@pytest.mark.asyncio
async def test_guarded_patch_runs_and_advances_the_workspace_epoch(tmp_path: Path) -> None:
    """The epoch is what makes a script decision go stale, so the write that invalidates
    inspected sources has to advance it."""
    runtime = _runtime(tmp_path, "guarded")

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "patched"

    before = runtime._workspace_epoch
    result = await runtime.invoke_mutating_tool(
        ctx=_ctx(),
        tool_name="apply_patch",
        raw_input="{}",
        invoke_tool=invoke,
    )
    after = runtime._workspace_epoch

    assert result == "patched"
    assert after > before


@pytest.mark.asyncio
async def test_a_patch_during_review_invalidates_a_script_decision(tmp_path: Path) -> None:
    """End-to-end pairing of the two halves: apply_patch bumps the epoch, and a decision
    compiled before it is refused rather than executed against changed sources."""
    runtime = _runtime(tmp_path, "guarded")

    async def patch_during_review() -> None:
        await runtime.invoke_mutating_tool(
            ctx=_ctx(agent_id="agent-2"),
            tool_name="apply_patch",
            raw_input="{}",
            invoke_tool=_noop_invoke,
        )

    runtime._reviewer = _StubReviewer(patch_during_review)
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await runtime.invoke_exec(
        ctx=_script_ctx(),
        arguments={"cmd": "python /workspace/app.py"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["safety"]["categories"] == ["stale_evidence"]
    assert invoked is False


async def _noop_invoke(_ctx: Any, _raw_input: str) -> str:
    return "patched"


@pytest.mark.asyncio
async def test_a_workspace_command_advances_the_epoch(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, "guarded")
    runtime._reviewer = _StubReviewer()

    before = runtime._workspace_epoch
    result = await runtime.invoke_exec(
        ctx=_script_ctx(),
        arguments={"cmd": "python /workspace/app.py"},
        invoke_tool=_noop_invoke,
    )
    after = runtime._workspace_epoch

    assert result == "patched"
    assert after > before


@pytest.mark.asyncio
async def test_a_read_only_command_leaves_the_epoch_alone(tmp_path: Path) -> None:
    """A read cannot invalidate another agent's inspected sources, so it must not bump the
    epoch; if it did, concurrent reads would spuriously stale each other's decisions."""
    runtime = _runtime(tmp_path, "guarded")

    before = runtime._workspace_epoch
    await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "ls /workspace"},
        invoke_tool=_noop_invoke,
    )

    assert runtime._workspace_epoch == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "agent-browser tab new https://example.test/admin",
        "agent-browser tab close 2",
        "agent-browser session clear",
    ],
)
async def test_grouped_browser_verbs_reach_the_reviewer(tmp_path: Path, command: str) -> None:
    """The bare verb sits in the passive set, so these are the commands that would slip
    through if passivity were decided on the verb alone."""
    runtime = _runtime(tmp_path, "guarded")
    reviewer = _StubReviewer()
    runtime._reviewer = reviewer

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        return "reviewed"

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": command},
        invoke_tool=invoke,
    )

    assert result == "reviewed"
    assert reviewer.calls == 1


@pytest.mark.asyncio
async def test_bare_tab_listing_keeps_the_fast_path(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, "guarded")
    reviewer = _StubReviewer()
    runtime._reviewer = reviewer

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "agent-browser tab"},
        invoke_tool=_noop_invoke,
    )

    assert result == "patched"
    assert reviewer.calls == 0
