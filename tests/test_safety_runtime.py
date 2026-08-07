"""Run-scoped safety enforcement."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from strix.config.settings import SafetySettings
from strix.safety.runtime import SafetyRuntime
from strix.safety.types import SafetyDecision


if TYPE_CHECKING:
    from pathlib import Path


class _InspectionRunner:
    async def run(self, *, evidence_dir: str, script: str) -> str:
        return f"unused: {evidence_dir} {script}"


def _runtime(tmp_path: Path, mode: str) -> SafetyRuntime:
    return SafetyRuntime(
        scan_id="scan-1",
        mode=mode,  # type: ignore[arg-type]
        scope={},
        user_instruction="",
        settings=SafetySettings(),
        run_dir=tmp_path,
        sandbox_image="image",
        inspection_runner=_InspectionRunner(),
    )


def _ctx() -> Any:
    return SimpleNamespace(
        context={"agent_id": "agent-1", "sandbox_session": object()},
        tool_call_id="call-1",
        turn_input=[],
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
async def test_direct_browser_command_gets_per_agent_session(tmp_path: Path) -> None:
    seen: list[str] = []

    async def invoke(_ctx: Any, raw_input: str) -> str:
        seen.append(json.loads(raw_input)["cmd"])
        return "snapshot"

    result = await _runtime(tmp_path, "guarded").invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "agent-browser snapshot -i"},
        invoke_tool=invoke,
    )

    assert result == "snapshot"
    assert seen == ["AGENT_BROWSER_SESSION=strix-scan-1-agent-1 agent-browser snapshot -i"]


@pytest.mark.asyncio
async def test_observe_mode_blocks_browser_click(tmp_path: Path) -> None:
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await _runtime(tmp_path, "observe").invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "agent-browser click @e3"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert invoked is False


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


class _StubReviewer:
    def __init__(self, on_review: Any = None) -> None:
        self.on_review = on_review
        self.calls = 0

    async def review(self, bundle: Any) -> SafetyDecision:
        self.calls += 1
        if self.on_review is not None:
            await self.on_review()
        return SafetyDecision(
            allowed=True,
            source="reviewer",
            reason="allowed",
            case_id=bundle.case_id,
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
async def test_observe_mode_blocks_a_mutating_request_without_the_reviewer(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, "observe")
    reviewer = _StubReviewer()
    runtime._reviewer = reviewer
    invoked = False

    async def invoke(_ctx: Any, _raw_input: str) -> str:
        nonlocal invoked
        invoked = True
        return "bad"

    result = await runtime.invoke_exec(
        ctx=_ctx(),
        arguments={"cmd": "curl -X DELETE https://example.test/v1/users/1042"},
        invoke_tool=invoke,
    )

    payload = json.loads(result)
    assert payload["status"] == "blocked"
    assert payload["safety"]["source"] == "deterministic"
    assert "DELETE" in payload["safety"]["reason"]
    assert reviewer.calls == 0
    assert invoked is False


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
