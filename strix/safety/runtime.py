"""Run-scoped safety orchestration and pre-execution enforcement."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from strix.safety.audit import SafetyAudit
from strix.safety.evidence import EvidenceBundle, compile_evidence, parse_command
from strix.safety.inspection import DockerInspectionRunner, InspectionRunner
from strix.safety.reviewer import SafetyReviewer
from strix.safety.types import SafetyDecision


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pathlib import Path

    from strix.config.settings import SafetyMode, SafetySettings
    from strix.safety.evidence import CommandPlan


InvokeTool = Callable[[Any, str], Awaitable[Any]]

_PASSIVE_BROWSER_ACTIONS = frozenset({"snapshot", "get", "is", "tab", "session"})
# ETX only: it discards the terminal's line buffer instead of submitting it, so it
# cannot smuggle a command through a session that was approved for something else.
_INTERRUPT_CHARS = frozenset({"\x03"})


@dataclass(frozen=True, slots=True)
class _ExecReview:
    decision: SafetyDecision
    summary: dict[str, Any]
    workspace_epoch: int
    workspace_evidence: bool


class SafetyRuntime:
    """One immutable safety policy shared by every agent in a scan."""

    def __init__(
        self,
        *,
        scan_id: str,
        mode: SafetyMode,
        scope: dict[str, Any],
        user_instruction: str,
        settings: SafetySettings,
        run_dir: Path,
        sandbox_image: str,
        inspection_runner: InspectionRunner | None = None,
    ) -> None:
        self.scan_id = scan_id
        self.mode = mode
        self.scope = scope
        self.user_instruction = user_instruction
        self.settings = settings
        self._workspace_lock = asyncio.Lock()
        self._workspace_epoch = 0
        self._browser_locks: dict[str, asyncio.Lock] = {}
        runner = inspection_runner or DockerInspectionRunner(
            settings=settings,
            fallback_image=sandbox_image,
        )
        self._reviewer = SafetyReviewer(inspection_runner=runner)
        self._audit = SafetyAudit(run_dir / ".state" / "safety-audit.jsonl")

    async def invoke_exec(
        self,
        *,
        ctx: Any,
        arguments: dict[str, Any],
        invoke_tool: InvokeTool,
    ) -> Any:
        raw_input = json.dumps(arguments, ensure_ascii=False)
        if self.mode == "off":
            return await invoke_tool(ctx, raw_input)

        agent_id = str(getattr(ctx, "context", {}).get("agent_id", "unknown"))
        plan = parse_command(str(arguments.get("cmd") or ""))

        # The review is not serialized: holding the run-wide workspace lock across a model
        # call would put every other agent behind this one. The lock covers execution only,
        # and the epoch recheck below rejects a decision whose evidence has since changed.
        review = await self._decide_exec(ctx=ctx, arguments=arguments)
        await self._audit.record(
            agent_id=agent_id,
            tool_call_id=str(getattr(ctx, "tool_call_id", "unknown")),
            tool_name="exec_command",
            decision=review.decision,
            summary=review.summary,
        )
        if not review.decision.allowed:
            return self.blocked_result(review.decision)

        browser_lock = (
            self._browser_locks.setdefault(agent_id, asyncio.Lock()) if plan.browser else None
        )
        if browser_lock is not None:
            await browser_lock.acquire()
        try:
            if plan.read_only or plan.browser:
                return await self._execute(
                    ctx=ctx,
                    agent_id=agent_id,
                    arguments=arguments,
                    plan=plan,
                    review=review,
                    invoke_tool=invoke_tool,
                )
            async with self._workspace_lock:
                return await self._execute(
                    ctx=ctx,
                    agent_id=agent_id,
                    arguments=arguments,
                    plan=plan,
                    review=review,
                    invoke_tool=invoke_tool,
                    workspace_locked=True,
                )
        finally:
            if browser_lock is not None:
                browser_lock.release()

    async def _execute(
        self,
        *,
        ctx: Any,
        agent_id: str,
        arguments: dict[str, Any],
        plan: CommandPlan,
        review: _ExecReview,
        invoke_tool: InvokeTool,
        workspace_locked: bool = False,
    ) -> Any:
        tool_call_id = str(getattr(ctx, "tool_call_id", "unknown"))
        if review.workspace_evidence and self._workspace_epoch != review.workspace_epoch:
            stale = SafetyDecision(
                allowed=False,
                source="deterministic",
                reason=(
                    "The workspace changed while this action was under review; the inspected "
                    "sources may no longer be what would run. Re-issue the command."
                ),
                categories=("stale_evidence",),
                case_id=review.decision.case_id,
            )
            await self._audit.record(
                agent_id=agent_id,
                tool_call_id=tool_call_id,
                tool_name="exec_command",
                decision=stale,
                summary=review.summary,
            )
            return self.blocked_result(stale)

        effective = dict(arguments)
        if plan.browser:
            session = f"strix-{self.scan_id}-{agent_id}"
            effective["cmd"] = f"AGENT_BROWSER_SESSION={shlex.quote(session)} {arguments['cmd']}"
        try:
            result = await invoke_tool(ctx, json.dumps(effective, ensure_ascii=False))
        except Exception:
            await self._audit.record(
                agent_id=agent_id,
                tool_call_id=tool_call_id,
                tool_name="exec_command",
                decision=review.decision,
                summary=review.summary,
                execution_status="failed",
            )
            raise
        finally:
            if workspace_locked:
                self._workspace_epoch += 1
        await self._audit.record(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            tool_name="exec_command",
            decision=review.decision,
            summary=review.summary,
            execution_status="succeeded",
        )
        return result

    async def invoke_write_stdin(
        self,
        *,
        ctx: Any,
        arguments: dict[str, Any],
        invoke_tool: InvokeTool,
    ) -> Any:
        raw_input = json.dumps(arguments, ensure_ascii=False)
        if self.mode == "off":
            return await invoke_tool(ctx, raw_input)

        chars = arguments.get("chars")
        payload = chars if isinstance(chars, str) else ""
        case_id = f"safety-{uuid4().hex[:12]}"
        if payload and set(payload) <= _INTERRUPT_CHARS:
            decision = SafetyDecision(
                allowed=True,
                source="deterministic",
                reason="Interrupt-only stdin payload.",
                case_id=case_id,
            )
        else:
            decision = SafetyDecision(
                allowed=False,
                source="deterministic",
                reason=(
                    "write_stdin is blocked in safety modes: what a live session does with the "
                    "payload depends on the process reading it and on buffered input, so the "
                    "effective action cannot be compiled before dispatch. Issue the command as "
                    "its own exec_command call."
                ),
                categories=("unreviewable_stdin",),
                case_id=case_id,
            )
        await self._audit.record(
            agent_id=str(getattr(ctx, "context", {}).get("agent_id", "unknown")),
            tool_call_id=str(getattr(ctx, "tool_call_id", "unknown")),
            tool_name="write_stdin",
            decision=decision,
            summary={"payload_digest": self._command_digest(payload)},
        )
        if not decision.allowed:
            return self.blocked_result(decision)
        return await invoke_tool(ctx, raw_input)

    async def _decide_exec(
        self,
        *,
        ctx: Any,
        arguments: dict[str, Any],
    ) -> _ExecReview:
        case_id = f"safety-{uuid4().hex[:12]}"
        workspace_epoch = self._workspace_epoch
        bundle = await compile_evidence(
            case_id=case_id,
            ctx=ctx,
            arguments=arguments,
            mode=self.mode,
            scope=self.scope,
            user_instruction=self.user_instruction,
            settings=self.settings,
            workspace_epoch=workspace_epoch,
        )
        summary = {
            "command_digest": self._command_digest(str(arguments.get("cmd") or "")),
            "executable": bundle.packet.get("pending_action", {}).get("executable"),
            "browser_action": bundle.packet.get("pending_action", {}).get("browser_action"),
            "artifact_digests": [
                artifact.get("digest")
                for artifact in bundle.packet.get("artifacts", [])
                if isinstance(artifact, dict)
            ],
            "complete": bundle.complete,
        }
        try:
            return _ExecReview(
                decision=await self._decide_bundle(bundle, case_id),
                summary=summary,
                workspace_epoch=workspace_epoch,
                workspace_evidence=bundle.workspace_evidence,
            )
        finally:
            bundle.cleanup()

    async def _decide_bundle(self, bundle: EvidenceBundle, case_id: str) -> SafetyDecision:
        if bundle.deterministic_block:
            return SafetyDecision(
                allowed=False,
                source="deterministic",
                reason=bundle.deterministic_block,
                categories=("policy_block",),
                case_id=case_id,
            )
        if not bundle.complete:
            return SafetyDecision(
                allowed=False,
                source="deterministic",
                reason="Safety evidence is incomplete: " + "; ".join(bundle.incomplete_reasons),
                categories=("incomplete_evidence",),
                case_id=case_id,
            )
        if self.mode == "observe":
            observed = self._observe_mode_block(bundle, case_id)
            if observed is not None:
                return observed
        if bundle.deterministic_allow:
            return SafetyDecision(
                allowed=True,
                source="deterministic",
                reason=bundle.deterministic_allow,
                case_id=case_id,
            )
        return await self._reviewer.review(bundle)

    @staticmethod
    def _observe_mode_block(bundle: EvidenceBundle, case_id: str) -> SafetyDecision | None:
        """Enforce the passive-only contract without relying on the model reviewer."""
        if bundle.packet.get("browser") is not None:
            action = bundle.packet.get("pending_action", {}).get("browser_action")
            if action not in _PASSIVE_BROWSER_ACTIONS:
                return SafetyDecision(
                    allowed=False,
                    source="deterministic",
                    reason=(
                        f"Browser action {action!r} is not passive and is blocked in observe mode."
                    ),
                    categories=("target_mutation",),
                    case_id=case_id,
                )
            return None
        if bundle.mutating_request is not None:
            return SafetyDecision(
                allowed=False,
                source="deterministic",
                reason=(
                    f"{bundle.mutating_request} changes target state and is blocked in "
                    "observe mode."
                ),
                categories=("target_mutation",),
                case_id=case_id,
            )
        return None

    async def invoke_mutating_tool(
        self,
        *,
        ctx: Any,
        tool_name: str,
        raw_input: str,
        invoke_tool: InvokeTool,
    ) -> Any:
        if self.mode == "off":
            return await invoke_tool(ctx, raw_input)
        case_id = f"safety-{uuid4().hex[:12]}"
        if self.mode == "observe":
            decision = SafetyDecision(
                allowed=False,
                source="deterministic",
                reason=f"{tool_name} mutates state and is blocked in observe mode.",
                categories=("state_mutation",),
                case_id=case_id,
            )
            return self.blocked_result(decision)
        if tool_name == "repeat_request":
            decision = SafetyDecision(
                allowed=False,
                source="deterministic",
                reason=(
                    "repeat_request is blocked in guarded mode until the final effective method, "
                    "destination, headers, and body can be compiled before dispatch."
                ),
                categories=("unresolved_network_mutation",),
                case_id=case_id,
            )
            return self.blocked_result(decision)
        async with self._workspace_lock:
            # Guarded workspaces are isolated copies; patches remain local to the run.
            try:
                return await invoke_tool(ctx, raw_input)
            finally:
                self._workspace_epoch += 1

    @staticmethod
    def blocked_result(decision: SafetyDecision) -> str:
        return json.dumps(
            {
                "success": False,
                "status": "blocked",
                "error": "Action blocked by safety policy",
                "safety": {
                    "case_id": decision.case_id,
                    "source": decision.source,
                    "reason": decision.reason,
                    "categories": list(decision.categories),
                },
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _command_digest(command: str) -> str:
        return hashlib.sha256(command.encode()).hexdigest()


def safety_runtime_from_context(ctx: Any) -> SafetyRuntime | None:
    inner = getattr(ctx, "context", None)
    if not isinstance(inner, dict):
        return None
    runtime = inner.get("safety_runtime")
    return runtime if isinstance(runtime, SafetyRuntime) else None
