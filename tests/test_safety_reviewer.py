"""The safety model may decide immediately or use one inspection call."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from agents.tool_context import ToolContext

import strix.safety.reviewer as reviewer_module
from strix.config.settings import SafetySettings
from strix.safety.evidence import EvidenceBundle
from strix.safety.reviewer import SafetyReviewer, run_inspection
from strix.safety.types import InspectionContext, SafetyVerdict


if TYPE_CHECKING:
    from pytest import MonkeyPatch


class _InspectionRunner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, *, evidence_dir: str, script: str) -> str:
        self.calls += 1
        return f"inspected {Path(evidence_dir).name}: {script}"


class _Result:
    def __init__(self, verdict: SafetyVerdict) -> None:
        self._verdict = verdict
        self.context_wrapper = SimpleNamespace(usage=SimpleNamespace())

    def final_output_as(self, _cls: type[Any], *, raise_if_incorrect_type: bool) -> SafetyVerdict:
        assert raise_if_incorrect_type is True
        return self._verdict


def _settings() -> Any:
    return SimpleNamespace(
        safety=SafetySettings(model="test-model"),
        llm=SimpleNamespace(
            model="main-model",
            extra_headers=None,
        ),
    )


@pytest.mark.asyncio
async def test_reviewer_is_capped_at_two_turns_and_zero_retries(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(agent: Any, *, input: str, context: Any, max_turns: int) -> _Result:  # noqa: A002
        captured.update(agent=agent, input=input, context=context, max_turns=max_turns)
        return _Result(
            SafetyVerdict(
                decision="allow",
                risk="low",
                categories=[],
                reason="read only",
                confidence=0.99,
            )
        )

    monkeypatch.setattr(reviewer_module, "load_settings", _settings)
    monkeypatch.setattr(reviewer_module, "configure_sdk_model_defaults", lambda _settings: None)
    monkeypatch.setattr(
        reviewer_module.StrixProvider, "get_model", lambda _self, _name: "test-model"
    )
    monkeypatch.setattr(reviewer_module.Runner, "run", fake_run)
    monkeypatch.setattr(reviewer_module, "get_global_report_state", lambda: None)
    bundle = EvidenceBundle(
        case_id="case-1",
        root=tmp_path,
        packet={"completeness": {"status": "complete"}},
        complete=True,
        incomplete_reasons=[],
    )

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(bundle)

    assert decision.allowed is True
    assert captured["max_turns"] == 2
    assert [tool.name for tool in captured["agent"].tools] == ["run_inspection"]
    assert captured["agent"].model_settings.retry.max_retries == 0
    # The cap also covers reasoning tokens; a verdict-sized budget would truncate the
    # structured output on a reasoning model and fail every review closed.
    assert captured["agent"].model_settings.max_tokens == SafetySettings().max_output_tokens


@pytest.mark.asyncio
async def test_review_budget_covers_both_turns_and_the_inspection(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_wait_for(awaitable: Any, *, timeout: float) -> Any:
        captured["timeout"] = timeout
        return await awaitable

    async def fake_run(_agent: Any, **_kwargs: Any) -> _Result:
        return _Result(
            SafetyVerdict(
                decision="allow",
                risk="low",
                categories=[],
                reason="read only",
                confidence=0.99,
            )
        )

    monkeypatch.setattr(reviewer_module, "load_settings", _settings)
    monkeypatch.setattr(reviewer_module, "configure_sdk_model_defaults", lambda _settings: None)
    monkeypatch.setattr(
        reviewer_module.StrixProvider, "get_model", lambda _self, _name: "test-model"
    )
    monkeypatch.setattr(reviewer_module.Runner, "run", fake_run)
    monkeypatch.setattr(reviewer_module, "get_global_report_state", lambda: None)
    monkeypatch.setattr(reviewer_module.asyncio, "wait_for", fake_wait_for)
    bundle = EvidenceBundle(
        case_id="case-budget",
        root=tmp_path,
        packet={"completeness": {"status": "complete"}},
        complete=True,
        incomplete_reasons=[],
    )

    await SafetyReviewer(inspection_runner=_InspectionRunner()).review(bundle)

    safety = SafetySettings()
    assert captured["timeout"] == 2 * safety.timeout + safety.inspection_timeout


@pytest.mark.asyncio
async def test_inspection_tool_can_only_run_once(tmp_path: Path) -> None:
    runner = _InspectionRunner()
    state = InspectionContext(evidence_dir=str(tmp_path), runner=runner)
    ctx = ToolContext(
        context=state,
        tool_name="run_inspection",
        tool_call_id="inspect-1",
        tool_arguments="{}",
    )
    raw = json.dumps({"reason": "correlate files", "script": "print('ok')"})

    first = await run_inspection.on_invoke_tool(ctx, raw)
    second = await run_inspection.on_invoke_tool(ctx, raw)

    assert "inspected" in first
    assert "already used" in second
    assert runner.calls == 1


@pytest.mark.asyncio
async def test_reviewer_failure_blocks(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    monkeypatch.setattr(reviewer_module, "load_settings", _settings)
    monkeypatch.setattr(reviewer_module, "configure_sdk_model_defaults", lambda _settings: None)
    monkeypatch.setattr(
        reviewer_module.StrixProvider, "get_model", lambda _self, _name: "test-model"
    )
    monkeypatch.setattr(reviewer_module.Runner, "run", fail)
    bundle = EvidenceBundle(
        case_id="case-2",
        root=tmp_path,
        packet={"completeness": {"status": "complete"}},
        complete=True,
        incomplete_reasons=[],
    )

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(bundle)

    assert decision.allowed is False
    assert decision.source == "review_error"


@pytest.fixture
def _patched_sdk(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(reviewer_module, "load_settings", _settings)
    monkeypatch.setattr(reviewer_module, "configure_sdk_model_defaults", lambda _settings: None)
    monkeypatch.setattr(
        reviewer_module.StrixProvider, "get_model", lambda _self, _name: "test-model"
    )
    monkeypatch.setattr(reviewer_module, "get_global_report_state", lambda: None)


def _bundle(tmp_path: Path, case_id: str) -> EvidenceBundle:
    return EvidenceBundle(
        case_id=case_id,
        root=tmp_path,
        packet={"completeness": {"status": "complete"}},
        complete=True,
        incomplete_reasons=[],
    )


def _verdict_run(verdict: SafetyVerdict) -> Any:
    async def fake_run(_agent: Any, **_kwargs: Any) -> _Result:
        return _Result(verdict)

    return fake_run


@pytest.mark.asyncio
@pytest.mark.usefixtures("_patched_sdk")
async def test_low_confidence_allow_is_refused(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """An allow the reviewer is unsure of is the case the threshold exists for."""
    monkeypatch.setattr(
        reviewer_module.Runner,
        "run",
        _verdict_run(
            SafetyVerdict(
                decision="allow",
                risk="medium",
                categories=["target_mutation"],
                reason="probably fine",
                confidence=0.5,
            )
        ),
    )

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(
        _bundle(tmp_path, "case-low-confidence")
    )

    assert decision.allowed is False
    assert decision.source == "reviewer"
    assert "below the 0.75 allow threshold" in decision.reason
    assert decision.categories == ("target_mutation",)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_patched_sdk")
async def test_confident_allow_passes(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        reviewer_module.Runner,
        "run",
        _verdict_run(
            SafetyVerdict(
                decision="allow",
                risk="low",
                categories=[],
                reason="read only",
                confidence=0.8,
            )
        ),
    )

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(
        _bundle(tmp_path, "case-confident")
    )

    assert decision.allowed is True
    assert decision.source == "reviewer"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_patched_sdk")
async def test_block_verdict_is_returned_as_a_block(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reviewer_module.Runner,
        "run",
        _verdict_run(
            SafetyVerdict(
                decision="block",
                risk="high",
                categories=["state_mutation"],
                reason="deletes a record",
                confidence=0.99,
            )
        ),
    )

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(
        _bundle(tmp_path, "case-block")
    )

    assert decision.allowed is False
    assert decision.source == "reviewer"
    assert decision.reason == "deletes a record"


@pytest.mark.asyncio
async def test_missing_model_configuration_blocks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reviewer_module,
        "load_settings",
        lambda: SimpleNamespace(
            safety=SafetySettings(model=None),
            llm=SimpleNamespace(model="", extra_headers=None),
        ),
    )

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(
        _bundle(tmp_path, "case-no-model")
    )

    assert decision.allowed is False
    assert decision.source == "review_error"
    assert decision.categories == ("review_unavailable",)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_patched_sdk")
async def test_allow_after_a_failed_inspection_is_refused(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """The reviewer decides from the inspection's own output, so an inspection that failed
    must not be able to underwrite an allow."""

    async def fake_run(_agent: Any, *, context: Any, **_kwargs: Any) -> _Result:
        context.incomplete = True
        return _Result(
            SafetyVerdict(
                decision="allow",
                risk="low",
                categories=[],
                reason="looked fine",
                confidence=0.99,
            )
        )

    monkeypatch.setattr(reviewer_module.Runner, "run", fake_run)

    decision = await SafetyReviewer(inspection_runner=_InspectionRunner()).review(
        _bundle(tmp_path, "case-bad-inspection")
    )

    assert decision.allowed is False
    assert decision.categories == ("inspection_incomplete",)


@pytest.mark.parametrize(
    "output",
    [
        "Inspection failed: frozen evidence directory is unavailable.",
        "Inspection exit code: 1",
        "... output truncated ...",
    ],
)
@pytest.mark.asyncio
async def test_inspection_failure_output_is_recognized(tmp_path: Path, output: str) -> None:
    """These strings are produced in inspection.py and matched by substring here, so a
    reword on either side silently stops marking failed inspections."""

    class _Failing:
        async def run(self, *, evidence_dir: str, script: str) -> str:  # noqa: ARG002
            return output

    state = InspectionContext(evidence_dir=str(tmp_path), runner=_Failing())
    ctx = ToolContext(
        context=state,
        tool_name="run_inspection",
        tool_call_id="inspect-1",
        tool_arguments="{}",
    )

    await run_inspection.on_invoke_tool(
        ctx, json.dumps({"reason": "check", "script": "print('x')"})
    )

    assert state.incomplete is True


def test_prompt_judges_security_testing_by_effect_not_technique() -> None:
    """Pins the effect-based guardrails so a future edit cannot silently revert to
    blocking in-scope offensive testing on the technique alone."""
    prompt = reviewer_module._SAFETY_PROMPT

    # Authorization framing and the effect-not-technique rule.
    assert "authorized penetration test" in prompt
    assert (
        'That an action is a "SQL\ninjection"' in prompt
        or "not, by itself, a reason to block" in prompt
    )
    # Read probes pass; writes and destruction block.
    assert "OR 1=1" in prompt
    for keyword in ("DROP", "DELETE", "INSERT", "TRUNCATE", "OUTFILE", "xp_cmdshell"):
        assert keyword in prompt
    # Fail-closed on ambiguity is preserved.
    assert "does not settle whether the effect is destructive" in prompt
    # Non-negotiable guardrails survive.
    assert 'Never allow when completeness.status is not "complete"' in prompt
    assert "Deterministic policy blocks cannot be overridden" in prompt
    assert "analysis.mutating_request is\nnever passive" in prompt


def test_prompt_scopes_subdomains_and_input_files() -> None:
    prompt = reviewer_module._SAFETY_PROMPT
    assert "authorized domain covers its subdomains" in prompt
    assert 'role "input"' in prompt or 'role "input"' in prompt
