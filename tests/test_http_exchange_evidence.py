"""HTTP exchange evidence on findings (upstream 22959a7 + 95e085e).

``http_exchange_ids`` is evidence attached to a lifecycle finding, never a second
definition of whether the finding is valid: the lifecycle state stays the only
authority, and supplying exchanges must not promote a candidate. The ids are
verified against the current Caido project, and a proxy outage drops them with a
warning instead of losing a valid vulnerability.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from agents.tool_context import ToolContext

from strix.config.settings import DEFAULT_MAX_TURNS
from strix.interface.tui.runtime import GoTuiRuntime
from strix.report.state import ReportState, set_global_report_state
from strix.tools.reporting.tool import (
    _do_create,
    create_vulnerability_report,
    update_vulnerability_report,
)


if TYPE_CHECKING:
    from pathlib import Path


_CVSS = {
    "attack_vector": "N",
    "attack_complexity": "L",
    "privileges_required": "N",
    "user_interaction": "N",
    "scope": "U",
    "confidentiality": "H",
    "integrity": "H",
    "availability": "H",
}

_DEP_CONTEXT = {
    "attack_vector": "N",
    "attack_complexity": "L",
    "privileges_required": "N",
    "user_interaction": "N",
    "scope": "U",
    "confidentiality": "N",
    "integrity": "N",
    "availability": "H",
}


@pytest.fixture
def report_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportState:
    monkeypatch.chdir(tmp_path)
    state = ReportState(run_name="test-run")
    set_global_report_state(state)
    return state


def _create_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "title": "IDOR on /api/audits",
        "description": "Any authenticated user can read another tenant's audits.",
        "impact": "Cross-tenant disclosure of audit records.",
        "target": "https://app.example.com/api/audits/1",
        "technical_analysis": "The handler never compares the tenant id.",
        "poc_description": "Swap the id in the request.",
        "poc_script_code": "curl -H 'Cookie: s=...' https://app.example.com/api/audits/2",
        "remediation_steps": "Scope the lookup to the caller's tenant.",
        "evidence": "HTTP 200 with another tenant's rows.",
        "assumptions": "A second account exists.",
        "counterevidence": "No authorization check found on this path.",
        "confidence": "high",
        "severity_change_conditions": "A tenant check upstream would lower this.",
        "fix_effort": "low",
        "cvss_breakdown": _CVSS,
        # Declared without defaults on the impl, so pass them explicitly.
        "endpoint": "/api/audits/1",
        "method": "GET",
        "cve": None,
        "cwe": "CWE-639",
        "code_locations": None,
    }
    args.update(overrides)
    return args


async def _invoke(tool: Any, args: dict[str, Any], **extra: Any) -> dict[str, Any]:
    ctx = ToolContext(
        context={"agent_id": "agent-1", **extra},
        tool_name=tool.name,
        tool_call_id="call-1",
        tool_arguments="{}",
    )
    raw: str = await tool.on_invoke_tool(ctx, json.dumps(args))
    return cast("dict[str, Any]", json.loads(raw))


def _existing(*ids: str) -> Any:
    """A proxy whose project contains exactly ``ids``."""

    async def _fake(_ctx: Any, request_ids: list[str]) -> set[str]:
        return {rid for rid in request_ids if rid in set(ids)}

    return _fake


def _outage(*_args: Any, **_kwargs: Any) -> Any:
    async def _boom(_ctx: Any, _request_ids: list[str]) -> set[str]:
        raise RuntimeError("caido unreachable")

    return _boom


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_create_with_verified_ids_stores_them_normalized(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _existing("11", "22"))

    result = await _invoke(
        create_vulnerability_report,
        _create_args(http_exchange_ids=["22", "11", "22"]),
    )

    assert result["success"] is True
    stored = report_state.vulnerability_reports[0]
    # Deduplicated once, first-seen order preserved.
    assert stored["http_exchange_ids"] == ["22", "11"]


@pytest.mark.asyncio
async def test_create_with_nonexistent_id_fails_and_stores_nothing(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _existing("11"))

    result = await _invoke(
        create_vulnerability_report,
        _create_args(http_exchange_ids=["11", "99"]),
    )

    assert result["success"] is False
    assert any("99" in err for err in result["errors"])
    assert report_state.vulnerability_reports == []


@pytest.mark.asyncio
async def test_create_survives_a_proxy_outage_without_the_ids(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreachable proxy must not cost us a valid finding."""
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _outage())

    result = await _invoke(
        create_vulnerability_report,
        _create_args(http_exchange_ids=["11"]),
    )

    assert result["success"] is True
    assert "were not stored" in result["warning"]
    stored = report_state.vulnerability_reports[0]
    # Unverified ids are never recorded as evidence.
    assert "http_exchange_ids" not in stored


@pytest.mark.asyncio
async def test_create_callback_failure_leaves_in_memory_state_unchanged(
    report_state: ReportState,
) -> None:
    """Persistence accepts the report before it is appended, so a throwing
    callback leaves nothing behind and the retry can file cleanly.

    The failure surfaces to the caller (the SDK turns a raised error into a
    model-visible tool failure); what matters is that the finding is not left
    half-registered, so the retry is unambiguous.
    """

    def _boom(_report: dict[str, Any]) -> None:
        raise RuntimeError("run.json write failed")

    report_state.vulnerability_found_callback = _boom

    with pytest.raises(RuntimeError, match=r"run\.json write failed"):
        await _do_create(**_create_args())

    assert report_state.vulnerability_reports == []

    # The retry succeeds once persistence works again.
    report_state.vulnerability_found_callback = None
    retry = await _do_create(**_create_args())
    assert retry["success"] is True
    assert len(report_state.vulnerability_reports) == 1


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #


async def _filed(report_state: ReportState, ids: list[str] | None = None) -> str:
    return report_state.add_vulnerability_report(
        title="IDOR on /api/audits",
        severity="high",
        agent_id="agent-1",
        http_exchange_ids=ids,
    )


def _report(report_state: ReportState, report_id: str) -> dict[str, Any]:
    return next(r for r in report_state.vulnerability_reports if r["id"] == report_id)


@pytest.mark.asyncio
async def test_update_replaces_the_linked_exchanges(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_id = await _filed(report_state, ["11"])
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _existing("22"))

    result = await _invoke(
        update_vulnerability_report,
        {
            "report_id": report_id,
            "update_reason": "attach the exploit exchange",
            "http_exchange_ids": ["22"],
        },
    )

    assert result["success"] is True
    assert _report(report_state, report_id)["http_exchange_ids"] == ["22"]


@pytest.mark.asyncio
async def test_update_with_an_empty_list_clears_the_exchanges(
    report_state: ReportState,
) -> None:
    report_id = await _filed(report_state, ["11"])

    result = await _invoke(
        update_vulnerability_report,
        {"report_id": report_id, "update_reason": "evidence withdrawn", "http_exchange_ids": []},
    )

    assert result["success"] is True
    assert not _report(report_state, report_id).get("http_exchange_ids")


@pytest.mark.asyncio
async def test_update_omitting_the_field_leaves_exchanges_untouched(
    report_state: ReportState,
) -> None:
    report_id = await _filed(report_state, ["11"])

    result = await _invoke(
        update_vulnerability_report,
        {
            "report_id": report_id,
            "update_reason": "correct the evidence",
            "evidence": "HTTP 200 with another tenant's rows, re-verified.",
        },
    )

    assert result["success"] is True
    assert _report(report_state, report_id)["http_exchange_ids"] == ["11"]


@pytest.mark.asyncio
async def test_update_with_a_nonexistent_id_leaves_report_and_revision_untouched(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_id = await _filed(report_state, ["11"])
    before = _report(report_state, report_id)
    before_history = len(before.get("update_history") or [])

    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _existing("11"))
    result = await _invoke(
        update_vulnerability_report,
        {"report_id": report_id, "update_reason": "attach", "http_exchange_ids": ["77"]},
    )

    assert result["success"] is False
    after = _report(report_state, report_id)
    assert after["http_exchange_ids"] == ["11"]
    assert len(after.get("update_history") or []) == before_history


@pytest.mark.asyncio
async def test_update_during_an_outage_does_not_erase_existing_ids(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped update must not silently withdraw evidence already attached."""
    report_id = await _filed(report_state, ["11"])
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _outage())

    result = await _invoke(
        update_vulnerability_report,
        {
            "report_id": report_id,
            "update_reason": "correct the evidence",
            "evidence": "Re-verified over HTTP.",
            "http_exchange_ids": ["22"],
        },
    )

    assert result["success"] is True
    assert "were not stored" in result["warning"]
    assert _report(report_state, report_id)["http_exchange_ids"] == ["11"]
    assert _report(report_state, report_id)["evidence"] == "Re-verified over HTTP."


@pytest.mark.asyncio
async def test_update_with_only_ids_during_an_outage_is_not_reported_as_empty(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dropped-evidence case must surface, not become 'no fields to update'."""
    report_id = await _filed(report_state)
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _outage())

    result = await _invoke(
        update_vulnerability_report,
        {"report_id": report_id, "update_reason": "attach", "http_exchange_ids": ["22"]},
    )

    assert result["success"] is False
    assert "were not stored" in result["error"]
    assert "No fields to update" not in json.dumps(result)
    assert _report(report_state, report_id).get("update_history") is None


# --------------------------------------------------------------------------- #
# lifecycle authority
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supplying_exchanges_never_changes_the_lifecycle_state(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _existing("11"))
    candidate = report_state.add_vulnerability_report(
        title="Candidate", severity="medium", agent_id="agent-1", state="candidate"
    )

    result = await _invoke(
        update_vulnerability_report,
        {"report_id": candidate, "update_reason": "attach proof", "http_exchange_ids": ["11"]},
    )

    assert result["success"] is True
    # Evidence attached, still a candidate: ids are not a promotion.
    assert _report(report_state, candidate)["http_exchange_ids"] == ["11"]
    assert _report(report_state, candidate)["state"] == "candidate"


@pytest.mark.asyncio
async def test_dependency_revision_rejects_http_exchange_ids(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package finding has no HTTP exchange to point at."""
    monkeypatch.setattr("strix.tools.reporting.tool.existing_request_ids", _existing("11"))
    report_id = report_state.add_vulnerability_report(
        title="Vulnerable lodash",
        severity="high",
        agent_id="agent-1",
        finding_class="dependency",
        dependency_metadata={
            "package_name": "lodash",
            "installed_version": "4.17.4",
            "package_ecosystem": "npm",
        },
    )

    result = await _invoke(
        update_vulnerability_report,
        {"report_id": report_id, "update_reason": "attach", "http_exchange_ids": ["11"]},
    )

    assert result["success"] is False
    assert "http_exchange_ids" in json.dumps(result)
    # Rejected before any mutation: the finding keeps its evidence shape.
    assert "http_exchange_ids" not in _report(report_state, report_id)


# --------------------------------------------------------------------------- #
# TUI sync fingerprint
# --------------------------------------------------------------------------- #


def _tui_args() -> argparse.Namespace:
    return argparse.Namespace(
        needs_setup=True,
        targets_info=[],
        instruction=None,
        scan_mode="deep",
        max_budget_usd=None,
        max_turns=DEFAULT_MAX_TURNS,
        scope_mode="auto",
        diff_base=None,
        local_sources=[],
        diff_scope={"active": False},
        user_explicit_instruction=None,
        run_name="test-run",
    )


def test_in_place_revision_changes_the_runtime_sync_fingerprint(
    report_state: ReportState,
) -> None:
    """The id is unchanged by a revision, so the fingerprint must track revisions
    or the Go TUI sync loop leaves the viewer showing stale evidence."""
    runtime = GoTuiRuntime(_tui_args())
    runtime.report_state = report_state
    report_id = report_state.add_vulnerability_report(
        title="IDOR", severity="high", agent_id="agent-1"
    )

    before = runtime._runtime_sync_fingerprint()
    assert runtime._runtime_sync_fingerprint() == before  # stable when idle

    report_state.update_vulnerability_report(
        report_id, {"http_exchange_ids": ["11", "22"]}, update_reason="attach evidence"
    )

    after = runtime._runtime_sync_fingerprint()
    assert after != before
    # The id itself did not change — only the revision did.
    assert report_state.vulnerability_reports[0]["id"] == report_id
