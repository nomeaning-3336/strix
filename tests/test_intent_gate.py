"""The developer-intent gate: a finding contradicted by the repo's own docs is not a finding.

The regression these tests exist for
-------------------------------------
A scan of a cloned GitLab reported that a user-boundary fine-grained PAT being
accepted on group-targeted project imports was a scope bypass. The clone already
contained ``doc/development/permissions/granular_access/rest_api_implementation_guide.md``
(lines ~181-199) stating that import endpoints may resolve a group *or* a user
boundary and that access is granted when *any one* resolved boundary is
satisfied — matching what ``lib/api/import_bitbucket.rb`` and
``lib/api/import_github.rb`` declared and what their request specs asserted. The
agent even cited that document in its counterevidence and filed anyway.

So this is tested at two levels: the gate's own decisions (deterministic, no model
calls), and the reporting tool's refusal to append a report the gate blocks. The
documented-but-still-dangerous primitive is tested too: a real conflicting
security invariant must be enough to file.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from strix.report import intent as intent_gate
from strix.report.state import ReportState
from strix.tools.reporting.tool import (
    _do_create,
    _filing_intent_gate,
)


if TYPE_CHECKING:
    from pathlib import Path


# The real path and section that the scan cited and then ignored.
GITLAB_GUIDE = (
    "doc/development/permissions/granular_access/rest_api_implementation_guide.md:181-199"
)
GITLAB_ROUTE = "lib/api/import_bitbucket.rb:41"

_CVSS = {
    "attack_vector": "N",
    "attack_complexity": "L",
    "privileges_required": "L",
    "user_interaction": "N",
    "scope": "U",
    "confidentiality": "L",
    "integrity": "L",
    "availability": "N",
}

_SEARCH_SCOPE = [
    "doc/development/permissions/ (grep for 'import', 'boundary', 'granular')",
    "spec/requests/api/import_*.rb",
    "git log -S import_bitbucket -- lib/api/import_bitbucket.rb",
    "CHANGELOG.md / doc/architecture/decisions",
]

_KNOWN_ISSUE_SEARCH = (
    "Searched the changelog, ADRs and linked issue references for the import "
    "boundary semantics; found only the granular-access guide describing them."
)

_ALTERNATIVE_SEMANTICS = (
    "The route resolves a group OR user boundary and grants access when any one "
    "resolved boundary is satisfied, so a user-boundary token satisfying one branch "
    "is not itself a bypass of the other."
)

_ASSUMPTIONS = (
    "Assuming the documented OR semantics are current for this version; no migration "
    "note suggests the boundary rule changed."
)


def _evidence() -> list[dict[str, Any]]:
    return [
        {
            "source": GITLAB_GUIDE,
            "authority": "implementation_spec",
            "supports": (
                "Import endpoints may use a group or user boundary; access is granted "
                "when any one resolved boundary is satisfied."
            ),
        },
        {
            "source": "spec/requests/api/import_bitbucket_spec.rb:118",
            "authority": "implementation_spec",
            "supports": "The spec asserts import is authorized for a user-boundary token.",
        },
    ]


def _review(verdict: str, *, reviewed_by: str = "reviewer-1") -> dict[str, Any]:
    return {
        "reviewer_kind": "independent_agent",
        "reviewed_by": reviewed_by,
        "verdict": verdict,
        "notes": "Read the guide, both route files and their specs independently before filing.",
    }


def _conflict() -> dict[str, Any]:
    return {
        "kind": "cross_tenant_harm",
        "invariant": (
            "The tenant isolation policy requires every project import to be authorized "
            "against the target group the data lands in."
        ),
        "source": "doc/security/tenant_isolation_policy.md:12",
        "impact": (
            "A user-scoped token imports into a group the user is not a member of, placing "
            "one tenant's data inside another tenant's namespace."
        ),
    }


def _gate(**overrides: Any) -> intent_gate.IntentGateResult:
    kwargs: dict[str, Any] = {
        "source_aware": True,
        "finding_class": "dynamic",
        "intent_check_status": "completed",
        "intent_search_scope": _SEARCH_SCOPE,
        "intent_evidence": _evidence(),
        "known_issue_or_duplicate_search": _KNOWN_ISSUE_SEARCH,
        "alternative_semantics_check": _ALTERNATIVE_SEMANTICS,
        "assumptions": _ASSUMPTIONS,
        "intent_review": _review("intent_confirmed"),
    }
    kwargs.update(overrides)
    return intent_gate.evaluate_intent_gate(**kwargs)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_documented_intent_blocks_the_gitlab_scope_bypass() -> None:
    """Path + spec describe the behaviour as intended -> not a vulnerability."""
    result = _gate()

    assert result.applies is True
    assert result.disposition == "intended_behavior"
    assert result.blocks_filing is True
    reason = " ".join(result.errors)
    assert GITLAB_GUIDE in reason
    # The rejection has to route the agent somewhere useful, not just say no.
    assert "record_coverage" in reason
    assert "security_contract_conflict" in reason


def test_the_same_finding_files_once_a_real_conflict_is_supplied() -> None:
    """A documented-but-dangerous primitive is still reportable with a real conflict."""
    result = _gate(
        security_contract_conflict=_conflict(),
        intent_review=_review("conflict_confirmed"),
    )

    assert result.disposition == "report_conflict"
    assert result.blocks_filing is False
    assert result.evidence_summary["authoritative_support"] == [
        GITLAB_GUIDE,
        "spec/requests/api/import_bitbucket_spec.rb:118",
    ]


def test_a_conflict_cannot_cite_the_document_it_overrides() -> None:
    """Otherwise 'the doc says X' would be its own override."""
    result = _gate(
        security_contract_conflict={**_conflict(), "source": GITLAB_GUIDE},
        intent_review=_review("conflict_confirmed"),
    )

    assert result.blocks_filing is True
    assert result.disposition == "needs_follow_up"
    assert any("same document" in error for error in result.errors)


def test_a_preference_is_not_a_conflict() -> None:
    result = _gate(security_contract_conflict={"kind": "it_would_be_better", "invariant": "nope"})

    assert result.blocks_filing is True
    assert any("kind must be one of" in error for error in result.errors)


def test_the_gate_cannot_be_skipped_by_omitting_the_intent_block() -> None:
    result = intent_gate.evaluate_intent_gate(source_aware=True)

    assert result.applies is True
    assert result.blocks_filing is True
    joined = " ".join(result.errors)
    assert "intent_check_status is REQUIRED" in joined
    assert "intent_search_scope" in joined
    assert "known_issue_or_duplicate_search" in joined
    assert "alternative_semantics_check" in joined
    assert "intent_review is REQUIRED" in joined


def test_documentation_that_contradicts_the_code_is_reportable() -> None:
    """Docs and code genuinely disagree: report the conflict, not a bypass."""
    result = _gate(
        intent_evidence=[
            {
                "source": "spec/requests/api/import_github_spec.rb:64",
                "authority": "implementation_spec",
                "contradicts": "The spec requires target-group authorization; the code accepts a "
                "user boundary and the spec does not cover that path.",
            }
        ],
        intent_review=_review("conflict_confirmed"),
    )

    assert result.disposition == "report_conflict"
    assert result.blocks_filing is False


def test_unresolved_intent_is_a_follow_up_not_a_finding() -> None:
    result = _gate(intent_review=_review("unresolved"))

    assert result.disposition == "needs_follow_up"
    assert result.blocks_filing is True
    assert any("needs_follow_up" in error for error in result.errors)


def test_stale_authoritative_evidence_is_surfaced() -> None:
    result = _gate(
        intent_evidence=[{**_evidence()[0], "stale": True}],
        security_contract_conflict=_conflict(),
        intent_review=_review("conflict_confirmed"),
    )

    assert result.disposition == "needs_follow_up"
    assert result.blocks_filing is True
    assert result.evidence_summary["stale"] == [GITLAB_GUIDE]


def test_authoritative_sources_that_disagree_are_not_resolved_silently() -> None:
    result = _gate(
        intent_evidence=[
            *_evidence(),
            {
                "source": "doc/security/review_checklist.md:7",
                "authority": "security_contract",
                "contradicts": "Every import must be authorized against the destination group.",
            },
        ],
    )

    assert result.disposition == "needs_follow_up"
    assert result.blocks_filing is True
    assert any("disagree" in error for error in result.errors)


def test_weak_evidence_alone_does_not_block_but_is_flagged() -> None:
    """A comment or changelog entry cannot establish intent by itself."""
    result = _gate(
        intent_evidence=[
            {
                "source": "lib/api/import_bitbucket.rb:12",
                "authority": "comment",
                "supports": "TODO: revisit whether user boundaries should be allowed here.",
            }
        ],
        intent_review=_review("conflict_confirmed"),
    )

    assert result.blocks_filing is False
    assert result.disposition == "report_with_assumptions"
    warnings = " ".join(result.warnings)
    assert "cannot establish intent on its own" in warnings


def test_no_evidence_found_is_filed_on_labelled_assumptions() -> None:
    """Absence is not proof either way, so the search is recorded and the presumption labelled."""
    result = _gate(intent_evidence=[], intent_review=_review("conflict_confirmed"))

    assert result.blocks_filing is False
    assert result.disposition == "report_with_assumptions"
    assert any("Absence is not evidence" in warning for warning in result.warnings)


def test_a_finding_cannot_review_itself() -> None:
    result = _gate(
        caller_agent_id="root",
        intent_review=_review("conflict_confirmed", reviewed_by="root"),
    )

    assert result.blocks_filing is True
    assert any(
        "independent review must come from a different agent" in error for error in result.errors
    )


def test_a_reviewer_is_required_at_all() -> None:
    result = _gate(intent_review=None, security_contract_conflict=_conflict())

    assert result.blocks_filing is True
    assert any("intent_review is REQUIRED" in error for error in result.errors)


def test_reviewer_kind_must_be_independent() -> None:
    result = _gate(intent_review={**_review("conflict_confirmed"), "reviewer_kind": "self"})

    assert result.blocks_filing is True
    assert any("reviewer_kind" in error for error in result.errors)


def test_black_box_and_dependency_findings_keep_their_contract() -> None:
    black_box = intent_gate.evaluate_intent_gate(source_aware=False)
    dependency = intent_gate.evaluate_intent_gate(source_aware=True, finding_class="dependency_cve")

    assert black_box.disposition == "not_required"
    assert black_box.blocks_filing is False
    assert dependency.disposition == "not_required"
    assert dependency.blocks_filing is False


def test_source_awareness_signals() -> None:
    assert intent_gate.is_source_aware(local_sources=["/src/app"], code_locations=None) is True
    assert intent_gate.is_source_aware(code_locations=[{"file": "a.py"}]) is True
    assert intent_gate.is_source_aware(local_sources=[], code_locations=None) is False
    assert (
        intent_gate.is_source_aware(
            local_sources=["/src/app"], finding_class="dependency_cve"
        )
        is False
    )


def test_the_report_projection_is_none_without_intent_metadata() -> None:
    """Reports written before this gate must render exactly as they did."""
    assert intent_gate.intent_block_for_output({"title": "x", "severity": "high"}) is None


def test_the_report_projection_surfaces_the_verdict_and_its_sources() -> None:
    report = {
        "intent_check_status": "completed",
        "intent_search_scope": _SEARCH_SCOPE,
        "intent_evidence": _evidence(),
        "known_issue_or_duplicate_search": _KNOWN_ISSUE_SEARCH,
        "alternative_semantics_check": _ALTERNATIVE_SEMANTICS,
        "security_contract_conflict": _conflict(),
        "intent_review": _review("conflict_confirmed"),
        "intent_gate": {"disposition": "report_conflict", "reasons": [], "warnings": ["w"]},
    }

    block = intent_gate.intent_block_for_output(report)

    assert block is not None
    assert block["disposition"] == "report_conflict"
    assert block["evidence"][0]["source"] == GITLAB_GUIDE
    assert block["conflict"]["kind"] == "cross_tenant_harm"
    assert block["review"]["verdict"] == "conflict_confirmed"
    assert block["warnings"] == ["w"]


# --------------------------------------------------------------------------- #
# the reporting tool
# --------------------------------------------------------------------------- #


def _create_args(**overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "title": "Fine-grained PAT accepted on group-targeted project import",
        "description": "A user-boundary PAT authorizes an import into a group.",
        "impact": "Import lands in a group the token owner is not a member of.",
        "target": "http://localhost:8929",
        "technical_analysis": "The import route resolves the boundary from the token.",
        "poc_description": "1. Create a user-boundary PAT. 2. POST the import endpoint.",
        "poc_script_code": "curl -X POST /api/v4/projects/import -H 'PRIVATE-TOKEN: <pat>'",
        "remediation_steps": "Require destination-group authorization.",
        "evidence": f"{GITLAB_ROUTE} resolves the boundary without a group check.",
        "assumptions": _ASSUMPTIONS,
        "counterevidence": f"{GITLAB_GUIDE} documents the OR boundary semantics.",
        "confidence": "medium",
        "confidence_rationale": "Static trace against the cloned repository.",
        "severity_change_conditions": "A cross-tenant write would raise this to critical.",
        "fix_effort": "medium",
        "cvss_breakdown": _CVSS,
        "code_locations": [{"file": "lib/api/import_bitbucket.rb", "line": 41}],
        "endpoint": "/api/v4/projects/import",
        "method": "POST",
        "cve": None,
        "cwe": "CWE-863",
    }
    args.update(overrides)
    return args


@pytest.fixture
def report_state(tmp_path: Path) -> ReportState:
    state = ReportState("intent-gate")
    state.run_dir = tmp_path
    # A white-box run: the scan mounted a source tree, so filings are source-aware.
    state.run_record["local_sources"] = [{"source_path": "/src/gitlab"}]
    return state


def _file(state: ReportState, **overrides: Any) -> dict[str, Any]:
    """Run the create path; it returns the result dict the tool then serialises."""
    with patch("strix.report.state.get_global_report_state", return_value=state):
        return asyncio.run(_do_create(**_create_args(**overrides)))


def test_tool_refuses_the_gitlab_finding_without_a_real_conflict(report_state: ReportState) -> None:
    """The whole point: the report must not be appended."""
    payload = _file(
        report_state,
        intent_check_status="completed",
        intent_search_scope=_SEARCH_SCOPE,
        intent_evidence=_evidence(),
        known_issue_or_duplicate_search=_KNOWN_ISSUE_SEARCH,
        alternative_semantics_check=_ALTERNATIVE_SEMANTICS,
        intent_review=_review("intent_confirmed"),
    )

    assert payload["success"] is False
    assert payload["intent_gate"]["disposition"] == "intended_behavior"
    assert GITLAB_GUIDE in " ".join(payload["errors"])
    assert report_state.vulnerability_reports == []


def test_tool_refuses_a_source_aware_finding_with_no_intent_block_at_all(
    report_state: ReportState,
) -> None:
    payload = _file(report_state)

    assert payload["success"] is False
    assert any("intent_check_status is REQUIRED" in error for error in payload["errors"])
    assert report_state.vulnerability_reports == []


def test_tool_files_the_finding_when_a_conflict_is_demonstrated(report_state: ReportState) -> None:
    payload = _file(
        report_state,
        intent_check_status="completed",
        intent_search_scope=_SEARCH_SCOPE,
        intent_evidence=_evidence(),
        known_issue_or_duplicate_search=_KNOWN_ISSUE_SEARCH,
        alternative_semantics_check=_ALTERNATIVE_SEMANTICS,
        security_contract_conflict=_conflict(),
        intent_review=_review("conflict_confirmed"),
    )

    assert payload["success"] is True, payload
    stored = report_state.vulnerability_reports[0]
    assert stored["intent_gate"]["disposition"] == "report_conflict"
    assert stored["security_contract_conflict"]["kind"] == "cross_tenant_harm"
    assert stored["intent_evidence"][0]["source"] == GITLAB_GUIDE
    assert stored["intent_review"]["verdict"] == "conflict_confirmed"


def test_tool_leaves_black_box_findings_alone(tmp_path: Path) -> None:
    """No local sources and no code locations: the previous contract holds."""
    state = ReportState("black-box")
    state.run_dir = tmp_path
    state.run_record["local_sources"] = []

    payload = _file(state, code_locations=None)

    assert payload["success"] is True, payload
    assert "intent_gate" not in payload.get("errors", [])
    assert state.vulnerability_reports[0]["intent_gate"]["disposition"] == "not_required"


def test_tool_requires_intent_for_a_finding_that_points_at_code(tmp_path: Path) -> None:
    """A dynamic-only scan that files with code_locations becomes source-aware."""
    state = ReportState("dynamic-with-source")
    state.run_dir = tmp_path
    state.run_record["local_sources"] = []

    payload = _file(state)  # _create_args carries code_locations

    assert payload["success"] is False
    assert state.vulnerability_reports == []


def test_dependency_findings_are_exempt_from_the_intent_gate(report_state: ReportState) -> None:
    """A CVE's validity comes from the advisory, not from this repo's design intent."""
    gate = _filing_intent_gate(
        report_state,
        finding_class="dependency_cve",
        code_locations=[{"file": "package-lock.json"}],
        intent_check_status=None,
        intent_search_scope=None,
        intent_evidence=None,
        known_issue_or_duplicate_search=None,
        alternative_semantics_check=None,
        security_contract_conflict=None,
        intent_review=None,
        assumptions=None,
        caller_agent_id="worker-1",
    )

    assert gate.disposition == "not_required"
    assert gate.blocks_filing is False
