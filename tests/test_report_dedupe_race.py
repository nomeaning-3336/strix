"""Tests for the concurrent report-dedupe race fix (#633).

The LLM dedupe check compares a candidate against the snapshot of reports taken
when filing began. Two agents filing the same finding concurrently can both pass
that check against the same stale snapshot and both land. The fix adds a
deterministic fingerprint (:func:`finding_fingerprint`) plus a locked
check-then-append commit in ``ReportState``: only reports that landed *after* the
caller's snapshot are compared, at commit time, under one lock — no second LLM
call, no race window between check and append.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any

import pytest

from strix.report.dedupe import finding_fingerprint
from strix.report.state import DuplicateVulnerabilityError, ReportState, set_global_report_state
from strix.telemetry import posthog, scarf
from strix.tools.reporting.tool import _do_create


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


def _finding_kwargs(
    *, endpoint: str, title: str = "Reflected XSS in search", **overrides: Any
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "title": title,
        "severity": "high",
        "description": "q reflects unencoded input.",
        "impact": "Session theft.",
        "target": "https://app.example.com",
        "endpoint": endpoint,
        "method": "GET",
        "cwe": "CWE-79",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def report_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportState:
    monkeypatch.chdir(tmp_path)
    state = ReportState(run_name="test-run")
    set_global_report_state(state)
    return state


# --- fingerprint semantics --------------------------------------------------


def test_fingerprint_normalises_reworded_duplicates_identically() -> None:
    a = finding_fingerprint(
        {
            "title": "SQL injection in api login",
            "target": "https://app.example.com",
            "endpoint": "/api/login",
            "method": "POST",
            "cwe": "cwe-89",
        }
    )
    b = finding_fingerprint(
        {
            # same title-word multiset, different order and punctuation
            "title": "api login: SQL injection in",
            "target": "HTTPS://APP.EXAMPLE.COM",
            "endpoint": "/Api/Login",
            "method": "post",
            "cwe": "89",
        }
    )
    assert a is not None
    assert a == b


def test_fingerprint_keeps_distinct_findings_distinct() -> None:
    base = {"title": "Reflected XSS in search", "target": "https://app.example.com"}
    endpoint = finding_fingerprint(
        {**base, "endpoint": "/api/login", "method": "GET", "cwe": "CWE-79"}
    )
    assert endpoint != finding_fingerprint(
        {**base, "endpoint": "/api/search", "method": "GET", "cwe": "CWE-79"}
    )
    assert endpoint != finding_fingerprint(
        {**base, "endpoint": "/api/login", "method": "POST", "cwe": "CWE-79"}
    )
    assert endpoint != finding_fingerprint(
        {**base, "endpoint": "/api/login", "method": "GET", "cwe": "CWE-89"}
    )
    assert endpoint != finding_fingerprint(
        {**base, "endpoint": "/api/login", "method": "GET", "cwe": "CWE-79", "target": "https://other.example.com"}
    )


def test_fingerprint_uses_code_location_when_no_endpoint() -> None:
    a = finding_fingerprint(
        {
            "title": "SQL injection in user lookup",
            "finding_class": "static",
            "target": "repo/src",
            "code_locations": [{"file": "src/Routes/Login.ts", "start_line": 14}],
            "cwe": "CWE-89",
        }
    )
    b = finding_fingerprint(
        {
            "title": "SQL injection in user lookup",
            "finding_class": "static",
            "target": "repo/src",
            # line drift between two concurrent filings must not break identity
            "code_locations": [{"file": "src/routes/login.ts", "start_line": 99}],
            "cwe": "CWE-89",
        }
    )
    assert a is not None
    assert a == b
    assert a != finding_fingerprint(
        {
            "title": "SQL injection in user lookup",
            "finding_class": "static",
            "target": "repo/src",
            "code_locations": [{"file": "src/routes/Admin.ts"}],
            "cwe": "CWE-89",
        }
    )


def test_fingerprint_is_none_without_identity_fields() -> None:
    assert finding_fingerprint({"finding_class": "dynamic"}) is None
    assert finding_fingerprint({}) is None


def test_fingerprint_dependency_identity() -> None:
    def dep(*, manifest: str | None = None, cve: str = "CVE-2024-1234") -> dict[str, Any]:
        return {
            "title": "Arbitrary title wording",
            "cve": cve,
            "finding_class": "dependency_cve",
            "dependency_metadata": {
                "package_name": "Lodash",
                "package_ecosystem": "npm",
                "manifest_path": manifest,
            },
        }

    same = finding_fingerprint(dep())
    assert same == finding_fingerprint(
        {
            "cve": "cve-2024-1234",
            "dependency_metadata": {"package_name": "lodash", "package_ecosystem": "NPM"},
        }
    )
    assert same != finding_fingerprint(dep(cve="CVE-2025-0001"))
    # Same CVE/package in a distinct manifest is a distinct finding shape
    # (mirrors _check_dependency_duplicate's distinct-manifest rule); both
    # sides of a concurrent filing always carry the same manifest, so exact
    # equality on the full key is unambiguous within the race window.
    with_manifest = finding_fingerprint(dep(manifest="package-lock.json"))
    assert with_manifest is not None
    assert same != with_manifest


# --- locked check-then-append race ------------------------------------------


def test_concurrent_same_finding_single_winner(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_state.add_vulnerability_report(title="seed", severity="low")  # vuln-0001
    snapshot = frozenset({"vuln-0001"})
    kwargs = _finding_kwargs(endpoint="/api/login")
    fingerprint = finding_fingerprint(kwargs)
    assert fingerprint is not None

    sent: list[str] = []

    def record_finding(*_args: Any, **_kwargs: Any) -> None:
        sent.append("finding")

    monkeypatch.setattr(posthog, "finding", record_finding)
    monkeypatch.setattr(scarf, "finding", record_finding)

    def file_one(_index: int) -> str:
        try:
            return report_state.add_vulnerability_report(
                **_finding_kwargs(endpoint="/api/login"),
                _duplicate_guard=(snapshot, fingerprint),
            )
        except DuplicateVulnerabilityError as exc:
            return f"dup:{exc.duplicate_id}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(file_one, range(8)))

    winners = [o for o in outcomes if o.startswith("vuln-")]
    losers = [o for o in outcomes if o.startswith("dup:")]
    assert len(winners) == 1
    assert len(losers) == 7
    assert all(loser == f"dup:{winners[0]}" for loser in losers)
    assert len(report_state.vulnerability_reports) == 2
    # telemetry fired exactly once: the losing filings never reach append
    assert len(sent) == 2


def test_concurrent_distinct_findings_all_land(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="seed", severity="low")  # vuln-0001
    snapshot = frozenset({"vuln-0001"})

    def file_one(index: int) -> str:
        kwargs = _finding_kwargs(endpoint=f"/api/endpoint-{index}")
        fingerprint = finding_fingerprint(kwargs)
        assert fingerprint is not None
        return report_state.add_vulnerability_report(
            **kwargs, _duplicate_guard=(snapshot, fingerprint)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(file_one, range(8)))

    assert all(o.startswith("vuln-") for o in outcomes)
    assert len(set(outcomes)) == 8  # distinct findings, distinct ids, no collisions
    assert len(report_state.vulnerability_reports) == 9


def test_guard_rejects_finding_filed_since_snapshot(report_state: ReportState) -> None:
    kwargs = _finding_kwargs(endpoint="/api/login")
    fingerprint = finding_fingerprint(kwargs)
    assert fingerprint is not None
    report_state.add_vulnerability_report(**kwargs)  # vuln-0001, filed before our snapshot
    with pytest.raises(DuplicateVulnerabilityError) as excinfo:
        report_state.add_vulnerability_report(
            **kwargs, _duplicate_guard=(frozenset(), fingerprint)
        )
    assert excinfo.value.duplicate_id == "vuln-0001"
    assert len(report_state.vulnerability_reports) == 1


def test_guard_passes_when_no_concurrent_report_matches(report_state: ReportState) -> None:
    kwargs = _finding_kwargs(endpoint="/api/login")
    fingerprint = finding_fingerprint(kwargs)
    report_state.add_vulnerability_report(title="seed", severity="low")  # vuln-0001
    report_id = report_state.add_vulnerability_report(
        **kwargs, _duplicate_guard=(frozenset({"vuln-0001"}), fingerprint)
    )
    assert report_id == "vuln-0002"


def test_guard_rejects_with_known_ids_missing_one(report_state: ReportState) -> None:
    """A snapshot that predates *one* concurrent sibling still catches it."""
    first = _finding_kwargs(endpoint="/api/login")
    fingerprint = finding_fingerprint(first)
    report_state.add_vulnerability_report(**first)  # vuln-0001
    # Snapshot was taken when nothing existed; sibling vuln-0002 also landed.
    report_state.add_vulnerability_report(**_finding_kwargs(endpoint="/other"))
    with pytest.raises(DuplicateVulnerabilityError):
        report_state.add_vulnerability_report(
            **first, _duplicate_guard=(frozenset(), fingerprint)
        )


# --- end-to-end create path --------------------------------------------------


async def test_duplicate_creates_race_single_winner(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent _do_create filings of the same finding: one wins, the
    other is told which report already won — no LLM re-ask, no stale-snapshot
    double-land."""
    calls = 0

    async def fake_check_duplicate(
        _candidate: dict[str, Any],
        _existing: list[dict[str, Any]],
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        # Yield so the sibling filing snapshots the (still empty) report list
        # before this one commits — that is the stale-snapshot race under test.
        await asyncio.sleep(0)
        return {"is_duplicate": False, "duplicate_id": "", "confidence": 1.0, "reason": "stub"}

    monkeypatch.setattr("strix.report.dedupe.check_duplicate", fake_check_duplicate)

    def make_kwargs() -> dict[str, Any]:
        return {
            "title": "Reflected XSS in search",
            "description": "q reflects unencoded input.",
            "impact": "Session theft.",
            "target": "https://app.example.com",
            "technical_analysis": "Input interpolated into HTML.",
            "poc_description": "1. open /search?q=<payload>",
            "poc_script_code": "GET /search?q=<script>alert(1)</script>",
            "remediation_steps": "Context-encode output.",
            "evidence": "Response echoes the payload verbatim.",
            "assumptions": "Assumes a victim opens a crafted link.",
            "counterevidence": "No output encoding or CSP observed on this response.",
            "confidence": "high",
            "severity_change_conditions": "A strict CSP would lower the severity.",
            "fix_effort": "low",
            "cvss_breakdown": _CVSS,
            "endpoint": "/api/login",
            "method": "GET",
            "cve": None,
            "cwe": "CWE-79",
            "code_locations": None,
        }

    first, second = await asyncio.gather(
        _do_create(**make_kwargs()),
        _do_create(**make_kwargs()),
    )

    assert calls == 2  # one LLM-dedupe snapshot per filing, no re-ask for the loser
    successes = [r for r in (first, second) if r.get("success") is True]
    duplicates = [r for r in (first, second) if r.get("success") is False]
    assert len(successes) == 1
    assert len(duplicates) == 1
    assert duplicates[0]["duplicate_of"] == successes[0]["report_id"]
    assert len(report_state.vulnerability_reports) == 1
