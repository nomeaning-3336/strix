"""Finding lifecycle states: CANDIDATE/VERIFIED/RETRACTED/REJECTED/PROOF_GAP.

Regression fixtures:

  CASE A — false security premise. A claimed fog-of-war bypass where the target
  has no visibility boundary: rejected/retracted, and the finding must stop
  counting and stop being rendered as actionable. The openfrontio_94a1 record
  (title prefixed ``[RETRACTED — NON-ISSUE]``, ``retracted: true``, a stale PoC
  and remediation in the body, and ``finish_scan`` still counting it) is the
  canonical example.

  CASE B — incomplete exploitation evidence. A real invariant discrepancy with
  no demonstrated impact is recorded as an open proof gap, never promoted to
  verified.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from strix.report import writer
from strix.report.finding_state import (
    INACTIVE_STATES,
    active_reports,
    is_active,
    state_of,
)
from strix.report.sarif import build_sarif_report
from strix.report.state import ReportState, set_global_report_state
from strix.telemetry import posthog, scarf
from strix.tools.reporting.tool import _do_set_finding_state


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def report_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReportState:
    monkeypatch.chdir(tmp_path)
    state = ReportState(run_name="test-run")
    set_global_report_state(state)
    return state


def _fog_record(*, title: str = "[RETRACTED — NON-ISSUE] Fog-of-war bypass") -> dict[str, Any]:
    """The openfrontio_94a1 shape: an ad-hoc legacy retraction, no `state` key."""
    return {
        "id": "vuln-0001",
        "title": title,
        "severity": "info",
        "timestamp": "2026-09-02 10:20:35 UTC",
        "description": "Retracted: OpenFront has no fog-of-war boundary.",
        "impact": "None — every honest client receives the same state.",
        "target": "/workspace/OpenFrontIO",
        "poc_description": "1. Join a local multiplayer game so fog hides one player...",
        "remediation_steps": "Enforce fog-of-war visibility at the broadcast point...",
        "cwe": "CWE-201",
        "finding_class": "dynamic",
        "retracted": True,
        "classification": "NON-ISSUE",
        "retraction_reason": "No fog-of-war or per-client visibility restriction exists.",
    }


def _active_finding(**overrides: Any) -> dict[str, Any]:
    record = {
        "id": "vuln-0002",
        "title": "Reflected XSS in search",
        "severity": "high",
        "timestamp": "2026-09-02 10:30:00 UTC",
        "description": "q reflects unencoded input.",
        "poc_description": "1. open /search?q=<payload>",
        "remediation_steps": "Context-encode output.",
        "endpoint": "/search",
        "method": "GET",
        "cwe": "CWE-79",
        "finding_class": "dynamic",
    }
    record.update(overrides)
    return record


# --- state mapping & legacy records ------------------------------------------


def test_legacy_ad_hoc_retraction_maps_to_retracted() -> None:
    assert state_of(_fog_record()) == "retracted"
    # A `retracted: true` flag alone also maps.
    assert state_of(_active_finding(retracted=True)) == "retracted"
    # No marker at all: a pre-state record was filed as verified evidence.
    assert state_of(_active_finding()) == "verified"


def test_state_field_wins_over_legacy_markers() -> None:
    record = _fog_record()
    record["state"] = "verified"
    assert state_of(record) == "verified"


def test_inactive_states_do_not_count() -> None:
    records = [
        _active_finding(),  # verified
        _fog_record(),  # legacy retracted
        _active_finding(id="vuln-0003", title="C2", state="rejected"),
        _active_finding(id="vuln-0004", title="Gap", state="proof_gap"),
        _active_finding(id="vuln-0005", title="C3", state="candidate"),
    ]
    active = active_reports(records)
    assert [r["id"] for r in active] == ["vuln-0002", "vuln-0005"]
    assert [r["id"] for r in records if is_active(r)] == ["vuln-0002", "vuln-0005"]
    for record in records:
        assert state_of(record) in ("verified", "candidate", "retracted", "rejected", "proof_gap")
    assert {"retracted", "rejected", "proof_gap"} == INACTIVE_STATES


# --- transitions --------------------------------------------------------------


def test_filed_findings_default_to_verified(report_state: ReportState) -> None:
    report_id = report_state.add_vulnerability_report(title="XSS", severity="high")
    assert report_state.vulnerability_reports[0]["state"] == "verified"
    assert report_id == "vuln-0001"


def test_filed_finding_can_start_as_candidate(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="Suspected", severity="low", state="candidate")
    assert report_state.vulnerability_reports[0]["state"] == "candidate"
    assert is_active(report_state.vulnerability_reports[0])


def test_retraction_records_reason_and_actor(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(
        title="Fog-of-war bypass", severity="medium", state="verified"
    )
    updated = report_state.set_finding_state(
        "vuln-0001",
        "retracted",
        reason="No fog-of-war boundary exists; full state is broadcast by design.",
        changed_by_agent_id="root-1",
        changed_by_agent_name="Root Agent",
    )
    assert updated is not None
    report = report_state.vulnerability_reports[0]
    assert report["state"] == "retracted"
    assert report["state_reason"].startswith("No fog-of-war boundary")
    assert report["state_changed_by_agent_name"] == "Root Agent"
    assert report["state_changed_at"]
    history = report["update_history"][-1]
    assert history["previous_state"] == "verified"
    assert history["state"] == "retracted"
    assert history["agent_id"] == "root-1"
    assert not is_active(report)
    assert len(report_state.get_active_vulnerabilities()) == 0
    assert report_state.get_state_counts()["retracted"] == 1


def test_verified_cannot_jump_directly_to_rejected(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="X", severity="medium")
    with pytest.raises(ValueError, match="Cannot move report vuln-0001 from verified to rejected"):
        report_state.set_finding_state("vuln-0001", "rejected", reason="nope")


def test_candidate_rejection_is_case_a(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(
        title="Fog-of-war bypass", severity="medium", state="candidate"
    )
    updated = report_state.set_finding_state(
        "vuln-0001", "rejected", reason="No such invariant: NON-ISSUE"
    )
    assert updated is not None
    report = report_state.vulnerability_reports[0]
    assert report["state"] == "rejected"
    assert not is_active(report)


def test_case_b_proof_gap_is_recorded_not_verified(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(
        title="Cross-map attack adjacency", severity="medium", state="verified"
    )
    updated = report_state.set_finding_state(
        "vuln-0001",
        "proof_gap",
        reason=(
            "UI checks sharesBorderWith but AttackExecution does not enforce "
            "adjacency; no exploitation impact was demonstrated."
        ),
    )
    assert updated is not None
    report = report_state.vulnerability_reports[0]
    assert report["state"] == "proof_gap"
    assert not is_active(report)
    assert len(report_state.get_active_vulnerabilities()) == 0


def test_reopen_requires_reason_and_allows_verified(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="X", severity="high")
    report_state.set_finding_state("vuln-0001", "retracted", reason="was false")
    updated = report_state.set_finding_state(
        "vuln-0001", "verified", reason="new discriminating PoC demonstrates impact"
    )
    assert updated is not None
    assert is_active(report_state.vulnerability_reports[0])
    # Still guarded: reason stays mandatory for any change.
    with pytest.raises(ValueError, match="needs a reason"):
        report_state.set_finding_state("vuln-0001", "candidate", reason="")


def test_noop_and_unknown_and_invalid(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="X", severity="low", state="verified")
    assert report_state.set_finding_state("vuln-0001", "verified", reason="already") is None
    assert report_state.set_finding_state("vuln-9999", "retracted", reason="r") is None
    with pytest.raises(ValueError, match="Invalid finding state"):
        report_state.set_finding_state("vuln-0001", "maybe", reason="r")


# --- set_finding_state agent tool ---------------------------------------------


def test_set_finding_state_tool_round_trip(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="XSS", severity="high")
    result = _do_set_finding_state(
        report_id="vuln-0001",
        new_state="retracted",
        reason="Never verified: payload was not reflected.",
        agent_id="agent-1",
        agent_name="Worker",
    )
    assert result["success"] is True
    assert result["action"] == "state_changed"
    assert result["state"] == "retracted"
    assert result["previous_state"] == "verified"
    assert result["state_reason"] == "Never verified: payload was not reflected."
    report = report_state.vulnerability_reports[0]
    assert report["state_changed_by_agent_id"] == "agent-1"


def test_set_finding_state_tool_validates(report_state: ReportState) -> None:
    report_state.add_vulnerability_report(title="X", severity="low")

    bad_state = _do_set_finding_state(report_id="vuln-0001", new_state="bogus", reason="r")
    assert bad_state["success"] is False
    assert any("Invalid state" in e for e in bad_state["errors"])

    missing = _do_set_finding_state(report_id="vuln-0001", new_state="retracted", reason="")
    assert missing["success"] is False
    assert any("reason is required" in e for e in missing["errors"])

    unknown = _do_set_finding_state(report_id="vuln-0099", new_state="retracted", reason="r")
    assert unknown["success"] is False
    assert "not found" in unknown["error"]

    disallowed = _do_set_finding_state(report_id="vuln-0001", new_state="rejected", reason="wrong")
    assert disallowed["success"] is False
    assert "Cannot move" in disallowed["error"]


# --- CASE A fixture end-to-end through the artifact sinks ----------------------


def test_94a1_fog_fixture_excluded_from_csv_sarif_but_retained(report_state: ReportState) -> None:
    fog = _fog_record()  # legacy ad-hoc retraction, exactly the 94a1 shape
    live = _active_finding()
    report_state.vulnerability_reports = [fog, live]
    report_state._saved_vuln_ids.clear()

    run_dir = report_state.get_run_dir()
    report_state.save_run_data()

    # Evidence is retained in the canonical JSON store for audit...
    on_disk = json.loads((run_dir / "vulnerabilities.json").read_text(encoding="utf-8"))
    assert len(on_disk) == 2
    assert on_disk[0]["id"] == "vuln-0001"
    assert "retraction_reason" in on_disk[0]

    # ...but the CSV index lists only the active finding...
    csv_text = (run_dir / "vulnerabilities.csv").read_text(encoding="utf-8")
    assert "vuln-0001" not in csv_text
    assert "vuln-0002" in csv_text

    # ...and SARIF carries only the active finding as a result.
    sarif = build_sarif_report(report_state.vulnerability_reports, coverage=None)
    results = sarif["runs"][0]["results"]
    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    assert len(results) == 1
    assert [rule["id"] for rule in rules] == ["CWE-79"]

    # The retracted finding's own markdown frames its archived material.
    md_text = (run_dir / "vulnerabilities" / "vuln-0001.md").read_text(encoding="utf-8")
    assert "RETRACTED — NOT A VULNERABILITY" in md_text
    assert "No fog-of-war or per-client visibility restriction exists." in md_text
    # The ad-hoc legacy markers still mean retracted to every consumer.
    assert state_of(fog) == "retracted"


def test_retracted_finding_markdown_keeps_evidence_but_banners_it(
    report_state: ReportState,
) -> None:
    live = _active_finding()
    report_state.vulnerability_reports = [live]
    state = report_state.set_finding_state(
        "vuln-0002",
        "retracted",
        reason="Response encoding was present after all.",
        changed_by_agent_id="root-1",
        changed_by_agent_name="Root Agent",
    )
    assert state is not None
    md = writer.render_vulnerability_md(state)
    assert "RETRACTED — NOT A VULNERABILITY" in md
    assert "Response encoding was present after all." in md
    assert "Recorded by:** Root Agent" in md
    # Evidence is retained (archived), not deleted.
    assert "q reflects unencoded input." in md
    assert "Context-encode output." in md


def test_active_finding_markdown_has_no_banner() -> None:
    md = writer.render_vulnerability_md(_active_finding())
    assert "RETRACTED" not in md
    assert "REJECTED" not in md
    assert "PROOF GAP" not in md


# --- counts: telemetry + finish -------------------------------------------------


def _capture_send(sent: list[dict[str, Any]]) -> Any:
    def _record(event: str, properties: dict[str, Any]) -> bool:
        sent.append({"event": event, **properties})
        return True

    return _record


def test_telemetry_end_counts_only_active(
    report_state: ReportState, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_state.vulnerability_reports = [_active_finding(), _fog_record()]
    monkeypatch.setattr(report_state, "posthog_scan_ended_sent", False)
    monkeypatch.setattr(report_state, "scarf_scan_ended_sent", False)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(posthog, "_send", _capture_send(sent))
    monkeypatch.setattr(scarf, "_send", _capture_send(sent))
    report_state.scan_ended_exit_reason = None

    posthog.end(report_state, exit_reason="finished_by_tool")
    scarf.end(report_state, exit_reason="finished_by_tool")

    totals = [s["vulnerabilities_total"] for s in sent if s["event"] == "scan_ended"]
    assert totals == [1, 1]
    highs = [s["vulnerabilities_high"] for s in sent if s["event"] == "scan_ended"]
    assert highs == [1, 1]
