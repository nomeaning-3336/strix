"""Finding lifecycle states.

A finding is not simply "on the report" or "not": it moves through states, and
every artifact that counts, renders, or bills a finding must agree on which
states count as a vulnerability.

    candidate  → filed, not yet confirmed
    verified   → confirmed (the default for a filed, fully-evidenced finding)
    retracted  → was counted, later found false
    rejected   → deliberately not promoted: the prerequisite does not hold
    proof_gap  → a real invariant discrepancy, but no demonstrated impact

Only ``candidate`` and ``verified`` count. A retracted or rejected finding

  - is excluded from vulnerability totals (finish, telemetry, run summaries);
  - is excluded from customer-facing artifacts (SARIF, CSV index);
  - keeps its original evidence, PoC, and remediation internally for audit,
    with the retraction/rejection recorded against it;
  - is never rendered as actionable guidance (its per-finding markdown carries
    a banner and frames the archived PoC/remediation as superseded).

Two canonical regression cases motivate the state set:

  CASE A — false security premise. A claimed "bypass" of a mechanic that the
  target does not implement (e.g. a fog-of-war visibility boundary that does
  not exist). Correct outcome: REJECTED / NON-ISSUE, not a verified finding
  with a downgraded severity. A finding filed before this was established is
  RETRACTED, with the pre-existing evidence kept but no longer offered as a
  Proof of Concept.

  CASE B — incomplete exploitation evidence. A real invariant discrepancy
  exists but no demonstrated impact (e.g. the UI checks adjacency while the
  authoritative execution path does not). Correct outcome: OPEN PROOF GAP —
  recorded as unresolved, never promoted to VERIFIED.

This module is deliberately dependency-free: `report.state` imports
`report.writer`, so the shared state model lives here where every consumer
(state, writer, sarif, telemetry, interface, tools) can import it without a
cycle.
"""

from __future__ import annotations

from typing import Any


CANDIDATE = "candidate"
VERIFIED = "verified"
RETRACTED = "retracted"
REJECTED = "rejected"
PROOF_GAP = "proof_gap"

FINDING_STATES = frozenset({CANDIDATE, VERIFIED, RETRACTED, REJECTED, PROOF_GAP})

# States that stop counting as vulnerabilities and stop appearing in
# customer-facing artifacts.
INACTIVE_STATES = frozenset({RETRACTED, REJECTED, PROOF_GAP})

# Transitions that make sense in the lifecycle. A filing is verified by
# default; a candidate is promoted or rejected; a verified finding found false
# later is retracted (never silently edited down); either inactive state can be
# reopened, but only with a documented reason.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    CANDIDATE: frozenset({VERIFIED, REJECTED, PROOF_GAP}),
    VERIFIED: frozenset({CANDIDATE, RETRACTED, PROOF_GAP}),
    RETRACTED: frozenset({VERIFIED}),
    REJECTED: frozenset({CANDIDATE, VERIFIED}),
    PROOF_GAP: frozenset({CANDIDATE, VERIFIED}),
}

# Legacy records used ad-hoc markers instead of a state: a boolean `retracted`
# flag and/or a "[RETRACTED ...]" title prefix. Both still mean retracted.
_LEGACY_RETRACTED_PREFIX = "[RETRACTED"

# Field carrying why a finding left the active set, and who decided.
STATE_REASON_FIELD = "state_reason"


def normalize_state(value: Any) -> str | None:
    """Return the canonical state for ``value``, or None when not a state."""
    if not isinstance(value, str):
        return None
    state = value.strip().lower()
    return state if state in FINDING_STATES else None


def state_of(report: dict[str, Any]) -> str:
    """The lifecycle state of a report, mapping legacy ad-hoc markers."""
    state = normalize_state(report.get("state"))
    if state is not None:
        return state
    if report.get("retracted") is True:
        return RETRACTED
    title = report.get("title")
    if isinstance(title, str) and title.lstrip().upper().startswith(_LEGACY_RETRACTED_PREFIX):
        return RETRACTED
    # A report written before states existed was filed as verified evidence.
    return VERIFIED


def is_active(report: dict[str, Any]) -> bool:
    """True when the finding counts as a vulnerability right now."""
    return state_of(report) not in INACTIVE_STATES


def active_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [report for report in reports if is_active(report)]


def transition_allowed(current: str, new_state: str) -> bool:
    return new_state in ALLOWED_TRANSITIONS.get(current, frozenset())


def state_reason(report: dict[str, Any]) -> str:
    """Why the finding is in its current state ("" for filed-as-verified)."""
    reason = report.get(STATE_REASON_FIELD)
    if isinstance(reason, str):
        return reason.strip()
    # Legacy retraction payloads used dedicated keys.
    if state_of(report) in INACTIVE_STATES:
        for key in ("retraction_reason", "rejection_reason", "classification"):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""
