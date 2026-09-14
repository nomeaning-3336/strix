"""Developer-intent gate for source-aware findings.

Why this exists
---------------
A finding is not a vulnerability merely because the source-level behaviour looks
looser than the reviewer would have designed it. Repository evidence frequently
says the behaviour is the intended contract: an authorization doc that names the
accepted boundary alternatives, a request spec that asserts them, a commit that
introduced the code together with both. Filing those as vulnerabilities is a
false positive that costs triage time and credibility — and it is worse when the
agent *cited* that evidence in its own counterevidence and filed anyway.

The regression that motivated this gate: a user-boundary fine-grained PAT was
accepted on group-targeted project imports and filed as a scope bypass, while the
cloned repo already documented (``doc/development/permissions/granular_access/
rest_api_implementation_guide.md``) that import endpoints may resolve a group *or*
a user boundary and that access is granted when any one resolved boundary is
satisfied — exactly what the route files declared and their specs asserted. That
was not a retrieval failure; it was a missing intent gate.

Authority hierarchy (highest first)
-----------------------------------
``security_contract``   published security policy, threat model, an explicit
                        security promise the product makes to users or customers.
``implementation_spec`` the exact semantics of this route/API/permission/flag:
                        authorization guides, API references, request specs,
                        integration specs, schema definitions.
``operator_doc``        how to run or configure the feature; describes deployment.
``history``             changelog, ADR, the commit/blame that introduced the
                        behaviour together with its docs or tests.
``comment``             code comments, TODOs, review discussion.

Rebuttable presumption
----------------------
When ``security_contract`` or ``implementation_spec`` evidence *supports* the
observed behaviour and nothing authoritative contradicts it, the default
disposition is ``intended_behavior`` and the finding is rejected. It may be filed
only with a ``security_contract_conflict`` that names a concrete conflicting
security invariant, an external security promise, cross-tenant harm, or an
exploitable boundary documentation cannot waive — anchored to a source *other
than* the document it overrides. "A stricter model would be better" is not a
conflict and cannot pass this gate.

Weaker evidence is rebuttable by default: an operator doc, changelog or comment
that supports the behaviour is recorded and raised as a warning (the finding must
carry labelled assumptions), but it does not block a filing on its own. Stale or
self-contradictory authoritative evidence is surfaced and routed to
``needs_follow_up`` instead of being resolved silently in the agent's favour.

Undocumented behaviour is not intent
------------------------------------
Documentation must never sanitize genuinely insecure behaviour. Evidence is
consulted per observation: an entry only counts as supporting intent when it is
about the behaviour actually observed, and the gate records the searches that
found nothing so a later reader can tell "checked and undocumented" from "not
checked". Absence is not proof in either direction.

The gate lives at the reporting-tool/schema layer, so it cannot be skipped by an
agent that simply forgets to write about intent in prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


# Authority classes, strongest first. The order is the rebuttable-presumption
# rule made explicit: policy and exact specs outweigh how-to docs, history and
# comments, so a generic comment can never quietly outrank a security contract.
AUTHORITY_ORDER: tuple[str, ...] = (
    "security_contract",
    "implementation_spec",
    "operator_doc",
    "history",
    "comment",
)

# Evidence that can establish intent on its own — and therefore block a filing
# unless a real conflict is supplied.
AUTHORITATIVE_AUTHORITIES: frozenset[str] = frozenset({"security_contract", "implementation_spec"})

INTENT_CHECK_STATUSES: tuple[str, ...] = ("completed", "unavailable")

# What an override has to be. Each kind is a claim a reviewer can check, which is
# what makes "I would have designed it differently" structurally insufficient.
CONFLICT_KINDS: tuple[str, ...] = (
    "conflicting_security_invariant",
    "external_security_promise",
    "cross_tenant_harm",
    "exploitable_boundary_not_waivable",
)

REVIEW_VERDICTS: tuple[str, ...] = (
    "intent_confirmed",
    "conflict_confirmed",
    "unresolved",
)

DISPOSITIONS: tuple[str, ...] = (
    "not_required",
    "report",
    "report_conflict",
    "report_with_assumptions",
    "intended_behavior",
    "needs_follow_up",
)

# A finding class whose validity is defined by an advisory rather than by this
# repository's design intent. Reachability of a dependency is still reviewed (the
# dependency path has its own reachability checks), but "the maintainers meant to
# do this" is not the right question for a CVE.
_EXEMPT_FINDING_CLASSES: frozenset[str] = frozenset({"dependency_cve"})

# Minimum number of recorded search locations. The gate cannot judge whether a
# search was *good*; it can refuse to accept a placeholder.
_MIN_SCOPE_ENTRIES_COMPLETED = 3
_MIN_SCOPE_ENTRIES_UNAVAILABLE = 2
_MIN_CONFLICT_TEXT = 40
_MIN_REVIEW_NOTES = 20


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def is_source_aware(
    *,
    local_sources: Any = None,
    code_locations: Any = None,
    finding_class: str | None = None,
) -> bool:
    """Whether this finding rests on the target's own source.

    Two independent signals, both observable at filing time: the run mounted a
    local source tree (white-box scan), or the finding itself points at code.
    A dynamic-only scan that never saw source is exempt, and stays exempt.
    """
    if (finding_class or "").strip().lower() in _EXEMPT_FINDING_CLASSES:
        return False
    if _as_list(code_locations):
        return True
    return bool(_as_list(local_sources))


@dataclass(frozen=True)
class IntentGateResult:
    """Verdict for one filing attempt."""

    applies: bool
    disposition: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def blocks_filing(self) -> bool:
        return bool(self.errors)

    def as_record(self) -> dict[str, Any]:
        """The verdict as persisted on the report (read-only downstream)."""
        return {
            "applies": self.applies,
            "disposition": self.disposition,
            "reasons": list(self.errors),
            "warnings": list(self.warnings),
            **self.evidence_summary,
        }


def _normalize_evidence(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(_as_list(raw)):
        where = f"intent_evidence[{index}]"
        if not isinstance(item, Mapping):
            errors.append(
                f"{where} must be an object with source/authority and supports or contradicts"
            )
            continue
        source = _text(item.get("source"))
        authority = _text(item.get("authority")).lower()
        supports = _text(item.get("supports"))
        contradicts = _text(item.get("contradicts"))
        if not source:
            errors.append(f"{where}.source is required - the path, URL, MR or commit that was read")
        if authority not in AUTHORITY_ORDER:
            errors.append(
                f"{where}.authority must be one of {list(AUTHORITY_ORDER)} - "
                "how much design intent this source can establish"
            )
        if not supports and not contradicts:
            errors.append(
                f"{where}: state what this source establishes - 'supports' the observed "
                "behaviour as intended, or 'contradicts' it"
            )
        entries.append(
            {
                "source": source,
                "authority": authority,
                "supports": supports,
                "contradicts": contradicts,
                "stale": bool(item.get("stale", False)),
                "note": _text(item.get("note")),
            }
        )
    return entries, errors


def _normalize_conflict(raw: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if raw in (None, "", {}, []):
        return None, []
    if not isinstance(raw, Mapping):
        return None, ["security_contract_conflict must be an object describing the conflict"]
    kind = _text(raw.get("kind")).lower()
    invariant = _text(raw.get("invariant"))
    source = _text(raw.get("source"))
    impact = _text(raw.get("impact"))
    errors: list[str] = []
    if kind not in CONFLICT_KINDS:
        errors.append(
            f"security_contract_conflict.kind must be one of {list(CONFLICT_KINDS)} - "
            "'a stricter model would be better' is not one of them"
        )
    if len(invariant) < _MIN_CONFLICT_TEXT:
        errors.append(
            "security_contract_conflict.invariant must state the specific security invariant or "
            f"promise that is broken (at least {_MIN_CONFLICT_TEXT} characters), not a preference"
        )
    if not source:
        errors.append(
            "security_contract_conflict.source is required - where the conflicting invariant comes "
            "from, and it must not be the document that establishes the intent being overridden"
        )
    if len(impact) < _MIN_CONFLICT_TEXT:
        errors.append(
            "security_contract_conflict.impact must state the concrete harm (cross-tenant access, "
            "privilege gained, promise broken) that documentation cannot waive"
        )
    return {
        "kind": kind,
        "invariant": invariant,
        "source": source,
        "impact": impact,
    }, errors


def _normalize_review(raw: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, Mapping):
        return None, []
    verdict = _text(raw.get("verdict")).lower()
    reviewer_kind = _text(raw.get("reviewer_kind")).lower()
    reviewed_by = _text(raw.get("reviewed_by"))
    notes = _text(raw.get("notes"))
    errors: list[str] = []
    if verdict and verdict not in REVIEW_VERDICTS:
        errors.append(f"intent_review.verdict must be one of {list(REVIEW_VERDICTS)}")
    if not reviewer_kind:
        errors.append("intent_review.reviewer_kind is required (independent_agent, or unavailable)")
    return {
        "reviewer_kind": reviewer_kind,
        "reviewed_by": reviewed_by,
        "verdict": verdict,
        "notes": notes,
    }, errors


def _check_intent_fields(
    *,
    status: str,
    scope: list[str],
    known_issue_or_duplicate_search: str | None,
    alternative_semantics_check: str | None,
    assumptions: str | None,
) -> list[str]:
    """Structural requirements for the intent record itself."""
    errors: list[str] = []
    if status not in INTENT_CHECK_STATUSES:
        errors.append(
            "intent_check_status is REQUIRED for a source-aware finding and must be one of "
            f"{list(INTENT_CHECK_STATUSES)}: 'completed' when the search ran (even if it found "
            "nothing), 'unavailable' when the source or its documentation could not be searched. "
            "Record the searches in intent_search_scope."
        )
    minimum_scope = (
        _MIN_SCOPE_ENTRIES_UNAVAILABLE if status == "unavailable" else _MIN_SCOPE_ENTRIES_COMPLETED
    )
    if len(scope) < minimum_scope:
        errors.append(
            f"intent_search_scope needs at least {minimum_scope} entries naming what was searched "
            "(docs/guides, specs/tests, changelog/ADRs, and the git commands used) - "
            f"got {len(scope)}. A finding that skipped the intent search cannot be filed."
        )
    if len(_text(known_issue_or_duplicate_search)) < _MIN_CONFLICT_TEXT:
        errors.append(
            "known_issue_or_duplicate_search is REQUIRED: say where you looked for a known issue, "
            "existing report, changelog entry or deliberate design note for this behaviour, and "
            "what that search returned."
        )
    if len(_text(alternative_semantics_check)) < _MIN_CONFLICT_TEXT:
        errors.append(
            "alternative_semantics_check is REQUIRED: state whether the path's semantics allow "
            "access when *any one* of several boundaries or conditions is satisfied, and why the "
            "observed case is not simply a narrower-but-satisfied alternative. Matching one branch "
            "of an OR is not a bypass."
        )
    if not _text(assumptions):
        errors.append(
            "assumptions is REQUIRED for a source-aware finding: label what you are presuming. "
            "When intent evidence is missing, say so here rather than implying the behaviour is "
            "undocumented by design."
        )
    return errors


def _check_independent_review(
    review: dict[str, Any] | None,
    *,
    caller_agent_id: str | None,
) -> list[str]:
    """The pre-filing review requirement, and who is allowed to have done it.

    A finding cannot be its own design-intent review: the filing agent's reading
    of the evidence is the thing under review, so the reviewer identity matters as
    much as the verdict.
    """
    if review is None:
        return [
            "intent_review is REQUIRED for a source-aware finding: have a different agent "
            "independently review the design intent and the counterevidence before filing "
            "(create_agent / send_message_to_agent), then record reviewer_kind, reviewed_by and "
            "the verdict. An unresolved review means record_coverage(outcome='needs_follow_up') "
            "instead of filing."
        ]

    errors: list[str] = []
    if review.get("reviewer_kind") != "independent_agent":
        errors.append(
            "intent_review.reviewer_kind must be 'independent_agent' - the agent filing a "
            "finding cannot be its own design-intent reviewer. If no reviewer is available, "
            "record the finding as needs_follow_up in the coverage ledger instead."
        )
    reviewed_by = review.get("reviewed_by", "")
    if not reviewed_by:
        errors.append("intent_review.reviewed_by must name the reviewing agent")
    elif caller_agent_id and reviewed_by == caller_agent_id:
        errors.append(
            f"intent_review.reviewed_by is the filing agent ({caller_agent_id}) - an "
            "independent review must come from a different agent"
        )
    if len(review.get("notes", "")) < _MIN_REVIEW_NOTES:
        errors.append("intent_review.notes must summarise what the reviewer checked and concluded")
    return errors


def evaluate_intent_gate(  # noqa: PLR0911, PLR0912 - one branch per routing rule, kept explicit
    *,
    source_aware: bool,
    finding_class: str | None = None,
    intent_check_status: str | None = None,
    intent_search_scope: Any = None,
    intent_evidence: Any = None,
    known_issue_or_duplicate_search: str | None = None,
    alternative_semantics_check: str | None = None,
    security_contract_conflict: Any = None,
    intent_review: Any = None,
    assumptions: str | None = None,
    caller_agent_id: str | None = None,
) -> IntentGateResult:
    """Decide whether this filing attempt may proceed, and on what footing.

    Deterministic and side-effect free: the same structured evidence always
    produces the same verdict, so a filing decision can be replayed after the
    fact. Every rejection names the outcome the agent should record instead.
    """
    style = (finding_class or "").strip().lower()
    exempt = style in _EXEMPT_FINDING_CLASSES

    evidence, evidence_errors = _normalize_evidence(intent_evidence)
    conflict, conflict_errors = _normalize_conflict(security_contract_conflict)
    review, review_errors = _normalize_review(intent_review)

    supplied_something = bool(
        evidence
        or conflict
        or review
        or _text(intent_check_status)
        or _as_list(intent_search_scope)
        or _text(known_issue_or_duplicate_search)
        or _text(alternative_semantics_check)
    )

    if not source_aware:
        # Black-box (and dependency) findings keep their previous contract. The
        # fields may still be supplied, and are validated when they are, so a
        # malformed block never reaches a stored report unnoticed.
        errors = [*evidence_errors, *conflict_errors, *review_errors]
        return IntentGateResult(
            applies=False,
            disposition="not_required",
            errors=tuple(errors),
            warnings=(
                ("Intent fields were supplied but are not required for this finding class.",)
                if supplied_something
                else ()
            ),
        )

    status = _text(intent_check_status).lower()
    scope = [entry for entry in _as_list(intent_search_scope) if _text(entry)]
    errors = [*evidence_errors, *conflict_errors, *review_errors]
    warnings: list[str] = []

    authoritative_support = [
        entry
        for entry in evidence
        if entry["authority"] in AUTHORITATIVE_AUTHORITIES
        and entry["supports"]
        and not entry["contradicts"]
    ]
    authoritative_contradict = [
        entry
        for entry in evidence
        if entry["authority"] in AUTHORITATIVE_AUTHORITIES and entry["contradicts"]
    ]
    weak_support = [
        entry
        for entry in evidence
        if entry["authority"] not in AUTHORITATIVE_AUTHORITIES and entry["supports"]
    ]
    stale = [entry for entry in evidence if entry["stale"]]

    summary = {
        "evidence_count": len(evidence),
        "authoritative_support": [entry["source"] for entry in authoritative_support],
        "authoritative_contradict": [entry["source"] for entry in authoritative_contradict],
        "weak_support": [entry["source"] for entry in weak_support],
        "stale": [entry["source"] for entry in stale],
    }

    if exempt:
        # Dependency CVEs are defined by an advisory, not by this repo's design
        # intent; the intended question (is the package reachable, is the version
        # affected) belongs to the dependency path's own checks.
        return IntentGateResult(
            applies=False,
            disposition="not_required",
            errors=tuple(errors),
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    errors.extend(
        _check_intent_fields(
            status=status,
            scope=scope,
            known_issue_or_duplicate_search=known_issue_or_duplicate_search,
            alternative_semantics_check=alternative_semantics_check,
            assumptions=assumptions,
        )
    )

    # --- independent pre-filing review -------------------------------------
    verdict = (review or {}).get("verdict", "")
    errors.extend(_check_independent_review(review, caller_agent_id=caller_agent_id))

    # A `conflict_confirmed` verdict with no conflict payload is only a problem
    # when an override is actually needed — i.e. when authoritative evidence
    # supports the behaviour, which the intended_behavior branch below decides.
    # Otherwise the reviewer is confirming there is no intent evidence to defer
    # to, which is a legitimate basis for filing.
    reviewed_needs_override = verdict == "intent_confirmed"

    # --- routing ------------------------------------------------------------
    if verdict == "unresolved":
        errors.append(
            "The reviewer left the intent unresolved. Record "
            "record_coverage(outcome='needs_follow_up', evidence=<what is unresolved>) and do not "
            "file: unresolved intent is a follow-up, not a vulnerability."
        )
        return IntentGateResult(
            applies=True,
            disposition="needs_follow_up",
            errors=tuple(errors),
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    if stale and not authoritative_contradict:
        errors.append(
            "Some intent evidence is marked stale ("
            + ", ".join(entry["source"] for entry in stale)
            + "). Surface the staleness and record the finding as needs_follow_up rather than "
            "letting a superseded document decide the outcome."
        )
        return IntentGateResult(
            applies=True,
            disposition="needs_follow_up",
            errors=tuple(errors),
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    if authoritative_support and authoritative_contradict:
        errors.append(
            "Authoritative sources disagree ("
            + ", ".join(entry["source"] for entry in authoritative_support)
            + " support the behaviour; "
            + ", ".join(entry["source"] for entry in authoritative_contradict)
            + " contradict it). Report the ambiguity as a follow-up or supply a "
            "security_contract_conflict that resolves which invariant governs."
        )
        if conflict is None:
            return IntentGateResult(
                applies=True,
                disposition="needs_follow_up",
                errors=tuple(errors),
                warnings=tuple(warnings),
                evidence_summary=summary,
            )

    if (authoritative_support or reviewed_needs_override) and conflict is None:
        sources = ", ".join(entry["source"] for entry in authoritative_support) or (
            "the independent review"
        )
        errors.append(
            f"Intended behaviour: {sources} describes the observed behaviour as the intended "
            "contract. A finding contradicted by the repository's own authoritative evidence must "
            "not be filed. Either record it as intended via "
            "record_coverage(outcome='not_applicable', evidence='<source> defines this exact "
            "behaviour'), file the *conflict* instead (docs and code genuinely disagree), or "
            "supply security_contract_conflict with a concrete conflicting invariant, an external "
            "security promise, cross-tenant harm, or an exploitable boundary that documentation "
            "cannot waive."
        )
        return IntentGateResult(
            applies=True,
            disposition="intended_behavior",
            errors=tuple(errors),
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    if conflict is not None:
        supporting_sources = {entry["source"] for entry in authoritative_support}
        if conflict["source"] in supporting_sources:
            errors.append(
                "security_contract_conflict.source points at the same document that establishes "
                "the intent being overridden - the conflicting invariant has to come from "
                "somewhere else."
            )
            return IntentGateResult(
                applies=True,
                disposition="needs_follow_up",
                errors=tuple(errors),
                warnings=tuple(warnings),
                evidence_summary=summary,
            )

    if errors:
        # Any structural error is fatal for a source-aware filing: an incomplete
        # intent block is exactly the state this gate exists to refuse.
        disposition = "needs_follow_up" if status == "unavailable" else "report_conflict"
        return IntentGateResult(
            applies=True,
            disposition=disposition,
            errors=tuple(errors),
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    if weak_support and not authoritative_support and not authoritative_contradict:
        warnings.append(
            "Only non-authoritative evidence supports this behaviour ("
            + ", ".join(entry["source"] for entry in weak_support)
            + "). A comment, changelog or operator doc cannot establish intent on its own; the "
            "assumptions field must say why it does not settle the question."
        )

    if authoritative_contradict:
        return IntentGateResult(
            applies=True,
            disposition="report_conflict",
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    if conflict is not None:
        return IntentGateResult(
            applies=True,
            disposition="report_conflict",
            warnings=tuple(warnings),
            evidence_summary=summary,
        )

    empty_evidence = not evidence
    if empty_evidence:
        warnings.append(
            "No intent evidence was found in the searches recorded in intent_search_scope. "
            "Absence is not evidence of intent in either direction; the finding stands on the "
            "labelled assumptions and counterevidence it carries."
        )
    return IntentGateResult(
        applies=True,
        disposition="report_with_assumptions",
        warnings=tuple(warnings),
        evidence_summary=summary,
    )


def intent_block_for_output(report: Mapping[str, Any]) -> dict[str, Any] | None:
    """Canonical projection of a report's intent block, for writers and the UI.

    Returns ``None`` when the report has no intent metadata at all, so existing
    reports (and black-box findings) render exactly as they did before.
    """
    gate = report.get("intent_gate")
    status = _text(report.get("intent_check_status"))
    evidence = report.get("intent_evidence")
    if not status and not gate and not evidence:
        return None

    entries: list[dict[str, Any]] = []
    for item in _as_list(evidence):
        if not isinstance(item, Mapping):
            continue
        entries.append(
            {
                "source": _text(item.get("source")),
                "authority": _text(item.get("authority")),
                "supports": _text(item.get("supports")),
                "contradicts": _text(item.get("contradicts")),
                "stale": bool(item.get("stale", False)),
                "note": _text(item.get("note")),
            }
        )

    conflict = report.get("security_contract_conflict")
    review = report.get("intent_review")
    search_scope = [
        _text(item) for item in _as_list(report.get("intent_search_scope")) if _text(item)
    ]
    return {
        "status": status or "not_recorded",
        "disposition": _text((gate or {}).get("disposition")) or "not_recorded",
        "search_scope": search_scope,
        "known_issue_search": _text(report.get("known_issue_or_duplicate_search")),
        "alternative_semantics_check": _text(report.get("alternative_semantics_check")),
        "evidence": entries,
        "conflict": dict(conflict) if isinstance(conflict, Mapping) else None,
        "review": dict(review) if isinstance(review, Mapping) else None,
        "warnings": list((gate or {}).get("warnings") or []),
        "reasons": list((gate or {}).get("reasons") or []),
    }


def summarize_for_message(result: IntentGateResult) -> str:
    """One-line summary used in tool results and logs."""
    if not result.applies:
        return "intent gate: not required"
    parts = [f"intent gate: {result.disposition}"]
    if result.errors:
        parts.append(f"{len(result.errors)} blocking issue(s)")
    if result.warnings:
        parts.append(f"{len(result.warnings)} warning(s)")
    return ", ".join(parts)


__all__ = [
    "AUTHORITATIVE_AUTHORITIES",
    "AUTHORITY_ORDER",
    "CONFLICT_KINDS",
    "DISPOSITIONS",
    "INTENT_CHECK_STATUSES",
    "REVIEW_VERDICTS",
    "IntentGateResult",
    "evaluate_intent_gate",
    "intent_block_for_output",
    "is_source_aware",
    "summarize_for_message",
]
