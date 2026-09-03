"""Passive team fan-out telemetry - run-record observability only.

Goal (observability only): record, on real scans we were already going to
run, deterministic local metrics about the team-fan-out stage into the
existing ``run_record`` (``run.json``) under a ``team_fanout`` section for
later offline analysis.  No new model calls, no A/B scans, no telemetry-driven
behaviour changes, no third-party analytics, no new files.

Lifecycle:

- **Stage metadata** (:func:`record_stage`) is persisted as soon as a
  ``TeamStageOutcome`` exists (any non-``None`` outcome, including
  ``width_1`` / ``no_source_roots`` / ``zero_units`` / ``spawner_unavailable``)
  so an interrupted scan after fan-out still records that the team existed.
- **Per-agent metrics** (:func:`finalize_and_persist`) are best-effort and
  non-fatal, gathered at scan settlement from existing observed state.

Data authorities (no parallel counter store, no LLM calls):

- ``AgentCoordinator`` per-agent snapshot - model_requests / tool_calls /
  tool_groups / avg_tool_width / tools_serial_ms / tools_wall_ms / status,
  and per-agent token totals fed by the wide-turn hooks
  (``agent_metrics_snapshot`` read seam; falls back to ``runtime_snapshot``).
- ``ReportState.run_record["llm_usage"]["agents"]`` - per-agent model id and
  the *allocated* cost (see :func:`_allocated_cost_usd`).
- Coverage ledger (``strix.tools.coverage.tools.get_coverage_entries``) -
  entries keyed by ``agent_id``.
- ``ReportState.vulnerability_reports`` - findings keyed by originating
  ``agent_id`` and lifecycle ``state``.

Aggregates are **workers-only** (root is deliberately kept out: the question
this telemetry answers is what the workers did on top of the root's own
effort, which is stored separately under ``"root"`` on the same metric scale).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

# The stage's own success predicate is reused on purpose so the record can
# never disagree with ``successfully_spawned`` (integration.py is frozen).
from strix.team.integration import (
    TeamStageOutcome,
    _is_successful_spawn,  # pyright: ignore[reportPrivateUsage]
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from strix.team.fanout import SpawnedWorker, TeamPlan

logger = logging.getLogger("strix.team.telemetry")

TEAM_FANOUT_KEY = "team_fanout"

#: The five lifecycle states, always present (zero counts explicit) so
#: downstream aggregation never has to special-case missing keys.
FINDING_STATE_KEYS: tuple[str, ...] = (
    "candidate",
    "verified",
    "retracted",
    "rejected",
    "proof_gap",
)
_EMPTY_FINDING_COUNTS: dict[str, int] = dict.fromkeys(FINDING_STATE_KEYS, 0)


def _as_dict(value: object) -> dict[str, Any]:
    """Narrow untyped observed data (run_record / snapshots) to a str-keyed dict."""
    if not isinstance(value, dict):
        return {}
    raw = cast("dict[object, object]", value)
    return {str(key): item for key, item in raw.items()}


def _as_dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    items = cast("Sequence[object]", value)
    # _as_dict maps non-dict items to {}; drop those so callers only see records.
    return [entry for entry in (_as_dict(item) for item in items) if entry]


def _assignment_by_shard(plan: TeamPlan | None) -> dict[int, Any]:
    if plan is None:
        return {}
    return {assignment.shard_id: assignment for assignment in plan.assignments}


def _worker_stage_entry(worker: SpawnedWorker, assignment: Any | None) -> dict[str, Any]:
    """Stage-time worker entry (identity + size only - never the file list)."""
    success = _is_successful_spawn(worker)
    return {
        "worker_id": worker.worker_id,
        "shard_id": worker.shard_id,
        "agent_id": worker.agent_id if success else None,
        "spawn_status": "spawned" if success else "failed",
        "file_count": len(assignment.files) if assignment is not None else 0,
        "shard_weight": assignment.weight if assignment is not None else 0,
        "shard_loc": assignment.loc if assignment is not None else 0,
    }


def build_stage_section(outcome: TeamStageOutcome) -> dict[str, Any]:
    """Early, deterministic ``team_fanout`` record (stage facts only).

    ``root`` and ``aggregates`` are ``None`` here; settlement fills them via
    :func:`finalize_team_section`.  The ``workers`` list carries only
    identity/size fields - the manifest owns full file lists and this section
    never duplicates them.
    """
    plan = outcome.plan
    assignments = _assignment_by_shard(plan)
    section: dict[str, Any] = {
        "team_width_requested": outcome.team_width,
        "team_width_effective": (
            plan.effective_workers if plan is not None else outcome.team_width
        ),
        "workers_attempted": outcome.attempted,
        "workers_spawned": outcome.successfully_spawned,
        "workers_failed": outcome.failed_to_spawn,
        "reason": outcome.reason,
        "total_weight": plan.total_weight if plan is not None else 0,
        # TeamPlan carries no total_loc; derive it from the frozen assignments.
        "total_loc": sum(assignment.loc for assignment in plan.assignments) if plan else 0,
        "workers": [
            _worker_stage_entry(worker, assignments.get(worker.shard_id))
            for worker in outcome.spawned
        ],
        # Settlement-time fields: absent (None) on an interrupted run.
        "root": None,
        "aggregates": None,
    }
    return section


def _allocated_cost_usd(usage_entry: dict[str, Any] | None) -> float | None:
    """Per-agent cost, named ``allocated_cost_usd`` on purpose.

    The ledger allocates the run's observed/estimated cost across agents by
    token share when only an aggregate provider charge is available - it is an
    allocation, not an exact provider-side charge.  Do NOT rename this field
    to ``cost_usd`` / ``exact_cost_usd``.
    """
    if usage_entry is None:
        return None
    cost = usage_entry.get("cost")
    if isinstance(cost, int | float):
        return round(max(0.0, float(cost)), 10)
    return None


def _agent_usage_map(usage_agents: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in _as_dict_list(list(usage_agents or ())):
        agent_id = str(entry.get("agent_id") or "").strip()
        if agent_id:
            out[agent_id] = entry
    return out


def _coordinator_snapshot(coordinator: Any | None, agent_id: str) -> dict[str, Any]:
    """Read one agent's metrics from the coordinator, defensively.

    Uses the ``agent_metrics_snapshot`` read seam when present, otherwise the
    existing ``runtime_snapshot()`` mapping; any failure yields ``{}`` (the
    collector is best-effort by contract).
    """
    if coordinator is None:
        return {}
    try:
        accessor = getattr(coordinator, "agent_metrics_snapshot", None)
        if callable(accessor):
            return _as_dict(accessor(agent_id))
        snapshot = getattr(coordinator, "runtime_snapshot", None)
        if callable(snapshot):
            return _as_dict(_as_dict(snapshot()).get(agent_id))
    except Exception:  # noqa: BLE001 - telemetry must never break a scan
        logger.warning("team telemetry: coordinator snapshot failed for %s (non-fatal)", agent_id)
    return {}


def _count_coverage(entries: Sequence[dict[str, Any]], agent_id: str) -> int:
    return sum(1 for entry in entries if str(entry.get("agent_id") or "") == agent_id)


def _count_findings(
    vulnerabilities: Sequence[dict[str, Any]],
    agent_id: str,
) -> dict[str, int]:
    """Per-lifecycle-state finding counts for one originating agent."""
    from strix.report.finding_state import state_of  # noqa: PLC0415 - light, lazy

    counts: dict[str, int] = dict(_EMPTY_FINDING_COUNTS)
    for report in vulnerabilities:
        if str(report.get("agent_id") or "") != agent_id:
            continue
        state = state_of(report)
        if state in counts:
            counts[state] += 1
    return counts


def _metric_block(
    *,
    agent_id: str,
    coordinator: Any | None,
    usage_map: dict[str, dict[str, Any]],
    coverage_entries: Sequence[dict[str, Any]],
    vulnerabilities: Sequence[dict[str, Any]],
    default_model: str | None,
) -> dict[str, Any]:
    """One agent's final metric block (root and workers share the shape)."""
    snapshot = _coordinator_snapshot(coordinator, agent_id)
    usage_entry = usage_map.get(agent_id)
    usage_model = usage_entry.get("model") if isinstance(usage_entry, dict) else None
    model = (
        usage_model
        if isinstance(usage_model, str) and usage_model.strip()
        else (default_model or "")
    )

    def counter(name: str) -> int:
        return max(0, int(snapshot.get(name) or 0))

    input_tokens = counter("input_tokens")
    cached_input_tokens = counter("cached_input_tokens")
    output_tokens = counter("output_tokens")
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    tools_wall_ms = float(snapshot.get("tools_wall_ms") or 0)
    tools_serial_ms = float(snapshot.get("tools_serial_ms") or 0)

    return {
        "final_status": str(snapshot.get("status") or "unknown"),
        "model": model or None,
        "model_requests": counter("model_requests"),
        "tool_calls": counter("tool_calls"),
        "tool_groups": counter("tool_groups"),
        "avg_tool_width": snapshot.get("avg_tool_width"),
        "tools_serial_ms": tools_serial_ms,
        "tools_wall_ms": tools_wall_ms,
        # tools_serial_ms / tools_wall_ms; null when no wall time was observed.
        "effective_parallelism": tools_serial_ms / tools_wall_ms if tools_wall_ms > 0 else None,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "cache_ratio": cached_input_tokens / max(1, input_tokens),
        # See _allocated_cost_usd for the allocation semantics of this name.
        "allocated_cost_usd": _allocated_cost_usd(usage_entry),
        "coverage_entries": _count_coverage(coverage_entries, agent_id),
        "findings_by_state": _count_findings(vulnerabilities, agent_id),
    }


def finalize_team_section(
    section: dict[str, Any],
    *,
    coordinator: Any | None = None,
    usage_agents: Sequence[dict[str, Any]] | None = None,
    root_agent_id: str | None = None,
    root_model: str | None = None,
    worker_model: str | None = None,
    coverage_entries: Sequence[dict[str, Any]] | None = None,
    vulnerabilities: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fill the per-agent metric blocks, root block, and team aggregates.

    Pure over the provided data sources (used by the runner with the real
    coordinator/report state, and by tests with fakes).  Deterministic:
    worker blocks keep their stage order and are additionally sorted by
    (shard_id, worker_id).  Failed attempts stay in ``workers`` with
    ``spawn_status: "failed"`` and no metric block.
    """
    usage_map = _agent_usage_map(usage_agents)
    coverage = list(coverage_entries or [])
    findings = list(vulnerabilities or [])

    successful: list[dict[str, Any]] = []
    workers_out: list[dict[str, Any]] = []
    for worker_entry in section.get("workers") or ():
        agent_id = worker_entry.get("agent_id")
        is_spawned = (
            worker_entry.get("spawn_status") == "spawned"
            and isinstance(agent_id, str)
            and bool(agent_id)
        )
        if is_spawned:
            worker_entry.update(
                _metric_block(
                    agent_id=agent_id,
                    coordinator=coordinator,
                    usage_map=usage_map,
                    coverage_entries=coverage,
                    vulnerabilities=findings,
                    default_model=worker_model,
                )
            )
            successful.append(worker_entry)
        workers_out.append(worker_entry)

    workers_out.sort(key=lambda entry: (entry.get("shard_id"), entry.get("worker_id")))

    root_block: dict[str, Any] | None = None
    if isinstance(root_agent_id, str) and root_agent_id:
        root_block = _metric_block(
            agent_id=root_agent_id,
            coordinator=coordinator,
            usage_map=usage_map,
            coverage_entries=coverage,
            vulnerabilities=findings,
            default_model=root_model,
        )

    section["workers"] = workers_out
    section["root"] = root_block
    section["aggregates"] = _aggregates(successful)
    return section


def _aggregates(worker_blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Team totals across **workers only**.

    Root is deliberately excluded: its own effort is stored under ``"root"``
    on the same scale, so later analysis compares "did workers reduce root
    reasoning?" without the root summing itself into the team totals.
    """

    def total_int(key: str) -> int:
        return sum(int(block.get(key) or 0) for block in worker_blocks)

    findings: dict[str, int] = dict(_EMPTY_FINDING_COUNTS)
    for block in worker_blocks:
        per_state = _as_dict(block.get("findings_by_state"))
        for state in FINDING_STATE_KEYS:
            findings[state] += int(per_state.get(state) or 0)

    allocated = [
        float(block["allocated_cost_usd"])
        for block in worker_blocks
        if isinstance(block.get("allocated_cost_usd"), int | float)
    ]
    return {
        "team_model_requests": total_int("model_requests"),
        "team_tool_calls": total_int("tool_calls"),
        "team_tool_groups": total_int("tool_groups"),
        "team_input_tokens": total_int("input_tokens"),
        "team_cached_input_tokens": total_int("cached_input_tokens"),
        "team_uncached_input_tokens": total_int("uncached_input_tokens"),
        "team_output_tokens": total_int("output_tokens"),
        # Sum of per-worker allocations (see _allocated_cost_usd semantics).
        "team_allocated_cost_usd": round(sum(allocated), 10),
        "team_coverage_entries": total_int("coverage_entries"),
        "team_findings_by_state": findings,
    }


def _default_usage_agents(report_state: Any | None) -> list[dict[str, Any]] | None:
    if report_state is None:
        return None
    try:
        record = _as_dict(getattr(report_state, "run_record", None))
        usage = _as_dict(record.get("llm_usage"))
        if usage:
            return _as_dict_list(usage.get("agents"))
    except Exception:  # noqa: BLE001 - best-effort
        logger.warning("team telemetry: llm_usage read failed (non-fatal)")
    return None


def _default_vulnerabilities(report_state: Any | None) -> list[dict[str, Any]]:
    if report_state is None:
        return []
    try:
        reader = getattr(report_state, "get_existing_vulnerabilities", None)
        if callable(reader):
            return _as_dict_list(reader())
        return _as_dict_list(getattr(report_state, "vulnerability_reports", None))
    except Exception:  # noqa: BLE001 - best-effort
        return []


def _default_coverage_entries() -> list[dict[str, Any]]:
    try:
        from strix.tools.coverage.tools import get_coverage_entries  # noqa: PLC0415

        return list(get_coverage_entries())
    except Exception:  # noqa: BLE001 - best-effort
        return []


def _persist(report_state: Any, section: dict[str, Any]) -> None:
    """Write the section under ``run_record`` and save."""
    record = getattr(report_state, "run_record", None)
    if not isinstance(record, dict):
        raise TypeError("report_state.run_record must be a dict")
    record[TEAM_FANOUT_KEY] = section
    saver = getattr(report_state, "save_run_data", None)
    if callable(saver):
        saver()
    logger.debug("team telemetry: persisted %r section", TEAM_FANOUT_KEY)


def record_stage(report_state: Any, outcome: TeamStageOutcome) -> dict[str, Any]:
    """Persist the early stage section; never raises into the scan."""
    try:
        section = build_stage_section(outcome)
        _persist(report_state, section)
    except Exception:  # noqa: BLE001 - telemetry failure must never break a pentest
        logger.warning("team telemetry: stage record failed (non-fatal)", exc_info=True)
        return {}
    return section


def finalize_and_persist(
    report_state: Any,
    *,
    coordinator: Any | None = None,
    root_agent_id: str | None = None,
    root_model: str | None = None,
    worker_model: str | None = None,
    coverage_entries: Sequence[dict[str, Any]] | None = None,
    vulnerabilities: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Best-effort settlement: fill metrics and persist; never raises.

    Reads defaults from ``report_state`` when the optional lists are not
    given; ``coverage_entries=None`` falls back to the live coverage ledger.
    """
    try:
        record = _as_dict(getattr(report_state, "run_record", None))
        section = _as_dict(record.get(TEAM_FANOUT_KEY))
        if not section:
            return {}
        usage_agents = _default_usage_agents(report_state)
        if vulnerabilities is None:
            vulnerabilities = _default_vulnerabilities(report_state)
        else:
            vulnerabilities = list(vulnerabilities)
        if coverage_entries is None:
            coverage_entries = _default_coverage_entries()
        else:
            coverage_entries = list(coverage_entries)
        finalized = finalize_team_section(
            section,
            coordinator=coordinator,
            usage_agents=usage_agents,
            root_agent_id=root_agent_id,
            root_model=root_model,
            worker_model=worker_model,
            coverage_entries=coverage_entries,
            vulnerabilities=vulnerabilities,
        )
        _persist(report_state, finalized)
    except Exception:  # noqa: BLE001 - telemetry failure must never break a pentest
        logger.warning("team telemetry: finalize failed (non-fatal)", exc_info=True)
        return {}
    return finalized
