"""Team fan-out wired into source scans - the orchestration call site.

This module is the *integration seam*: it runs the canonical
inventory -> partition -> team-plan -> spawn pipeline at the scan-orchestration
layer (the runner calls it for fresh whitebox/local-source scans) and returns a
deterministic outcome the caller can log / surface.  It deliberately lives
outside ``strix.tools.source_partition``:

- the partition package stays a pure library that only knows
  ``Sequence[Path]`` roots + ``PartitionConfig`` - it never sees
  ``report_state``, coordinators, agents, or spawners;
- this module is the only place that glues partition output to the existing
  agent spawning primitives (``strix.core.execution.spawn_child_agent`` via an
  injected ``spawn_worker`` - the runner passes its own adapter closure, tests
  pass fakes).  Runtime context (``parent_ctx``) is injected by that adapter
  at the runner boundary, never inside the deterministic assignment layer.

``STRIX_TEAM_WIDTH`` (default ``1``) is the cost-control gate:

- ``team_width <= 1`` (reason ``width_1``): legacy behaviour preserved -
  inventory/partition still run (always-on manifest validation and note
  propagation), but **no** team workers are spawned; the root agent remains
  the only agent.
- ``team_width > 1``: full fan-out - ``partition_units`` is asked for
  ``team_width`` workers and one ``SpawnedWorker`` is produced per effective
  shard (``effective_workers`` may be lower when the tree has fewer useful
  units).  Turning this knob up is the deliberate, explicit act that spends
  PAYG model budget on parallel workers.

Non-source scans (no local sources / repo checkouts reachable) skip the whole
path and are a clean no-op (``no_source_roots``).  A manifest with zero
effective workers produces zero team workers without error (``zero_units``).

Outcome accounting distinguishes *attempted* from *successfully spawned*:
``SpawnedWorker(error=...)`` entries never count as spawned.  ``reason`` is
one of: ``no_source_roots``, ``zero_units``, ``width_1``,
``spawner_unavailable``, ``spawned``, ``partial_spawn_failure``,
``spawn_failed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strix.team.fanout import (
    SpawnedWorker,
    TeamFanout,
    TeamPlan,
    WorkerSpawner,
    build_team_plan,
)
from strix.tools.source_partition import (
    PartitionConfig,
    build_partition_units,
    inventory_source,
    partition_units,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from strix.tools.source_partition.models import PartitionManifest

logger = logging.getLogger("strix.team.integration")


@dataclass(frozen=True, slots=True)
class TeamStageOutcome:
    """Deterministic result of one team-fan-out stage call.

    ``plan`` is always present when the scan had source roots (inventory +
    partition always run), even when no workers were spawned; ``notes`` from
    the partitioner are therefore reachable on ``plan.notes``.  ``reason``
    explains why nothing (or fewer things) were spawned.

    Success accounting is explicit: ``spawned`` (the retained per-worker
    tuple, for backwards compatibility) may contain failed attempts
    (``SpawnedWorker.error is not None``); ``attempted`` /
    ``successfully_spawned`` / ``failed_to_spawn`` are the counters the runner
    should log and act on.
    """

    enabled: bool
    team_width: int
    plan: TeamPlan | None
    spawned: tuple[SpawnedWorker, ...]
    reason: str
    attempted: int = 0
    successfully_spawned: int = 0
    failed_to_spawn: int = 0


def resolve_team_roots(
    *,
    report_state: Any | None,
    local_sources: list[Any] | None,
) -> list[Path]:
    """Root discovery, reusing the authoritative source_inspect resolver.

    ``resolve_authorized_roots(report_state)`` (from
    ``strix.tools.source_inspect``) is the canonical rule - local sources plus
    cloned repository checkouts recorded on the run.  When no report state is
    live (or it yields nothing), fall back to the runner's ``local_sources``
    parameter, which carries the same ``source_path`` shape for CLI mounts.
    """
    if report_state is not None:
        try:
            from strix.tools.source_inspect.tool import resolve_authorized_roots  # noqa: PLC0415

            resolved = resolve_authorized_roots(report_state)
        except Exception:  # noqa: BLE001 - root discovery must never break the scan
            logger.debug("team roots: report-state resolution failed; falling back", exc_info=True)
            resolved = []
        if resolved:
            return resolved

    fallback_roots: list[Path] = []
    for source in local_sources or []:
        raw = (
            cast("dict[str, Any]", source).get("source_path") if isinstance(source, dict) else None
        )
        if isinstance(raw, str) and raw.strip():
            fallback_roots.append(Path(raw))
    # Deduplicate deterministically (the inventory canonicalizes + sorts the
    # surviving roots anyway; this only avoids passing the same path twice).
    seen: set[str] = set()
    deduped: list[Path] = []
    for root in fallback_roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _team_width() -> int:
    """Resolve ``STRIX_TEAM_WIDTH`` (default 1)."""
    from strix.config.loader import load_settings  # noqa: PLC0415

    settings = load_settings()
    team = getattr(settings, "team", None)
    return int(getattr(team, "team_width", 1) or 1)


async def stage_source_team_fanout(
    *,
    report_state: Any | None,
    local_sources: list[Any] | None,
    spawn_worker: WorkerSpawner | None,
    objective: str,
    team_width: int | None = None,
    worker_skills: Sequence[str] = (),
    config: PartitionConfig | None = None,
    manifest: PartitionManifest | None = None,
) -> TeamStageOutcome:
    """Run the canonical source-scan team stage.

    ``manifest`` (optional) lets a caller inject an already-built manifest
    (tests use it to pin malformed-manifest behaviour at this seam); when
    ``None`` the stage runs inventory + partition itself.  A malformed
    manifest raises ``ValueError`` before any worker is spawned.
    ``worker_skills`` (default empty) is forwarded to every worker's spawn
    call via ``TeamFanout(skills=...)`` - the runner passes the scan's
    configured skills.  ``parent_history`` is never propagated.
    """
    width = team_width if team_width is not None else _team_width()

    roots = resolve_team_roots(report_state=report_state, local_sources=local_sources)
    if not roots:
        return TeamStageOutcome(
            enabled=False, team_width=width, plan=None, spawned=(), reason="no_source_roots"
        )

    if manifest is None:
        cfg = config or PartitionConfig()
        inventory = inventory_source(roots, config=cfg)
        units, unit_notes = build_partition_units(inventory, config=cfg)
        manifest = partition_units(
            units,
            workers=width,
            config=cfg,
            notes=(*inventory.notes, *unit_notes),
        )

    plan = build_team_plan(manifest, objective=objective)  # validates the manifest
    if width <= 1:
        return TeamStageOutcome(
            enabled=True, team_width=width, plan=plan, spawned=(), reason="width_1"
        )
    if manifest.effective_workers == 0:
        return TeamStageOutcome(
            enabled=True, team_width=width, plan=plan, spawned=(), reason="zero_units"
        )
    if spawn_worker is None:
        return TeamStageOutcome(
            enabled=True,
            team_width=width,
            plan=plan,
            spawned=(),
            reason="spawner_unavailable",
        )

    fanout = TeamFanout(plan, spawn_worker, skills=tuple(worker_skills))
    spawned = await fanout.spawn_all()
    attempted = len(spawned)
    successfully_spawned = sum(1 for worker in spawned if worker.error is None)
    failed_to_spawn = attempted - successfully_spawned
    if successfully_spawned and failed_to_spawn == 0:
        reason = "spawned"
    elif successfully_spawned and failed_to_spawn:
        reason = "partial_spawn_failure"
    else:
        reason = "spawn_failed"
    logger.debug(
        "team fan-out: requested=%d effective=%d attempted=%d spawned=%d failed=%d "
        "notes=%d objective_len=%d",
        plan.requested_workers,
        plan.effective_workers,
        attempted,
        successfully_spawned,
        failed_to_spawn,
        len(plan.notes),
        len(objective),
    )
    return TeamStageOutcome(
        enabled=True,
        team_width=width,
        plan=plan,
        spawned=spawned,
        reason=reason,
        attempted=attempted,
        successfully_spawned=successfully_spawned,
        failed_to_spawn=failed_to_spawn,
    )


def build_root_team_handoff(
    plan: TeamPlan | None,
    spawned: Sequence[SpawnedWorker],
) -> str | None:
    """Compact coordinator-owned handoff for the root's initial task.

    Lists only workers that actually spawned (``error is None``), ordered by
    shard id, with their agent ids - no file lists, no transcripts.  Returns
    ``None`` when nothing spawned (the root then runs exactly as before).
    """
    if plan is None:
        return None
    successful = sorted(
        (worker for worker in spawned if worker.error is None and worker.agent_id),
        key=lambda worker: (worker.shard_id, worker.worker_id),
    )
    if not successful:
        return None
    worker_lines = [
        f"- {worker.name} ({worker.agent_id}) → shard {worker.shard_id}" for worker in successful
    ]
    return "\n".join(
        [
            "Deterministic source team already started.",
            "",
            "Workers:",
            *worker_lines,
            "",
            "Broad repository discovery has already been partitioned.",
            "Do not redo broad source discovery or spawn overlapping source workers.",
            "Continue orchestration/non-duplicative investigation, monitor these children,",
            "and consume their completion reports through the existing agent mechanisms.",
        ]
    )


def log_team_stage_outcome(
    outcome: TeamStageOutcome,
    *,
    scan_id: str | None = None,
    sink: logging.Logger = logger,
) -> None:
    """Log the stage outcome with correct success accounting.

    ``spawned N worker(s)`` is only ever emitted when ``N`` equals the number
    of workers that actually spawned; failed attempts are never reported as
    spawned.
    """
    label = f" for scan {scan_id}" if scan_id else ""
    if not outcome.enabled:
        sink.info("team fan-out: no-op (no source roots)%s", label)
        return
    if outcome.successfully_spawned:
        if outcome.failed_to_spawn:
            sink.info(
                "team fan-out: spawned %d worker(s), %d failed to spawn%s",
                outcome.successfully_spawned,
                outcome.failed_to_spawn,
                label,
            )
        else:
            sink.info("team fan-out: spawned %d worker(s)%s", outcome.successfully_spawned, label)
    elif outcome.attempted:
        sink.info(
            "team fan-out: no workers spawned (%d attempt(s) failed)%s",
            outcome.attempted,
            label,
        )
    else:
        sink.info("team fan-out: no-op (reason=%s)%s", outcome.reason, label)
