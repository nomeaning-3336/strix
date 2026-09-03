"""Deterministic team fan-out over source partitions - v1 library.

Consumes a :class:`PartitionManifest` (see ``strix.tools.source_partition``)
and turns it into immutable per-worker assignments that cooperating worker
agents execute in parallel.  Partitioning stays fully independent of agents:
this package never re-invents inventory/partition logic and never knows about
``report_state`` / coordinator internals - it only shapes already-produced
shards into worker scope.

Design rules enforced here (and by the tests):

- **One immutable assignment per effective shard.**  Assignments are built
  from ``manifest.effective_workers`` shards, never from the requested count;
  an empty manifest (``effective_workers == 0``) produces zero assignments
  without error.
- **Manifests are coordinator-owned execution plans, not prompts.**  Workers
  receive an exact, immutable file list plus a compact structured task packet
  (``objective``, ``shard_id``, ``files``, ``known_facts``,
  ``open_questions``, ``evidence_refs``, ``do_not_repeat``).  They cannot see
  or mutate the :class:`PartitionManifest` through their assignment, and a
  worker is told its scope is exactly its shard: no broad re-discovery, no
  silent expansion into another shard.
- **No parent-transcript duplication.**  Workers are spawned with an empty
  ``parent_history`` (the existing SDK adapter requires the argument; the team
  layer deliberately passes nothing) - the compact packet is the handoff.
- **Reuses existing Strix spawning.**  ``spawn_worker`` is an injected
  ``name/task/skills/parent_history``-shaped callable so tests use fake
  workers while real runs bind :func:`scan_worker_spawner` to
  ``strix.core.execution.spawn_child_agent`` (the same primitive the
  ``create_agent`` tool uses).  No new agent runtime, no model-visible tool,
  no ``tool_policy`` change.
- **Failure isolation.**  A spawn failure for one worker is recorded and the
  remaining workers still spawn; completion aggregation is deterministic
  (ordered by shard id) and never mutates other workers' results.

Not in v1 (documented limitations): worker-to-worker messaging, dynamic shard
stealing/rebalancing, adaptive routing, trajectory diet, and wiring this layer
into a coordinator module (a later commit's job).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence

    from strix.tools.source_partition.models import PartitionManifest

__all__ = [
    "WORKER_SCOPE_DIRECTIVE",
    "SpawnedWorker",
    "TeamAssignment",
    "TeamFanout",
    "TeamPlan",
    "TeamResult",
    "WorkerOutcome",
    "WorkerSpawner",
    "aggregate_worker_results",
    "bind_worker_spawner",
    "build_team_assignments",
    "build_team_plan",
    "build_worker_task_packet",
    "render_worker_task",
]

#: Spawner shape compatible with ``strix.core.execution.spawn_child_agent``
#: keyword subset (``name``, ``task``, ``skills``, ``parent_history``); the
#: runner-scope arguments are captured by the binder, never passed per worker.
WorkerSpawner = Callable[..., Awaitable[dict[str, Any]]]

#: The one sentence every worker is told, verbatim (single source of truth so
#: tests can assert the exact scope wording and it cannot drift per caller).
WORKER_SCOPE_DIRECTIVE = (
    "Your primary source scope is exactly this shard. "
    "Do not repeat broad repository discovery already completed by the coordinator. "
    "Follow dependencies outside the shard only when evidence from assigned files "
    "requires it, and record that boundary crossing."
)

_DEFAULT_DO_NOT_REPEAT: tuple[str, ...] = (
    "broad repository-level discovery already performed by the coordinator",
    "scanning files that belong to another worker's shard",
)


@dataclass(frozen=True, slots=True)
class TeamAssignment:
    """One immutable worker scope derived from exactly one manifest shard.

    Immutable by construction: the file list is copied to a tuple at build
    time and the assignment never references the ``PartitionManifest``, so a
    worker (or a caller that later mutates the manifest) cannot change what
    the worker was assigned.
    """

    #: Ordinal in the team (0..effective-1); equals the manifest shard id
    #: after the partitioner's contiguous renumbering.
    worker_id: int
    #: Manifest shard id this assignment came from.
    shard_id: int
    #: Exact, immutable file list (manifest spelling, case-folded order).
    files: tuple[str, ...]
    #: Aggregate shard weight (deterministic integer).
    weight: int
    #: Aggregate shard meaningful-LOC.
    loc: int
    #: Shared overall objective every worker investigates within its shard.
    objective: str


@dataclass(frozen=True, slots=True)
class TeamPlan:
    """Coordinator-facing plan: assignments plus the manifest facts/notes.

    ``notes`` surfaces the partitioner's deterministic diagnostics (inventory
    warnings, oversized/streamed-file flags, conservative data skips) to the
    coordinator without burying them in logs.
    """

    requested_workers: int
    effective_workers: int
    total_weight: int
    assignments: tuple[TeamAssignment, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_workers": self.requested_workers,
            "effective_workers": self.effective_workers,
            "total_weight": self.total_weight,
            "notes": list(self.notes),
            "assignments": [
                {
                    "worker_id": assignment.worker_id,
                    "shard_id": assignment.shard_id,
                    "files": list(assignment.files),
                    "weight": assignment.weight,
                    "loc": assignment.loc,
                    "objective": assignment.objective,
                }
                for assignment in self.assignments
            ],
        }


@dataclass(frozen=True, slots=True)
class SpawnedWorker:
    """Result of one spawn attempt; a failed spawn carries ``error``."""

    worker_id: int
    shard_id: int
    name: str
    agent_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    """Deterministic completion record for one worker (pure data).

    ``cross_shard_refs`` carries the compact dependency/evidence references a
    worker recorded when it had to leave its shard - v1 only records them; the
    coordinator decides what to act on.
    """

    shard_id: int
    success: bool
    summary: str = ""
    findings: tuple[str, ...] = ()
    open_items: tuple[str, ...] = ()
    cross_shard_refs: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TeamResult:
    """Aggregated worker results, deterministic (ordered by shard id)."""

    results: tuple[WorkerOutcome, ...]
    succeeded: int
    failed: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "succeeded": self.succeeded,
            "failed": self.failed,
            "errors": list(self.errors),
            "workers": [
                {
                    "shard_id": outcome.shard_id,
                    "success": outcome.success,
                    "summary": outcome.summary,
                    "findings": list(outcome.findings),
                    "open_items": list(outcome.open_items),
                    "cross_shard_refs": list(outcome.cross_shard_refs),
                    "error": outcome.error,
                }
                for outcome in self.results
            ],
        }


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    """Order-preserving dedupe (deterministic for identical inputs)."""
    return tuple(dict.fromkeys(values))


def build_team_assignments(
    manifest: PartitionManifest,
    *,
    objective: str,
) -> tuple[TeamAssignment, ...]:
    """Map every effective shard to exactly one immutable assignment.

    Pure, deterministic: assignments come out in manifest shard order
    (already id-ordered and case-folded by the partitioner).  An empty
    manifest yields no assignments - never an error.
    """
    assignments: list[TeamAssignment] = []
    for worker_id, shard in enumerate(manifest.shards):
        assignments.append(
            TeamAssignment(
                worker_id=worker_id,
                shard_id=shard.shard_id,
                files=tuple(shard.files),
                weight=shard.weight,
                loc=shard.loc,
                objective=objective,
            )
        )
    return tuple(assignments)


def build_team_plan(
    manifest: PartitionManifest,
    *,
    objective: str,
) -> TeamPlan:
    """Assignments plus the coordinator facts (worker counts, notes)."""
    return TeamPlan(
        requested_workers=manifest.requested_workers,
        effective_workers=manifest.effective_workers,
        total_weight=manifest.total_weight,
        assignments=build_team_assignments(manifest, objective=objective),
        notes=tuple(manifest.notes),
    )


def build_worker_task_packet(
    assignment: TeamAssignment,
    *,
    known_facts: Sequence[str] = (),
    open_questions: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    do_not_repeat: Sequence[str] = (),
) -> dict[str, Any]:
    """Compact structured worker handoff (JSON-safe, deterministic keys).

    This is the worker's whole context: the compact packet replaces a raw
    parent-transcript dump.  Lists are deduplicated order-preserving.
    """
    return {
        "objective": assignment.objective,
        "shard_id": assignment.shard_id,
        "worker_id": assignment.worker_id,
        "files": list(assignment.files),
        "shard_weight": assignment.weight,
        "shard_loc": assignment.loc,
        "scope_constraint": WORKER_SCOPE_DIRECTIVE,
        "known_facts": list(_dedupe(known_facts)),
        "open_questions": list(_dedupe(open_questions)),
        "evidence_refs": list(_dedupe(evidence_refs)),
        "do_not_repeat": list(_dedupe((*do_not_repeat, *_DEFAULT_DO_NOT_REPEAT))),
    }


def render_worker_task(assignment: TeamAssignment, packet: dict[str, Any]) -> str:
    """Human-first, deterministic task text for the child's first message.

    The packet is embedded as canonical JSON (sorted keys) so the worker sees
    exactly the structured scope the coordinator recorded.
    """
    lines = [
        f"# Worker assignment - shard {assignment.shard_id}",
        "",
        packet["scope_constraint"],
        "",
        f"Objective: {assignment.objective}",
        "",
        f"Shard weight: {assignment.weight}  Shard LOC: {assignment.loc}",
        "",
        "Assigned files:",
        *(f"- {file}" for file in assignment.files),
        "",
        "Task packet (machine-readable):",
        json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False),
    ]
    return "\n".join(lines)


class TeamFanout:
    """Spawn one worker agent per team assignment, through an injected
    spawner (existing Strix child-agent primitive in production, a fake in
    tests).  Spawns are ordered by shard id and isolated per worker."""

    def __init__(
        self,
        plan: TeamPlan,
        spawn_worker: WorkerSpawner,
        *,
        skills: Sequence[str] = (),
        name_prefix: str = "worker",
    ) -> None:
        self._plan = plan
        self._spawn_worker = spawn_worker
        self._skills = tuple(skills)
        self._name_prefix = name_prefix

    async def spawn_all(
        self,
        *,
        known_facts: Sequence[str] = (),
        open_questions: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
        do_not_repeat: Sequence[str] = (),
    ) -> tuple[SpawnedWorker, ...]:
        """Spawn every assignment; per-worker failures are recorded, not fatal.

        An empty plan spawns nothing and returns ``()``.
        """
        spawned: list[SpawnedWorker] = []
        for assignment in self._plan.assignments:
            name = f"{self._name_prefix}-{assignment.shard_id}"
            packet = build_worker_task_packet(
                assignment,
                known_facts=known_facts,
                open_questions=open_questions,
                evidence_refs=evidence_refs,
                do_not_repeat=do_not_repeat,
            )
            task = render_worker_task(assignment, packet)
            try:
                result = await self._spawn_worker(
                    name=name,
                    task=task,
                    skills=list(self._skills),
                    # Deliberately empty: the packet is the handoff - do not
                    # duplicate the entire parent transcript per worker.
                    parent_history=[],
                )
            except Exception as exc:  # noqa: BLE001 - per-worker isolation.
                spawned.append(
                    SpawnedWorker(
                        worker_id=assignment.worker_id,
                        shard_id=assignment.shard_id,
                        name=name,
                        error=f"spawn failed: {exc}",
                    )
                )
                continue
            agent_id = result.get("agent_id")
            spawned.append(
                SpawnedWorker(
                    worker_id=assignment.worker_id,
                    shard_id=assignment.shard_id,
                    name=name,
                    agent_id=agent_id if isinstance(agent_id, str) else None,
                )
            )
        return tuple(spawned)


def aggregate_worker_results(
    outcomes: Sequence[WorkerOutcome],
) -> TeamResult:
    """Deterministic aggregation: sorted by shard id, failure-isolated."""
    ordered = tuple(sorted(outcomes, key=lambda outcome: outcome.shard_id))
    succeeded = sum(1 for outcome in ordered if outcome.success and outcome.error is None)
    failed = len(ordered) - succeeded
    errors = tuple(
        outcome.error or "worker reported failure"
        for outcome in ordered
        if outcome.error is not None or not outcome.success
    )
    return TeamResult(results=ordered, succeeded=succeeded, failed=failed, errors=errors)


def bind_worker_spawner(
    *,
    coordinator: Any,
    factory: Any,
    agents_db_path: Any,
    sessions_to_close: Any,
    run_config: Any,
    max_turns: int,
    interactive: bool,
    parent_ctx: dict[str, Any],
    event_sink: Any = None,
    hooks: Any = None,
) -> WorkerSpawner:
    """Bind the existing Strix child spawner for one team run.

    Mirrors the runner's own ``spawn_child_agent`` closure
    (``strix/core/runner.py``) and delegates to the exact same primitive
    (``strix.core.execution.spawn_child_agent``) that the ``create_agent``
    tool uses, so team workers inherit the scan's coordinator, factories,
    run config, model routing and supervision without a second runtime.

    Imported lazily to keep ``strix.team`` import-light (tests never touch
    the SDK path).
    """

    async def _spawn_worker(**kwargs: Any) -> dict[str, Any]:
        # Imported lazily so strix.team stays import-light.  The SDK-typed
        # primitive carries partially-unknown runner-scope types (mirrors
        # strix/core/runner.py's closure); route it through an explicit Any.
        import strix.core.execution as _execution  # noqa: PLC0415 - lazy binder import

        spawn_impl: Any = _execution.spawn_child_agent  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        result = await spawn_impl(
            coordinator=coordinator,
            factory=factory,
            agents_db_path=agents_db_path,
            sessions_to_close=sessions_to_close,
            run_config=run_config,
            max_turns=max_turns,
            interactive=interactive,
            parent_ctx=parent_ctx,
            event_sink=event_sink,
            hooks=hooks,
            **kwargs,
        )
        return dict(result)

    return _spawn_worker
