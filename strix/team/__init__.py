"""Deterministic team fan-out over source partitions (library API).

See ``strix.team.fanout`` for the full documentation.  This package is a
pure orchestration layer: it shapes ``PartitionManifest`` shards into
immutable worker assignments and reuses the existing Strix child-agent
spawner - it adds no agent runtime, no model-visible tools, and no
``tool_policy`` entries.
"""

from __future__ import annotations

from strix.team.fanout import (
    WORKER_SCOPE_DIRECTIVE,
    SpawnedWorker,
    TeamAssignment,
    TeamFanout,
    TeamPlan,
    TeamResult,
    WorkerOutcome,
    WorkerSpawner,
    aggregate_worker_results,
    bind_worker_spawner,
    build_team_assignments,
    build_team_plan,
    build_worker_task_packet,
    render_worker_task,
)


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
