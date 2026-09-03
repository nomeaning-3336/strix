"""Deterministic team fan-out over source partitions - contract tests.

No paid scans, no model calls, no A/B runs: everything here runs against fake
worker/spawn implementations and the deterministic partitioner on tiny local
trees (``tmp_path``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from strix.agents.factory import registered_agent_tools
from strix.core.execution import spawn_child_agent
from strix.team import (
    WORKER_SCOPE_DIRECTIVE,
    SpawnedWorker,
    TeamFanout,
    TeamPlan,
    WorkerOutcome,
    aggregate_worker_results,
    build_team_assignments,
    build_team_plan,
    build_worker_task_packet,
    render_worker_task,
    validate_partition_manifest,
)
from strix.team.fanout import _DEFAULT_DO_NOT_REPEAT
from strix.tools.source_partition import partition_source
from strix.tools.source_partition.models import PartitionManifest, PartitionShard


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


OBJECTIVE = "Review the assigned source for security weaknesses relevant to the scan."

#: The worker scope directive the human approved - pinned verbatim so any
#: future rewording of WORKER_SCOPE_DIRECTIVE breaks a test.
EXPECTED_WORKER_SCOPE_DIRECTIVE = (
    "Your primary source scope is exactly this shard. "
    "Do not repeat broad repository discovery already completed by the coordinator. "
    "Follow dependencies outside the shard only when evidence from your assigned files "
    "requires it. "
    "Record any such boundary crossing explicitly."
)


def _code(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v{i} = {i}\n" for i in range(lines)), encoding="utf-8")


def _repo(root: Path) -> None:
    _code(root / "app" / "a.py", lines=40)
    _code(root / "app" / "b.py", lines=40)
    _code(root / "lib" / "c.py", lines=40)
    _code(root / "docs" / "guide.md", lines=8)


class _FakeSpawner:
    """Records every spawn call; optional per-shard failure injection."""

    def __init__(self, fail_shard: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_shard = fail_shard
        self.next_id = 0

    async def spawn(
        self, *, name: str, task: str, skills: list[str], parent_history: list[Any]
    ) -> dict[str, Any]:
        shard = int(name.rsplit("-", 1)[-1])
        if self.fail_shard == shard:
            raise RuntimeError(f"boom on shard {shard}")
        self.calls.append(
            {"name": name, "task": task, "skills": list(skills), "parent_history": parent_history}
        )
        self.next_id += 1
        return {"success": True, "agent_id": f"agent_{self.next_id}", "name": name}


def _plan(root: Path, *, workers: int) -> TeamPlan:
    manifest = partition_source([root], workers=workers)
    return build_team_plan(manifest, objective=OBJECTIVE)


async def _spawn_all(plan: TeamPlan, spawner: _FakeSpawner) -> tuple[SpawnedWorker, ...]:
    fanout = TeamFanout(plan, spawner.spawn)
    return await fanout.spawn_all()


# ---------------------------------------------------------------------------
# 1/3/4. Deterministic assignments, one per effective shard, empty => none
# ---------------------------------------------------------------------------


def test_assignments_deterministic_one_per_effective_shard(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    manifest = partition_source([root], workers=3)

    first = build_team_assignments(manifest, objective=OBJECTIVE)
    second = build_team_assignments(manifest, objective=OBJECTIVE)
    assert first == second
    assert len(first) == manifest.effective_workers == 3
    assert [assignment.worker_id for assignment in first] == [0, 1, 2]
    assert [assignment.shard_id for assignment in first] == [0, 1, 2]


def test_one_assignment_per_shard_and_no_swapping(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    manifest = partition_source([root], workers=2)
    assignments = build_team_assignments(manifest, objective=OBJECTIVE)
    assert len(assignments) == manifest.effective_workers == 2
    for assignment, shard in zip(assignments, manifest.shards, strict=True):
        assert assignment.shard_id == shard.shard_id
        assert assignment.files == shard.files
        assert assignment.weight == shard.weight
        assert assignment.loc == shard.loc
    # Shard 0 never receives shard 1's files (and vice versa).
    assert set(assignments[0].files).isdisjoint(assignments[1].files)


def test_fewer_effective_than_requested_workers(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    plan = _plan(root, workers=9)  # 4 files only -> 4 effective workers
    assert plan.requested_workers == 9
    assert plan.effective_workers == 4
    assert len(plan.assignments) == 4


def test_empty_manifest_spawns_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    plan = _plan(empty, workers=4)
    assert plan.requested_workers == 4
    assert plan.effective_workers == 0
    assert plan.assignments == ()

    spawner = _FakeSpawner()
    spawned = _run_async(_spawn_all(plan, spawner))
    assert spawned == ()
    assert spawner.calls == []


def _run_async(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 5. No duplicate / missing source paths
# ---------------------------------------------------------------------------


def test_no_duplicate_or_missing_source_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    manifest = partition_source([root], workers=3)
    plan = _plan(root, workers=3)

    listed = [file for assignment in plan.assignments for file in assignment.files]
    assert len(listed) == len(set(listed))  # no duplicates
    shard_files = {file for shard in manifest.shards for file in shard.files}
    assert set(listed) == shard_files  # every manifest file represented
    assert set(listed) == set(manifest.file_to_shard)


# ---------------------------------------------------------------------------
# 6/7. Objective + compact task packet
# ---------------------------------------------------------------------------


def test_root_objective_propagated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    plan = _plan(root, workers=2)
    assert all(assignment.objective == OBJECTIVE for assignment in plan.assignments)
    for assignment in plan.assignments:
        packet = build_worker_task_packet(assignment)
        assert packet["objective"] == OBJECTIVE
        assert OBJECTIVE in render_worker_task(assignment, packet)


def test_compact_task_packet_shape_and_determinism(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    (assignment,) = _plan(root, workers=1).assignments

    packet = build_worker_task_packet(
        assignment,
        known_facts=("fact-1", "fact-1"),
        open_questions=("q-1",),
        evidence_refs=("ref-1",),
        do_not_repeat=("custom-1",),
    )
    # Every required logical key is present and JSON-safe.
    for key in (
        "objective",
        "shard_id",
        "worker_id",
        "files",
        "shard_weight",
        "shard_loc",
        "scope_constraint",
        "known_facts",
        "open_questions",
        "evidence_refs",
        "do_not_repeat",
    ):
        assert key in packet
    json.dumps(packet, sort_keys=True)  # must not raise
    assert packet["scope_constraint"] == WORKER_SCOPE_DIRECTIVE
    assert packet["known_facts"] == ["fact-1"]  # order-preserving dedupe
    assert packet["do_not_repeat"] == ["custom-1", *_DEFAULT_DO_NOT_REPEAT]
    assert packet["shard_weight"] == assignment.weight
    assert packet["files"] == list(assignment.files)

    rebuilt = build_worker_task_packet(
        assignment,
        known_facts=("fact-1", "fact-1"),
        open_questions=("q-1",),
        evidence_refs=("ref-1",),
        do_not_repeat=("custom-1",),
    )
    assert json.dumps(packet, sort_keys=True) == json.dumps(rebuilt, sort_keys=True)


def test_worker_scope_directive_wording_is_pinned() -> None:
    assert WORKER_SCOPE_DIRECTIVE == EXPECTED_WORKER_SCOPE_DIRECTIVE


# ---------------------------------------------------------------------------
# 8. Spawn uses exact scope + empty parent history
# ---------------------------------------------------------------------------


def test_spawn_packet_task_and_no_parent_transcript(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    plan = _plan(root, workers=2)
    spawner = _FakeSpawner()
    spawned = _run_async(_spawn_all(plan, spawner))

    assert len(spawned) == 2
    assert [item.name for item in spawned] == ["worker-0", "worker-1"]
    assert len(spawner.calls) == 2
    marker = "Task packet (machine-readable):\n"
    for assignment, call in zip(plan.assignments, spawner.calls, strict=True):
        # Scope + files are in the task text; no parent transcript was dumped.
        assert call["name"] == f"worker-{assignment.shard_id}"
        assert WORKER_SCOPE_DIRECTIVE in call["task"]
        for file in assignment.files:
            assert file in call["task"]
        assert call["parent_history"] == []
        assert call["skills"] == []
        # The task embeds the canonical JSON packet.
        payload = call["task"].split(marker, 1)[1]
        assert json.loads(payload)["shard_id"] == assignment.shard_id
        assert json.loads(payload)["files"] == list(assignment.files)


# ---------------------------------------------------------------------------
# 9/10. Aggregation + failure isolation
# ---------------------------------------------------------------------------


def test_completion_aggregation_is_deterministic() -> None:
    outcomes = [
        WorkerOutcome(shard_id=2, success=True, summary="two", findings=("f2",)),
        WorkerOutcome(shard_id=0, success=True, summary="zero"),
        WorkerOutcome(shard_id=1, success=False, summary="one-broken", error="stuck"),
        WorkerOutcome(shard_id=3, success=False, summary="unresolved"),
    ]
    first = aggregate_worker_results(outcomes)
    second = aggregate_worker_results(list(reversed(outcomes)))
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert [outcome.shard_id for outcome in first.results] == [0, 1, 2, 3]
    assert first.succeeded == 2
    assert first.failed == 2
    assert first.errors == ("stuck", "worker reported failure")
    json.dumps(first.to_dict(), sort_keys=True)


def test_one_spawn_failure_does_not_corrupt_others(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    plan = _plan(root, workers=3)
    spawner = _FakeSpawner(fail_shard=1)

    spawned = _run_async(_spawn_all(plan, spawner))

    assert len(spawned) == 3
    assert spawned[0].agent_id is not None
    assert spawned[1].error is not None and "boom on shard 1" in spawned[1].error
    assert spawned[1].agent_id is None
    assert spawned[2].agent_id is not None
    # The other two workers still spawned with their own exact scope.
    assert {call["name"] for call in spawner.calls} == {"worker-0", "worker-2"}
    # No exception leaked, and no error object mutated a sibling.
    assert isinstance(spawned[1], SpawnedWorker)


# ---------------------------------------------------------------------------
# 11. Workers cannot mutate the manifest through their assignment
# ---------------------------------------------------------------------------


def test_assignment_is_immutable_and_manifest_independent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    manifest = partition_source([root], workers=1)
    assignment = build_team_assignments(manifest, objective=OBJECTIVE)[0]

    # The assignment is frozen...
    with pytest.raises(FrozenInstanceError):
        assignment.files = ()  # type: ignore[misc]
    # ...has no back-reference to the manifest...
    with pytest.raises(AttributeError):
        getattr(assignment, "manifest")  # noqa: B009
    # ...and mutating the manifest's file list afterwards cannot change it.
    shard_files = list(manifest.shards[0].files)
    shard_files.append("injected.py")
    assert "injected.py" not in assignment.files
    assert assignment.files == tuple(shard_files[:-1])
    # The worker packet is plain data - no manifest object travels with it.
    assert "manifest" not in build_worker_task_packet(assignment)


# ---------------------------------------------------------------------------
# 12. Existing subagent spawning primitives are untouched and compatible
# ---------------------------------------------------------------------------


def test_fanout_spawner_shape_matches_existing_child_spawner() -> None:
    signature = inspect.signature(spawn_child_agent)
    for keyword in ("name", "task", "skills", "parent_history"):
        assert keyword in signature.parameters
    assert "parent_ctx" in signature.parameters


def test_no_model_visible_tool_added() -> None:
    names = {getattr(tool, "name", repr(tool)) for tool in registered_agent_tools()}
    assert not any("team" in name or "fanout" in name or "shard" in name for name in names)


# ---------------------------------------------------------------------------
# 13. Manifest validation at the fan-out boundary (hand-built manifests)
# ---------------------------------------------------------------------------


def _hand_manifest(*, shards: list[PartitionShard], effective_workers: int) -> PartitionManifest:
    file_to_shard = {file: shard.shard_id for shard in shards for file in shard.files}
    return PartitionManifest(
        requested_workers=effective_workers,
        effective_workers=effective_workers,
        total_weight=sum(shard.weight for shard in shards),
        total_loc=sum(shard.loc for shard in shards),
        shards=tuple(shards),
        file_to_shard=file_to_shard,
        notes=(),
    )


def _shard(shard_id: int, files: list[str], weight: int = 1, loc: int = 1) -> PartitionShard:
    return PartitionShard(shard_id=shard_id, files=tuple(files), weight=weight, loc=loc)


def test_manifest_validation_rejects_duplicate_shard_ids() -> None:
    manifest = _hand_manifest(
        shards=[_shard(0, ["a.py"]), _shard(0, ["b.py"])], effective_workers=2
    )
    with pytest.raises(ValueError, match="duplicate shard ids"):
        validate_partition_manifest(manifest)
    with pytest.raises(ValueError, match="duplicate shard ids"):
        build_team_assignments(manifest, objective=OBJECTIVE)


def test_manifest_validation_rejects_effective_mismatch() -> None:
    manifest = _hand_manifest(shards=[_shard(0, ["a.py"])], effective_workers=2)
    with pytest.raises(ValueError, match="effective_workers=2"):
        validate_partition_manifest(manifest)
    with pytest.raises(ValueError, match="effective_workers=2"):
        build_team_assignments(manifest, objective=OBJECTIVE)


def test_manifest_validation_rejects_gapped_shard_ids() -> None:
    manifest = _hand_manifest(
        shards=[_shard(0, ["a.py"]), _shard(2, ["b.py"])], effective_workers=2
    )
    with pytest.raises(ValueError, match="contiguous"):
        validate_partition_manifest(manifest)


def test_manifest_validation_rejects_duplicate_file_across_shards() -> None:
    manifest = _hand_manifest(
        shards=[_shard(0, ["a.py", "dup.py"]), _shard(1, ["dup.py"])], effective_workers=2
    )
    with pytest.raises(ValueError, match=r"dup\.py"):
        validate_partition_manifest(manifest)
    with pytest.raises(ValueError, match=r"dup\.py"):
        build_team_assignments(manifest, objective=OBJECTIVE)


def test_manifest_validation_valid_manifest_flows_through() -> None:
    manifest = _hand_manifest(
        shards=[_shard(0, ["a.py"]), _shard(1, ["b.py"])], effective_workers=2
    )
    validate_partition_manifest(manifest)  # must not raise
    assignments = build_team_assignments(manifest, objective=OBJECTIVE)
    assert len(assignments) == 2
    plan = build_team_plan(manifest, objective=OBJECTIVE)
    spawner = _FakeSpawner()
    TeamFanout(plan, spawner.spawn)  # boundary validation accepts a valid plan
    spawned = _run_async(_spawn_all(plan, spawner))
    assert len(spawned) == 2
