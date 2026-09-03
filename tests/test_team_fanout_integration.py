"""Team fan-out wired into source scans - deterministic integration tests.

These tests exercise the orchestration seam (``strix.team.integration``) and
the runner-facing adapter shape with fake spawn workers and tiny ``tmp_path``
trees only: no model calls, no network, no paid scans, no live runner.
``STRIX_TEAM_WIDTH`` gating is tested through the settings loader exactly like
other env knobs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pytest

from strix.config import loader
from strix.config.loader import load_settings
from strix.team import SpawnedWorker, TeamFanout, build_team_plan
from strix.team.fanout import WORKER_SCOPE_DIRECTIVE
from strix.team.integration import (
    TeamStageOutcome,
    _is_successful_spawn,
    build_root_team_handoff,
    log_team_stage_outcome,
    stage_source_team_fanout,
)
from strix.tools.source_partition import PartitionConfig, partition_source
from strix.tools.source_partition.models import PartitionManifest, PartitionShard


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

OBJECTIVE = "Investigate the assigned shard for security weaknesses."

#: Root-handoff directive fragments the human approved - pinned so future
#: rewording of the handoff breaks a test.
EXPECTED_HANDOFF_FRAGMENT_SUCCESS = (
    "Do not redo broad source discovery or spawn overlapping source workers."
)
EXPECTED_HANDOFF_FRAGMENT_PARTIAL = (
    "Do not duplicate work assigned to active workers.\n"
    "Explicitly cover or reassign any uncovered shards before concluding source review."
)


class _MalformedSpawner:
    """Fake spawner that returns a "success" payload with a bad agent_id."""

    def __init__(self, agent_id_value: Any) -> None:
        self.agent_id_value = agent_id_value
        self.calls: list[dict[str, Any]] = []
        self.count = 0

    async def spawn(
        self, *, name: str, task: str, skills: list[str], parent_history: list[Any]
    ) -> dict[str, Any]:
        self.count += 1
        self.calls.append(
            {"name": name, "task": task, "skills": list(skills), "parent_history": parent_history}
        )
        return {"success": True, "agent_id": self.agent_id_value, "name": name}


def _code(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v{i} = {i}\n" for i in range(lines)), encoding="utf-8")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Recorder:
    """Sentinel fake spawner: records every call; optional per-shard/all-fail."""

    def __init__(self, fail_shard: int | None = None, fail_all: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_shard = fail_shard
        self.fail_all = fail_all
        self.count = 0

    async def spawn(
        self, *, name: str, task: str, skills: list[str], parent_history: list[Any]
    ) -> dict[str, Any]:
        shard = int(name.rsplit("-", 1)[-1])
        if self.fail_all or self.fail_shard == shard:
            raise RuntimeError("boom")
        self.count += 1
        self.calls.append(
            {"name": name, "task": task, "skills": list(skills), "parent_history": parent_history}
        )
        return {"success": True, "agent_id": f"agent-{self.count}", "name": name}


def _source_repo(root: Path) -> None:
    for index in range(6):
        _code(root / f"pkg{index}" / "code.py", lines=30)


def _stage(
    *,
    roots: list[Path],
    recorder: _Recorder,
    width: int | None,
    config: PartitionConfig | None = None,
    manifest: PartitionManifest | None = None,
    worker_skills: tuple[str, ...] = (),
) -> TeamStageOutcome:
    local_sources = [{"source_path": str(root)} for root in roots]
    return _run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=local_sources,
            spawn_worker=recorder.spawn,
            objective=OBJECTIVE,
            team_width=width,
            worker_skills=worker_skills,
            config=config,
            manifest=manifest,
        )
    )


def _shard(shard_id: int, files: list[str]) -> PartitionShard:
    return PartitionShard(shard_id=shard_id, files=tuple(files), weight=1, loc=1)


def _hand_manifest(
    shards: list[PartitionShard], *, effective_workers: int | None = None
) -> PartitionManifest:
    effective = len(shards) if effective_workers is None else effective_workers
    return PartitionManifest(
        requested_workers=effective,
        effective_workers=effective,
        total_weight=sum(shard.weight for shard in shards),
        total_loc=sum(shard.loc for shard in shards),
        shards=tuple(shards),
        file_to_shard={file: shard.shard_id for shard in shards for file in shard.files},
        notes=(),
    )


def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STRIX_TEAM_WIDTH", raising=False)
    monkeypatch.setattr(loader, "_cached", None)
    monkeypatch.setattr(loader, "_override", None)


# ---------------------------------------------------------------------------
# 1/11. STRIX_TEAM_WIDTH default (1) preserves the legacy single-agent flow
# ---------------------------------------------------------------------------


def test_team_width_defaults_to_one_and_no_workers_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_settings(monkeypatch)
    assert load_settings().team.team_width == 1

    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=1)
    assert outcome.enabled is True
    assert outcome.plan is not None
    # Inventory + partition ran (manifest validated, notes reachable)...
    assert outcome.plan.effective_workers >= 1
    assert outcome.spawned == ()
    assert outcome.attempted == 0
    assert outcome.successfully_spawned == 0
    assert outcome.reason == "width_1"
    assert recorder.count == 0  # ...but no team workers fire at width=1.


def test_settings_env_override_strix_team_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reset_settings(monkeypatch)
    monkeypatch.setenv("STRIX_TEAM_WIDTH", "3")
    assert load_settings().team.team_width == 3

    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=None)
    assert outcome.team_width == 3
    assert outcome.plan is not None
    assert outcome.successfully_spawned >= 1


# ---------------------------------------------------------------------------
# 2. width>1: one SpawnedWorker per effective shard, no dup/missing shards
# ---------------------------------------------------------------------------


def test_team_width_above_one_spawns_per_effective_shard(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    # Manifest computed out-of-band so we can compare 1:1.
    manifest = partition_source([root], workers=3)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=3)

    assert outcome.plan is not None
    assert outcome.plan.effective_workers == manifest.effective_workers
    assert len(outcome.spawned) == manifest.effective_workers
    assert outcome.attempted == manifest.effective_workers
    assert outcome.successfully_spawned == manifest.effective_workers
    assert outcome.failed_to_spawn == 0
    assert outcome.reason == "spawned"
    assert recorder.count == manifest.effective_workers
    spawned_shards = sorted(worker.shard_id for worker in outcome.spawned)
    assert spawned_shards == sorted(shard.shard_id for shard in manifest.shards)
    assert len(set(spawned_shards)) == len(spawned_shards)  # no duplicates


# ---------------------------------------------------------------------------
# 3. Non-source / DAST scan: clean no-op (no roots)
# ---------------------------------------------------------------------------


def test_non_source_scan_is_a_clean_noop() -> None:
    recorder = _Recorder()
    outcome = _stage(roots=[], recorder=recorder, width=3)
    assert outcome.enabled is False
    assert outcome.plan is None
    assert outcome.spawned == ()
    assert outcome.attempted == 0
    assert outcome.successfully_spawned == 0
    assert outcome.reason == "no_source_roots"
    assert recorder.count == 0


# ---------------------------------------------------------------------------
# 4. Zero useful units => zero team workers
# ---------------------------------------------------------------------------


def test_zero_units_zero_workers(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    recorder = _Recorder()
    outcome = _stage(roots=[empty], recorder=recorder, width=3)
    assert outcome.enabled is True
    assert outcome.plan is not None
    assert outcome.plan.effective_workers == 0
    assert outcome.reason == "zero_units"
    assert outcome.spawned == ()
    assert outcome.attempted == 0
    assert outcome.successfully_spawned == 0
    assert recorder.count == 0


# ---------------------------------------------------------------------------
# 5. Exact approved scope directive reaches every child task
# ---------------------------------------------------------------------------


def test_approved_scope_directive_reaches_workers(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    assert outcome.plan is not None
    assert len(recorder.calls) == outcome.successfully_spawned
    for call in recorder.calls:
        assert WORKER_SCOPE_DIRECTIVE in call["task"]
        # The human-approved sentences are present verbatim.
        assert "evidence from your assigned files requires it." in call["task"]
        assert "Record any such boundary crossing explicitly." in call["task"]


# ---------------------------------------------------------------------------
# 6. Root objective propagated compactly (packet, not a parent transcript)
# ---------------------------------------------------------------------------


def test_objective_propagated_compactly(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=2)
    assert outcome.plan is not None
    marker = "Task packet (machine-readable):\n"
    for assignment, call in zip(outcome.plan.assignments, recorder.calls, strict=True):
        assert call["parent_history"] == []  # compact handoff, no transcript dump
        packet = json.loads(call["task"].split(marker, 1)[1])
        assert packet["objective"] == OBJECTIVE
        assert packet["shard_id"] == assignment.shard_id
        assert packet["files"] == list(assignment.files)
        assert set(packet) == {
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
        }


# ---------------------------------------------------------------------------
# 7. Manifest diagnostics (notes) stay visible at the integration site
# ---------------------------------------------------------------------------


def test_manifest_notes_surfaced_on_plan(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    _code(root / "big.py", lines=60)  # > 64 bytes -> streamed with a tiny threshold
    config = PartitionConfig(max_file_bytes=64)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=2, config=config)
    assert outcome.plan is not None
    assert any("oversized file counted by streaming LOC" in note for note in outcome.plan.notes)


# ---------------------------------------------------------------------------
# 8. Malformed manifests fail before any spawn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        _hand_manifest([_shard(0, ["a.py"]), _shard(0, ["b.py"])]),  # duplicate shard ids
        _hand_manifest([_shard(0, ["a.py"])], effective_workers=2),  # eff mismatch
        _hand_manifest([_shard(0, ["a.py", "dup.py"]), _shard(1, ["dup.py"])]),  # dup file
        _hand_manifest([_shard(0, ["a.py"]), _shard(2, ["b.py"])]),  # gapped ids
    ],
    ids=["dup-ids", "eff-mismatch", "dup-file", "gapped-ids"],
)
def test_malformed_manifest_fails_before_spawn(tmp_path: Path, bad: PartitionManifest) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    with pytest.raises(ValueError):
        _stage(roots=[root], recorder=recorder, width=2, manifest=bad)
    assert recorder.count == 0  # nothing was ever spawned


# ---------------------------------------------------------------------------
# 9. Legacy spawner call shape unchanged (four worker kwargs)
# ---------------------------------------------------------------------------


def test_worker_spawn_kwargs_match_existing_child_spawner_shape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    _stage(roots=[root], recorder=recorder, width=2)
    assert recorder.calls
    for call in recorder.calls:
        assert set(call) == {"name", "task", "skills", "parent_history"}
        assert isinstance(call["task"], str)
        assert call["skills"] == []
        assert call["parent_history"] == []


# ---------------------------------------------------------------------------
# Real-spawner adapter: parent_ctx injection at the runner boundary
# ---------------------------------------------------------------------------


def test_runner_adapter_injects_parent_ctx_into_real_spawner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strix.core import execution as execution_module  # noqa: PLC0415

    # Mimic strix.core.execution.spawn_child_agent's mandatory kw-only
    # signature (parent_ctx required) without touching the real runtime.
    recorded: list[dict[str, Any]] = []

    async def fake_execution(
        *,
        parent_ctx: dict[str, Any],
        name: str,
        task: str,
        skills: list[str],
        parent_history: list[Any],
        **_rest: Any,
    ) -> dict[str, Any]:
        recorded.append(
            {
                "parent_ctx": parent_ctx,
                "name": name,
                "task": task,
                "skills": list(skills),
                "parent_history": parent_history,
            }
        )
        return {"success": True, "agent_id": f"agent:{name}", "name": name}

    monkeypatch.setattr(execution_module, "spawn_child_agent", fake_execution)
    context = {"agent_id": "root", "marker": "runner-parent-ctx"}

    async def runner_inner(**spawn_kwargs: Any) -> dict[str, Any]:
        # Mirrors the runner's own spawn_child_agent closure: fixed runner-scope
        # kwargs are bound here; the underlying call is the (patched) module fn.
        return await execution_module.spawn_child_agent(
            coordinator=None,
            factory=None,
            agents_db_path=None,
            sessions_to_close=[],
            run_config=None,
            max_turns=5,
            interactive=False,
            **spawn_kwargs,
        )

    async def team_adapter(**spawn_kwargs: Any) -> dict[str, Any]:
        # The runner-built adapter injects parent_ctx at the boundary.
        return await runner_inner(parent_ctx=context, **spawn_kwargs)

    # Regression: TeamFanout's four worker kwargs alone cannot satisfy the real
    # spawner - without the adapter, parent_ctx is missing.
    with pytest.raises(TypeError, match="parent_ctx"):
        _run(runner_inner(name="worker-0", task="t", skills=[], parent_history=[]))

    root = tmp_path / "repo"
    _source_repo(root)
    plan = build_team_plan(partition_source([root], workers=2), objective=OBJECTIVE)
    fanout = TeamFanout(plan, team_adapter)
    spawned = _run(fanout.spawn_all())

    assert len(spawned) == 2
    assert len(recorded) == 2
    for call in recorded:
        assert call["parent_ctx"] is context
        assert set(call) == {"parent_ctx", "name", "task", "skills", "parent_history"}
        assert call["parent_history"] == []


# ---------------------------------------------------------------------------
# Success accounting: attempted vs successfully spawned vs failed
# ---------------------------------------------------------------------------


def test_partial_spawn_failure_accounting(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder(fail_shard=1)
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    assert outcome.attempted == 3
    assert outcome.successfully_spawned == 2
    assert outcome.failed_to_spawn == 1
    assert outcome.reason == "partial_spawn_failure"
    by_shard = {worker.shard_id: worker for worker in outcome.spawned}
    assert by_shard[1].error is not None
    assert by_shard[0].error is None
    assert by_shard[2].error is None


def test_all_spawn_failures_accounting(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder(fail_all=True)
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    assert outcome.attempted == 3
    assert outcome.successfully_spawned == 0
    assert outcome.failed_to_spawn == 3
    assert outcome.reason == "spawn_failed"
    assert all(worker.error is not None for worker in outcome.spawned)


# ---------------------------------------------------------------------------
# Runner logging never claims failed attempts as spawned
# ---------------------------------------------------------------------------


def test_runner_logging_uses_successful_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    plan = build_team_plan(partition_source([root], workers=2), objective=OBJECTIVE)

    all_ok = TeamStageOutcome(
        enabled=True,
        team_width=2,
        plan=plan,
        spawned=(),
        reason="spawned",
        attempted=2,
        successfully_spawned=2,
        failed_to_spawn=0,
    )
    partial = TeamStageOutcome(
        enabled=True,
        team_width=3,
        plan=plan,
        spawned=(),
        reason="partial_spawn_failure",
        attempted=3,
        successfully_spawned=2,
        failed_to_spawn=1,
    )
    all_failed = TeamStageOutcome(
        enabled=True,
        team_width=3,
        plan=plan,
        spawned=(),
        reason="spawn_failed",
        attempted=3,
        successfully_spawned=0,
        failed_to_spawn=3,
    )
    legacy = TeamStageOutcome(enabled=True, team_width=1, plan=plan, spawned=(), reason="width_1")

    with caplog.at_level(logging.INFO, logger="strix.team.integration"):
        log_team_stage_outcome(all_ok, scan_id="s1")
        log_team_stage_outcome(partial, scan_id="s2")
        log_team_stage_outcome(all_failed, scan_id="s3")
        log_team_stage_outcome(legacy, scan_id="s4")

    text = caplog.text
    assert "team fan-out: spawned 2 worker(s) for scan s1" in text
    assert "team fan-out: spawned 2 worker(s), 1 failed to spawn for scan s2" in text
    assert "team fan-out: no workers spawned (3 attempt(s) failed) for scan s3" in text
    assert "team fan-out: no-op (reason=width_1) for scan s4" in text
    # Never report failed attempts as spawned.
    assert "spawned 3 worker(s)" not in text


# ---------------------------------------------------------------------------
# Root handoff: success/partial layouts, uncovered shards, pinned prose
# ---------------------------------------------------------------------------


def test_handoff_all_success_has_no_uncovered_section(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    assert outcome.successfully_spawned == 3
    handoff = build_root_team_handoff(outcome.plan, outcome.spawned)
    assert handoff is not None
    assert "Workers:" in handoff
    assert "Active workers:" not in handoff
    assert "Uncovered shards:" not in handoff
    assert EXPECTED_HANDOFF_FRAGMENT_SUCCESS in handoff
    assert handoff.index("worker-0") < handoff.index("worker-1") < handoff.index("worker-2")
    for worker in outcome.spawned:
        assert worker.name in handoff
        assert f"shard {worker.shard_id}" in handoff
        assert worker.agent_id in handoff


def test_handoff_partial_lists_active_and_uncovered_shards(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder(fail_shard=1)
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    assert outcome.reason == "partial_spawn_failure"
    handoff = build_root_team_handoff(outcome.plan, outcome.spawned)
    assert handoff is not None
    assert "Active workers:" in handoff
    assert "Uncovered shards:" in handoff
    # The uncovered shard appears exactly once.
    assert handoff.count("- shard 1 → worker failed to spawn") == 1
    # Active workers are the successful ones, ordered by shard id.
    assert "\n- worker-0 (agent-1) → shard 0" in handoff
    assert "\n- worker-2 (agent-2) → shard 2" in handoff
    assert "\n- worker-1 " not in handoff  # the failed worker has no active line
    # Both partial-failure directives are pinned present.
    assert EXPECTED_HANDOFF_FRAGMENT_PARTIAL in handoff


def test_handoff_partial_successful_shards_not_uncovered(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder(fail_shard=1)
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    handoff = build_root_team_handoff(outcome.plan, outcome.spawned)
    assert handoff is not None
    uncovered_section = handoff.split("Uncovered shards:", 1)[1]
    assert "worker-0" not in uncovered_section
    assert "worker-2" not in uncovered_section
    assert "shard 0" not in uncovered_section
    assert "shard 2" not in uncovered_section
    assert uncovered_section.count("→ worker failed to spawn") == 1


def test_handoff_none_when_nothing_spawned(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder(fail_all=True)
    outcome = _stage(roots=[root], recorder=recorder, width=3)
    assert outcome.successfully_spawned == 0
    assert build_root_team_handoff(outcome.plan, outcome.spawned) is None


# ---------------------------------------------------------------------------
# Centralized success predicate: malformed success payloads are failures
# ---------------------------------------------------------------------------


def test_malformed_agent_id_counts_as_spawn_failure(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    spawner = _MalformedSpawner("")
    outcome = _run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=[{"source_path": str(root)}],
            spawn_worker=spawner.spawn,
            objective=OBJECTIVE,
            team_width=3,
        )
    )
    assert outcome.attempted == 3
    assert outcome.successfully_spawned == 0
    assert outcome.failed_to_spawn == 3
    assert outcome.reason == "spawn_failed"
    # Retained in the tuple for observability, but counted as failed.
    assert len(outcome.spawned) == 3
    assert all(worker.error is None and worker.agent_id == "" for worker in outcome.spawned)
    # No team handoff: the legacy root path handles the scan.
    assert build_root_team_handoff(outcome.plan, outcome.spawned) is None


@pytest.mark.parametrize("bad_agent_id", ["", None, 123], ids=["empty", "none", "nonstring"])
def test_malformed_success_payloads_are_failures(tmp_path: Path, bad_agent_id: Any) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    spawner = _MalformedSpawner(bad_agent_id)
    outcome = _run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=[{"source_path": str(root)}],
            spawn_worker=spawner.spawn,
            objective=OBJECTIVE,
            team_width=3,
        )
    )
    assert outcome.successfully_spawned == 0
    assert outcome.failed_to_spawn == outcome.attempted
    assert outcome.reason == "spawn_failed"
    assert build_root_team_handoff(outcome.plan, outcome.spawned) is None


def test_counters_reason_and_handoff_share_one_predicate(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    spawner = _MalformedSpawner("")
    outcome = _run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=[{"source_path": str(root)}],
            spawn_worker=spawner.spawn,
            objective=OBJECTIVE,
            team_width=3,
        )
    )
    expected_failed = sum(1 for worker in outcome.spawned if not _is_successful_spawn(worker))
    assert outcome.failed_to_spawn == expected_failed
    assert outcome.successfully_spawned == outcome.attempted - expected_failed
    if outcome.successfully_spawned == 0:
        assert outcome.reason == "spawn_failed"
    assert (build_root_team_handoff(outcome.plan, outcome.spawned) is None) == (
        outcome.successfully_spawned == 0
    )


def test_is_successful_spawn_predicate_cases() -> None:
    def worker(**overrides: Any) -> SpawnedWorker:
        fields: dict[str, Any] = {
            "worker_id": 0,
            "shard_id": 0,
            "name": "worker-0",
            "agent_id": "agent-1",
        }
        fields.update(overrides)
        return SpawnedWorker(**fields)

    assert _is_successful_spawn(worker()) is True  # success case
    assert _is_successful_spawn(worker(error="boom")) is False
    assert _is_successful_spawn(worker(agent_id="")) is False
    assert _is_successful_spawn(worker(agent_id=None)) is False
    assert _is_successful_spawn(worker(agent_id=123)) is False


# ---------------------------------------------------------------------------
# Skills propagate to workers; parent_history stays empty
# ---------------------------------------------------------------------------


def test_worker_skills_propagate_and_parent_history_empty(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _source_repo(root)
    recorder = _Recorder()
    _stage(roots=[root], recorder=recorder, width=2, worker_skills=("skill-a", "skill-b"))
    assert recorder.calls
    for call in recorder.calls:
        assert call["skills"] == ["skill-a", "skill-b"]
        assert call["parent_history"] == []


# ---------------------------------------------------------------------------
# create_agent (model-driven arbitrary subagent spawning) stays unchanged
# ---------------------------------------------------------------------------


def test_create_agent_signature_unchanged() -> None:
    from strix.tools.agents_graph.tools import create_agent  # noqa: PLC0415

    # create_agent is an SDK FunctionTool; its JSON schema is the model-facing
    # signature. The legacy tool still takes the four original arguments and
    # no team-fan-out kwargs.
    schema = create_agent.params_json_schema
    properties = schema.get("properties", {})
    assert {"name", "task", "inherit_context", "skills"} <= set(properties)
    assert not set(properties) & {"shard_id", "worker_id", "objective", "parent_ctx"}
