"""Passive team fan-out telemetry - deterministic tests with fakes only.

No real coordinator / report state / model / network / paid scans: every data
authority is a fake, and the ``team_fanout`` section is asserted purely as a
data shape.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from strix.agents.factory import registered_agent_tools
from strix.team.integration import TeamStageOutcome, _is_successful_spawn, stage_source_team_fanout
from strix.team.telemetry import (
    FINDING_STATE_KEYS,
    TEAM_FANOUT_KEY,
    build_stage_section,
    finalize_and_persist,
    finalize_team_section,
    record_stage,
)


OBJECTIVE = "Review the assigned shard for security weaknesses."


def _code(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v{i} = {i}\n" for i in range(lines)), encoding="utf-8")


def _repo(root: Path) -> None:
    for index in range(6):
        _code(root / f"pkg{index}" / "code.py", lines=30)


class _FakeReportState:
    def __init__(
        self,
        *,
        usage_agents: list[dict[str, Any]] | None = None,
        vulnerabilities: list[dict[str, Any]] | None = None,
    ) -> None:
        self.run_record: dict[str, Any] = {"llm_usage": {"agents": list(usage_agents or [])}}
        self.vulnerability_reports = list(vulnerabilities or [])
        self.saves: list[dict[str, Any]] = []

    def save_run_data(self) -> None:
        self.saves.append(json.loads(json.dumps(self.run_record)))

    def get_existing_vulnerabilities(self) -> list[dict[str, Any]]:
        return list(self.vulnerability_reports)


class _FakeCoordinator:
    def __init__(self, snapshots: dict[str, dict[str, Any]], fail: bool = False) -> None:
        self.snapshots = snapshots
        self.fail = fail

    def agent_metrics_snapshot(self, agent_id: str) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("coordinator exploded")
        return dict(self.snapshots.get(agent_id, {"status": "unknown"}))


class _Spawner:
    """Async fake: fail on a shard, or return a chosen agent_id."""

    def __init__(self, fail_shard: int | None = None, agent_id: Any = "agent") -> None:
        self.fail_shard = fail_shard
        self.agent_id = agent_id
        self.count = 0

    async def spawn(
        self, *, name: str, task: str, skills: list[str], parent_history: list[Any]
    ) -> dict[str, Any]:
        shard = int(name.rsplit("-", 1)[-1])
        self.count += 1
        if self.fail_shard == shard:
            raise RuntimeError("boom")
        agent = f"agent-{self.count}" if self.agent_id == "agent" else self.agent_id
        return {
            "success": True,
            "agent_id": agent,
            "name": name,
            "task_len": len(task),
            "skills_len": len(skills),
            "history_len": len(parent_history),
        }


def _stage_outcome(root: Path, *, width: int, spawner: Any) -> TeamStageOutcome:
    return asyncio.run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=[{"source_path": str(root)}],
            spawn_worker=spawner.spawn,
            objective=OBJECTIVE,
            team_width=width,
        )
    )


def _finding(agent_id: str, state: str, title: str = "x") -> dict[str, Any]:
    return {"id": f"{agent_id}-{state}", "title": title, "agent_id": agent_id, "state": state}


# ---------------------------------------------------------------------------
# 1-4. Persisted stage record shapes
# ---------------------------------------------------------------------------


def test_width1_record_has_empty_workers_and_no_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=1, spawner=_Spawner())
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["team_width_requested"] == 1
    assert section["team_width_effective"] == outcome.plan.effective_workers  # type: ignore[union-attr]
    assert section["workers_attempted"] == 0
    assert section["workers_spawned"] == 0
    assert section["reason"] == "width_1"
    assert section["workers"] == []
    assert section["root"] is None
    assert section["aggregates"] is None
    assert state.run_record[TEAM_FANOUT_KEY] == section
    assert state.saves  # saved early


def test_all_success_record_counts_and_workers(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner())
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["team_width_requested"] == 3
    assert section["team_width_effective"] == 3
    assert section["workers_attempted"] == 3
    assert section["workers_spawned"] == 3
    assert section["workers_failed"] == 0
    assert section["reason"] == "spawned"
    assert len(section["workers"]) == 3
    assert all(entry["spawn_status"] == "spawned" for entry in section["workers"])
    assert all(entry["agent_id"] for entry in section["workers"])
    # file_count / shard metrics come from the plan, not the file list.
    for entry in section["workers"]:
        assert entry["file_count"] >= 1
        assert entry["shard_weight"] >= 1


def test_partial_failure_record_keeps_failed_entry(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner(fail_shard=1))
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["workers_attempted"] == 3
    assert section["workers_spawned"] == 2
    assert section["workers_failed"] == 1
    assert section["reason"] == "partial_spawn_failure"
    by_shard = {entry["shard_id"]: entry for entry in section["workers"]}
    assert by_shard[1]["spawn_status"] == "failed"
    assert by_shard[1]["agent_id"] is None
    assert by_shard[0]["spawn_status"] == "spawned"
    assert by_shard[0]["agent_id"] is not None
    assert by_shard[2]["spawn_status"] == "spawned"


@pytest.mark.parametrize("bad", ["", None, 123], ids=["empty", "none", "nonstring"])
def test_malformed_success_counts_as_failed_in_record(tmp_path: Path, bad: Any) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner(agent_id=bad))
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["workers_spawned"] == 0
    assert section["workers_failed"] == 3
    assert section["reason"] == "spawn_failed"
    assert all(entry["spawn_status"] == "failed" for entry in section["workers"])
    assert all(entry["agent_id"] is None for entry in section["workers"])


# ---------------------------------------------------------------------------
# 5. Deterministic mapping
# ---------------------------------------------------------------------------


def test_stage_record_deterministic_across_runs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    first = record_stage(_FakeReportState(), _stage_outcome(root, width=3, spawner=_Spawner()))
    second = record_stage(_FakeReportState(), _stage_outcome(root, width=3, spawner=_Spawner()))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert [entry["shard_id"] for entry in first["workers"]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 6-12. Final per-agent metrics keyed by agent id
# ---------------------------------------------------------------------------


def _early_success_section(root: Path) -> dict[str, Any]:
    outcome = _stage_outcome(root, width=3, spawner=_Spawner())
    return build_stage_section(outcome)


def test_final_metrics_read_from_each_agents_own_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    early = _early_success_section(root)
    agent_ids = [entry["agent_id"] for entry in early["workers"]]
    assert len(agent_ids) == 3
    coordinator = _FakeCoordinator(
        {
            agent_ids[0]: {
                "status": "completed",
                "model_requests": 7,
                "tool_calls": 12,
                "tool_groups": 4,
                "avg_tool_width": 3,
                "tools_serial_ms": 1000.0,
                "tools_wall_ms": 500.0,
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 50,
            },
            agent_ids[1]: {
                "status": "completed",
                "model_requests": 1,
                "tool_calls": 2,
                "tool_groups": 1,
                "avg_tool_width": 2,
                "tools_serial_ms": 100.0,
                "tools_wall_ms": 200.0,
                "input_tokens": 10,
                "cached_input_tokens": 5,
                "output_tokens": 6,
            },
            agent_ids[2]: {"status": "running"},
        }
    )
    usage = [
        {"agent_id": agent_ids[0], "model": "model-a", "cost": 0.02},
        {"agent_id": agent_ids[1], "model": "model-b", "cost": 0.01},
    ]
    coverage = [
        {"agent_id": agent_ids[0], "entry_id": "c1"},
        {"agent_id": agent_ids[0], "entry_id": "c2"},
        {"agent_id": agent_ids[1], "entry_id": "c3"},
    ]
    findings = [
        _finding(agent_ids[0], "verified"),
        _finding(agent_ids[0], "candidate"),
        _finding(agent_ids[1], "verified"),
        _finding(agent_ids[0], "retracted"),
        _finding(agent_ids[0], "rejected"),
        _finding(agent_ids[0], "proof_gap"),
    ]
    finalized = finalize_team_section(
        early,
        coordinator=coordinator,
        usage_agents=usage,
        coverage_entries=coverage,
        vulnerabilities=findings,
        worker_model="model-worker",
        root_agent_id=None,
    )
    by_agent = {entry["agent_id"]: entry for entry in finalized["workers"] if entry["agent_id"]}
    first = by_agent[agent_ids[0]]
    second = by_agent[agent_ids[1]]
    assert first["model"] == "model-a"
    assert first["model_requests"] == 7
    assert first["tool_calls"] == 12
    assert first["final_status"] == "completed"
    assert first["coverage_entries"] == 2
    assert second["coverage_entries"] == 1
    assert first["findings_by_state"] == {
        "candidate": 1,
        "verified": 1,
        "retracted": 1,
        "rejected": 1,
        "proof_gap": 1,
    }
    assert second["findings_by_state"] == {
        "candidate": 0,
        "verified": 1,
        "retracted": 0,
        "rejected": 0,
        "proof_gap": 0,
    }


def test_root_and_worker_metrics_not_swapped(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    early = _early_success_section(root)
    worker_agent = early["workers"][0]["agent_id"]
    coordinator = _FakeCoordinator(
        {
            "root-agent": {"status": "completed", "model_requests": 99, "tool_calls": 88},
            worker_agent: {"status": "completed", "model_requests": 3, "tool_calls": 4},
        }
    )
    finalized = finalize_team_section(
        early,
        coordinator=coordinator,
        root_agent_id="root-agent",
        root_model="root-model",
        worker_model="worker-model",
    )
    root_block = finalized["root"]
    assert root_block is not None
    assert root_block["model_requests"] == 99
    assert root_block["model"] == "root-model"
    worker_blocks = [e for e in finalized["workers"] if e["agent_id"] == worker_agent]
    assert worker_blocks[0]["model_requests"] == 3
    # Worker-only identity keys are not duplicated into root.
    for key in ("worker_id", "shard_id", "shard_weight", "shard_loc", "file_count", "spawn_status"):
        assert key not in root_block


def test_uncached_tokens_never_negative(tmp_path: Path) -> None:
    early = _section_with_snapshot(
        tmp_path,
        {"status": "completed", "input_tokens": 10, "cached_input_tokens": 40, "output_tokens": 1},
    )
    worker = early["workers"][0]
    assert worker["uncached_input_tokens"] == 0
    assert worker["input_tokens"] == 10


def _section_with_snapshot(tmp_path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    root = tmp_path / "repo"
    _repo(root)
    early = _early_success_section(root)
    agent_id = early["workers"][0]["agent_id"]
    coordinator = _FakeCoordinator({agent_id: snapshot})
    return finalize_team_section(early, coordinator=coordinator)


def test_effective_parallelism_derivation(tmp_path: Path) -> None:
    section = _section_with_snapshot(
        tmp_path,
        {"status": "completed", "tools_serial_ms": 1000.0, "tools_wall_ms": 500.0},
    )
    assert section["workers"][0]["effective_parallelism"] == 2.0


def test_zero_wall_time_parallelism_is_null(tmp_path: Path) -> None:
    section = _section_with_snapshot(
        tmp_path,
        {"status": "completed", "tools_serial_ms": 100.0, "tools_wall_ms": 0.0},
    )
    assert section["workers"][0]["effective_parallelism"] is None


def test_cache_ratio_avoid_division_by_zero(tmp_path: Path) -> None:
    section = _section_with_snapshot(
        tmp_path,
        {"status": "completed", "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
    )
    assert section["workers"][0]["cache_ratio"] == 0.0


# ---------------------------------------------------------------------------
# 13. Cost naming is load-bearing
# ---------------------------------------------------------------------------


def test_cost_field_is_allocated_cost_usd(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    early = _early_success_section(root)
    agent_id = early["workers"][0]["agent_id"]
    usage = [{"agent_id": agent_id, "model": "m", "cost": 0.25}]
    finalized = finalize_team_section(
        early, usage_agents=usage, root_agent_id="root", vulnerabilities=[]
    )
    assert finalized["workers"][0]["allocated_cost_usd"] == 0.25
    assert finalized["root"] is not None
    root_block = finalized["root"]
    assert "allocated_cost_usd" in root_block
    for entry in finalized["workers"]:
        assert "cost_usd" not in entry
        assert "exact_cost_usd" not in entry
    assert "team_allocated_cost_usd" in finalized["aggregates"]


# ---------------------------------------------------------------------------
# 14-15. Interrupted records & non-fatal collection
# ---------------------------------------------------------------------------


def test_interrupted_early_record_is_valid_json(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner(fail_shard=1))
    state = _FakeReportState()
    record_stage(state, outcome)
    raw = json.dumps(state.run_record)  # must not raise
    assert json.loads(raw)[TEAM_FANOUT_KEY]["reason"] == "partial_spawn_failure"


def test_finalize_and_persist_reads_report_state_authorities(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner())
    agent_ids = [w.agent_id for w in outcome.spawned]
    state = _FakeReportState(
        usage_agents=[
            {"agent_id": "root", "model": "root-model", "cost": 0.5},
            {"agent_id": agent_ids[0], "model": "worker-model", "cost": 0.25},
        ],
        vulnerabilities=[_finding("root", "verified"), _finding(agent_ids[0], "candidate")],
    )
    record_stage(state, outcome)
    coordinator = _FakeCoordinator(
        {
            "root": {"status": "completed", "model_requests": 5, "input_tokens": 100},
            agent_ids[0]: {"status": "completed", "model_requests": 2, "input_tokens": 10},
        }
    )
    result = finalize_and_persist(
        state,
        coordinator=coordinator,
        root_agent_id="root",
        root_model="fallback-root",
        worker_model="fallback-worker",
        coverage_entries=[{"agent_id": agent_ids[0]}, {"agent_id": "root"}],
    )
    # llm_usage / vulnerabilities came from the (fake) ReportState authorities.
    assert result["root"]["model"] == "root-model"
    assert result["root"]["allocated_cost_usd"] == 0.5
    assert result["root"]["findings_by_state"]["verified"] == 1
    assert result["workers"][0]["allocated_cost_usd"] == 0.25
    assert result["workers"][0]["findings_by_state"]["candidate"] == 1
    assert result["workers"][1]["model"] == "fallback-worker"  # no usage entry -> runner default
    # Aggregates are workers-only: root's 5 requests / $0.5 / 1 coverage are excluded.
    assert result["aggregates"]["team_model_requests"] == 2
    assert result["aggregates"]["team_allocated_cost_usd"] == 0.25
    assert result["aggregates"]["team_coverage_entries"] == 1
    assert result["aggregates"]["team_findings_by_state"]["verified"] == 0
    # Persisted through save_run_data (early stage save + final save).
    assert len(state.saves) == 2
    assert state.saves[-1][TEAM_FANOUT_KEY] == result
    assert state.saves[0][TEAM_FANOUT_KEY]["root"] is None


def test_settlement_failure_is_non_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner())
    state = _FakeReportState()
    record_stage(state, outcome)
    coordinator = _FakeCoordinator({}, fail=True)

    with caplog.at_level(logging.WARNING, logger="strix.team.telemetry"):
        result = finalize_and_persist(
            state, coordinator=coordinator, root_agent_id="root", root_model="m", worker_model="w"
        )
    # Per-agent coordinator failure degrades to empty counters + a warning;
    # the section is still persisted and remains valid JSON.
    assert any("non-fatal" in record.message for record in caplog.records)
    assert result["workers_spawned"] == 3
    assert all(entry["model_requests"] == 0 for entry in result["workers"])
    assert result["root"]["final_status"] == "unknown"
    json.dumps(state.run_record)
    assert state.run_record[TEAM_FANOUT_KEY]["reason"] == "spawned"

    # A failure of the persistence path itself is swallowed by the outer guard.
    def _boom() -> None:
        raise OSError("disk full")

    state.save_run_data = _boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="strix.team.telemetry"):
        assert finalize_and_persist(state, coordinator=None, root_agent_id="root") == {}
    assert any("finalize failed" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 16. No model-facing tool or policy change
# ---------------------------------------------------------------------------


def test_no_new_model_tool_or_policy_entry() -> None:
    names = {getattr(tool, "name", repr(tool)) for tool in registered_agent_tools()}
    assert not any(
        name in {"source_partition", "partition_source", "team_fanout"} or "telemetry" in name
        for name in names
    )


def test_tool_policy_unchanged_against_origin_main() -> None:
    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        pytest.skip("not a git checkout")
    proc = subprocess.run(
        ["git", "diff", "origin/main..HEAD", "--", "strix/core/tool_policy.py"],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"origin/main unavailable: {proc.stderr.strip()[:120]}")
    assert proc.stdout.strip() == ""


def test_worker_blocks_use_same_predicate_as_counter(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner())
    assert outcome.successfully_spawned == sum(
        1 for worker in outcome.spawned if _is_successful_spawn(worker)
    )
    section = build_stage_section(outcome)
    assert section["workers_spawned"] == sum(
        1 for entry in section["workers"] if entry["spawn_status"] == "spawned"
    )
    assert set(FINDING_STATE_KEYS) == {
        "candidate",
        "verified",
        "retracted",
        "rejected",
        "proof_gap",
    }


# ---------------------------------------------------------------------------
# Telemetry semantic correction: effective width must be 0 when no plan exists
# ---------------------------------------------------------------------------

#: Pinned no-plan effective width: zero workers could exist regardless of the
#: requested width, so effective width is 0 (not the requested value).
EXPECTED_NO_SOURCE_EFFECTIVE_WIDTH: int = 0


def _no_source_outcome(*, width: int) -> TeamStageOutcome:
    return asyncio.run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=[],
            spawn_worker=_Spawner().spawn,
            objective=OBJECTIVE,
            team_width=width,
        )
    )


def test_no_source_roots_records_effective_width_zero() -> None:
    outcome = _no_source_outcome(width=3)
    assert outcome.reason == "no_source_roots"
    assert outcome.plan is None
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["team_width_requested"] == 3
    assert section["team_width_effective"] == EXPECTED_NO_SOURCE_EFFECTIVE_WIDTH == 0
    assert section["workers_attempted"] == 0
    assert section["workers_spawned"] == 0
    assert section["workers_failed"] == 0
    assert section["reason"] == "no_source_roots"
    assert section["workers"] == []
    assert section["root"] is None
    assert section["aggregates"] is None


def test_zero_units_records_effective_width_zero(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    outcome = asyncio.run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=[{"source_path": str(empty)}],
            spawn_worker=_Spawner().spawn,
            objective=OBJECTIVE,
            team_width=3,
        )
    )
    assert outcome.reason == "zero_units"
    assert outcome.plan is not None
    assert outcome.plan.effective_workers == 0
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["team_width_requested"] == 3
    assert section["team_width_effective"] == EXPECTED_NO_SOURCE_EFFECTIVE_WIDTH == 0
    assert section["workers_attempted"] == 0
    assert section["workers_spawned"] == 0
    assert section["workers_failed"] == 0
    assert section["reason"] == "zero_units"
    assert section["workers"] == []


def test_spawner_unavailable_records_effective_width_zero() -> None:
    # Defensive path: width>1 reached the seam with no injected spawner. The
    # real runner always injects one, so construct the outcome directly.
    outcome = TeamStageOutcome(
        enabled=False,
        team_width=3,
        plan=None,
        spawned=(),
        reason="spawner_unavailable",
    )
    state = _FakeReportState()
    section = record_stage(state, outcome)
    assert section["team_width_requested"] == 3
    assert section["team_width_effective"] == EXPECTED_NO_SOURCE_EFFECTIVE_WIDTH == 0
    assert section["workers_attempted"] == 0
    assert section["workers_spawned"] == 0
    assert section["workers_failed"] == 0
    assert section["reason"] == "spawner_unavailable"
    assert section["workers"] == []


def test_plan_exists_effective_matches_plan(tmp_path: Path) -> None:
    # Regression: the `plan is not None` branch must override outcome.team_width
    # when the two disagree (e.g. a manifest with effective_workers < width).
    root = tmp_path / "repo"
    _repo(root)
    outcome = _stage_outcome(root, width=3, spawner=_Spawner())
    assert outcome.plan is not None
    plan_effective = outcome.plan.effective_workers
    assert plan_effective > 0
    section = build_stage_section(outcome)
    assert section["team_width_requested"] == outcome.team_width
    assert section["team_width_effective"] == plan_effective
    # Effective width comes from the plan, never from outcome.team_width.
    assert section["team_width_effective"] != 0 or plan_effective == 0
