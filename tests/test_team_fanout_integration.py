"""Team fan-out wired into source scans - deterministic integration tests.

These tests exercise the orchestration seam (``strix.team.integration``) with
fake spawn workers and tiny ``tmp_path`` trees only: no model calls, no
network, no paid scans, no live runner.  ``STRIX_TEAM_WIDTH`` gating is tested
through the settings loader exactly like other env knobs.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from strix.config import loader
from strix.config.loader import load_settings
from strix.team.fanout import WORKER_SCOPE_DIRECTIVE
from strix.team.integration import TeamStageOutcome, stage_source_team_fanout
from strix.tools.source_partition import PartitionConfig, partition_source
from strix.tools.source_partition.models import PartitionManifest, PartitionShard


if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

OBJECTIVE = "Investigate the assigned shard for security weaknesses."


def _code(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v{i} = {i}\n" for i in range(lines)), encoding="utf-8")


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


class _Recorder:
    """Sentinel fake spawner: records every call; optional failure shard."""

    def __init__(self, fail_shard: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_shard = fail_shard
        self.count = 0

    async def spawn(
        self, *, name: str, task: str, skills: list[str], parent_history: list[Any]
    ) -> dict[str, Any]:
        shard = int(name.rsplit("-", 1)[-1])
        if self.fail_shard == shard:
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
) -> TeamStageOutcome:
    local_sources = [{"source_path": str(root)} for root in roots]
    return _run(
        stage_source_team_fanout(
            report_state=None,
            local_sources=local_sources,
            spawn_worker=recorder.spawn,
            objective=OBJECTIVE,
            team_width=width,
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
# 1. STRIX_TEAM_WIDTH default (1) preserves the legacy single-agent flow
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
    assert len(recorder.calls) == outcome.plan.effective_workers
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
# 9. Legacy spawner call shape unchanged at width>1 (same four kwargs)
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
