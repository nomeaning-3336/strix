"""A/B comparison harness tests (WideTurn acceptance gate)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from strix.dev.ab_compare import compare_runs


if TYPE_CHECKING:
    from pathlib import Path


def _write_run(
    run_dir: Path,
    *,
    requests: int,
    seconds: float,
    cost: float,
    cache_ratio: float,
    start: str = "2026-09-02 10:00:00Z",
    findings: list[dict[str, str]] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "start_time": start,
                "end_time": "2026-09-02 10:00:00Z" if False else _after(start, seconds),
                "llm_usage": {
                    "requests": requests,
                    "input_tokens": 10_000,
                    "output_tokens": 1_000,
                    "total_tokens": 11_000,
                    "cost": cost,
                    "cache_ratio": cache_ratio,
                },
            }
        ),
        encoding="utf-8",
    )
    run_dir.joinpath("vulnerabilities.json").write_text(
        json.dumps(findings or []), encoding="utf-8"
    )
    run_dir.joinpath("coverage.json").write_text(
        json.dumps({"findings_filed": len(findings or [])}), encoding="utf-8"
    )
    return run_dir


def _after(start: str, seconds: float) -> str:
    parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def test_acceptance_gate_passes_on_wide_improvements(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path / "base",
        requests=500,
        seconds=3000.0,
        cost=5.0,
        cache_ratio=0.60,
        findings=[{"id": "a", "cwe": "CWE-79"}, {"id": "b", "cwe": "CWE-89"}],
    )
    wide = _write_run(
        tmp_path / "wide",
        requests=350,  # -30% requests
        seconds=2100.0,  # -30% wall
        cost=4.0,
        cache_ratio=0.58,  # -2 points, within tolerance
        findings=[{"id": "a", "cwe": "CWE-79"}, {"id": "b", "cwe": "CWE-89"}],
    )
    result = compare_runs(base, wide)
    assert result["pass"] is True
    assert result["checks"] == {
        "requests": True,
        "wall_clock": True,
        "coverage_not_dropped": True,
        "no_new_fp_class": True,
        "cache_ratio": True,
    }
    assert result["metrics"]["wide_requests"] == 350


def test_acceptance_gate_fails_when_requests_do_not_improve(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", requests=500, seconds=3000.0, cost=5.0, cache_ratio=0.6)
    wide = _write_run(tmp_path / "wide", requests=480, seconds=2000.0, cost=4.0, cache_ratio=0.6)
    result = compare_runs(base, wide)
    assert result["pass"] is False
    assert result["checks"]["requests"] is False
    assert result["checks"]["wall_clock"] is True


def test_new_finding_class_flags_for_review(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path / "base",
        requests=100,
        seconds=1000.0,
        cost=1.0,
        cache_ratio=0.5,
        findings=[{"id": "a", "cwe": "CWE-79"}],
    )
    wide = _write_run(
        tmp_path / "wide",
        requests=60,
        seconds=600.0,
        cost=1.0,
        cache_ratio=0.5,
        findings=[{"id": "a", "cwe": "CWE-79"}, {"id": "c", "cwe": "CWE-918"}],
    )
    result = compare_runs(base, wide)
    assert result["pass"] is False
    assert result["classes"]["wide_only"] == ["CWE-918"]


def test_cache_ratio_regression_fails_gate(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", requests=100, seconds=1000.0, cost=1.0, cache_ratio=0.60)
    wide = _write_run(tmp_path / "wide", requests=60, seconds=600.0, cost=1.0, cache_ratio=0.30)
    result = compare_runs(base, wide)
    assert result["pass"] is False
    assert result["checks"]["cache_ratio"] is False


def test_unfinished_run_is_rejected(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", requests=100, seconds=100.0, cost=1.0, cache_ratio=0.5)
    base.joinpath("run.json").write_text(
        base.joinpath("run.json").read_text(encoding="utf-8").replace('"completed"', '"running"'),
        encoding="utf-8",
    )
    wide = _write_run(tmp_path / "wide", requests=80, seconds=80.0, cost=1.0, cache_ratio=0.5)
    with pytest.raises(ValueError, match="not completed"):
        compare_runs(base, wide)
