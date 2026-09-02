"""A/B comparison harness tests (WideTurn acceptance gate)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
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
    surfaces: list[str] | None = None,
    completed: int = 0,
    gaps: int = 0,
    findings: list[dict[str, object]] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "start_time": "2026-09-02 10:00:00Z",
                "end_time": _after("2026-09-02 10:00:00Z", seconds),
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
    surfaces = surfaces or [f"surface-{i}" for i in range(3)]
    run_dir.joinpath("coverage.json").write_text(
        json.dumps(
            {
                "summary": {"surfaces_reviewed": len(surfaces), "findings_filed": 2},
                "entries": [
                    {"surface": s, "outcome": "completed" if i < completed else "partial"}
                    for i, s in enumerate(surfaces)
                ],
                "gaps": [{"surface": s} for s in surfaces[gaps:]] if gaps else [],
            }
        ),
        encoding="utf-8",
    )
    run_dir.joinpath("vulnerabilities.json").write_text(
        json.dumps(findings or []), encoding="utf-8"
    )
    return run_dir


def _after(start: str, seconds: float) -> str:
    parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return (parsed + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _finding(cwe: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": f"vuln-{abs(hash(cwe)) % 10000:04d}",
        "title": f"{cwe} in search",
        "severity": "high",
        "timestamp": "2026-09-02 10:00:00 UTC",
        "cwe": cwe,
        "endpoint": "/search",
        "method": "GET",
        "finding_class": "dynamic",
    }
    record.update(overrides)
    return record


def test_acceptance_gate_passes_on_wide_improvements(tmp_path: Path) -> None:
    findings = [_finding("CWE-79"), _finding("CWE-89")]
    base = _write_run(
        tmp_path / "base",
        requests=500,
        seconds=3000.0,
        cost=5.0,
        cache_ratio=0.60,
        findings=findings,
    )
    wide = _write_run(
        tmp_path / "wide",
        requests=350,  # -30% requests
        seconds=2100.0,  # -30% wall
        cost=4.0,
        cache_ratio=0.58,  # -2 points, within tolerance
        findings=findings,
    )
    result = compare_runs(base, wide)
    assert result["pass"] is True
    assert result["checks"] == {
        "requests": True,
        "wall_clock": True,
        "coverage_not_dropped": True,
        "verified_not_lost": True,
        "cache_ratio": True,
    }
    assert result["findings"]["new_verified"] == []
    assert result["findings"]["lost_verified"] == []


def test_acceptance_gate_fails_when_requests_do_not_improve(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", requests=500, seconds=3000.0, cost=5.0, cache_ratio=0.6)
    wide = _write_run(tmp_path / "wide", requests=480, seconds=2000.0, cost=4.0, cache_ratio=0.6)
    result = compare_runs(base, wide)
    assert result["pass"] is False
    assert result["checks"]["requests"] is False
    assert result["checks"]["wall_clock"] is True


def test_dropped_coverage_surface_fails_the_gate(tmp_path: Path) -> None:
    base = _write_run(tmp_path / "base", requests=500, seconds=3000.0, cost=5.0, cache_ratio=0.6)
    # Wide inspected one fewer surface but "filed" the same number of findings
    # (summary.findings_filed stays 2) — the surface gate must still fail.
    wide = _write_run(
        tmp_path / "wide",
        requests=350,
        seconds=2100.0,
        cost=4.0,
        cache_ratio=0.6,
        surfaces=["surface-0", "surface-1"],
    )
    result = compare_runs(base, wide)
    assert result["pass"] is False
    assert result["checks"]["coverage_not_dropped"] is False
    assert result["coverage"]["dropped_surfaces"] == ["surface-2"]


def test_lost_verified_finding_fails_gate_but_new_findings_do_not(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path / "base",
        requests=500,
        seconds=3000.0,
        cost=5.0,
        cache_ratio=0.6,
        findings=[_finding("CWE-79"), _finding("CWE-89")],
    )
    # Wide lost the CWE-89 finding but legitimately found a new CWE-918 one.
    wide = _write_run(
        tmp_path / "wide",
        requests=350,
        seconds=2100.0,
        cost=4.0,
        cache_ratio=0.6,
        findings=[_finding("CWE-79"), _finding("CWE-918")],
    )
    result = compare_runs(base, wide)
    assert result["pass"] is False
    assert result["checks"]["verified_not_lost"] is False
    assert any("CWE-89" in key for key in result["findings"]["lost_verified"])
    # A new class is review material, not an automatic failure.
    assert any("CWE-918" in key for key in result["findings"]["new_verified"])


def test_new_verified_findings_alone_do_not_fail_the_gate(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path / "base",
        requests=500,
        seconds=3000.0,
        cost=5.0,
        cache_ratio=0.6,
        findings=[_finding("CWE-79")],
    )
    wide = _write_run(
        tmp_path / "wide",
        requests=350,
        seconds=2100.0,
        cost=4.0,
        cache_ratio=0.6,
        findings=[_finding("CWE-79"), _finding("CWE-918")],
    )
    result = compare_runs(base, wide)
    assert result["pass"] is True
    assert any("CWE-918" in key for key in result["findings"]["new_verified"])


def test_state_deltas_are_reported(tmp_path: Path) -> None:
    base = _write_run(
        tmp_path / "base",
        requests=500,
        seconds=3000.0,
        cost=5.0,
        cache_ratio=0.6,
        findings=[_finding("CWE-79")],
    )
    wide = _write_run(
        tmp_path / "wide",
        requests=350,
        seconds=2100.0,
        cost=4.0,
        cache_ratio=0.6,
        findings=[
            _finding("CWE-79"),
            _finding("CWE-89", state="retracted", state_reason="no invariant"),
        ],
    )
    result = compare_runs(base, wide)
    assert result["findings"]["wide_states"]["retracted"] == 1
    assert result["pass"] is True  # a retraction is recorded, not swept under the rug


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
