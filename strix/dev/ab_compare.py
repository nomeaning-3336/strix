"""A/B comparison for the WideTurn acceptance gate.

Run the same assessment objective twice — once on the pre-WideTurn harness
(``STRIX_PARALLEL_TOOLS=0``) and once with wide turns enabled — then compare the
two run directories:

    python -m strix.dev.ab_compare --baseline <run_dir> --wide <run_dir>

The gate (thresholds configurable):

  - >= 20% fewer LLM requests
  - >= 20% lower wall-clock time
  - coverage not reduced: every baseline coverage *surface* still reviewed
    (real coverage breadth from coverage.json entries — not findings_filed)
  - no previously verified finding lost (active finding fingerprints)
  - prompt-cache ratio does not materially regress (>= -5 points)

New findings and new CWE classes are REPORTED for review, not treated as
automatic false positives, and retracted/rejected/proof_gap deltas are
reported alongside — that is what the finding-lifecycle machinery is for.

Compares artifacts, not in-memory state: both runs must have completed
normally (run.json status ``completed``).
"""

from __future__ import annotations

# ruff: noqa: T201 - CLI module; prints the comparison summary to stdout.
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from strix.report.dedupe import finding_fingerprint
from strix.report.finding_state import INACTIVE_STATES, state_of


DEFAULT_REQUEST_IMPROVEMENT = 0.20
DEFAULT_WALL_IMPROVEMENT = 0.20
DEFAULT_CACHE_RATIO_REGRESSION = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not an object")
    return data


def _list_json(path: Path) -> list[Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} unreadable: {exc}") from exc
    if not isinstance(data, list):
        raise TypeError(f"{path} is not a list")
    return data


def _wall_seconds(run: dict[str, Any]) -> float:
    start = run.get("start_time")
    end = run.get("end_time")
    if not isinstance(start, str) or not isinstance(end, str):
        raise TypeError("run.json must carry start_time and end_time as strings")

    def _parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return max(0.0, (_parse(end) - _parse(start)).total_seconds())


def _coverage_breadth(run_dir: Path) -> dict[str, Any]:
    """Real coverage breadth: the surfaces actually reviewed, plus outcomes.

    Deliberately NOT ``summary.findings_filed`` — that is a finding count, not
    coverage. A run that inspected half as much code but filed the same number
    of findings must not pass the coverage gate.
    """
    coverage_path = run_dir / "coverage.json"
    if not coverage_path.exists():
        return {"surfaces": set(), "entries": 0, "completed": 0, "gaps": 0}
    coverage = _read_json(coverage_path)
    entries = coverage.get("entries") or []
    surfaces: set[str] = set()
    completed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        surface = entry.get("surface")
        if isinstance(surface, str) and surface.strip():
            surfaces.add(surface.strip())
        if str(entry.get("outcome", "")) == "completed":
            completed += 1
    gaps = len(coverage.get("gaps") or [])
    if not surfaces:
        summary = coverage.get("summary") or {}
        surfaces_count = int(summary.get("surfaces_reviewed") or 0)
        if surfaces_count:
            surfaces = {f"surface-{index}" for index in range(surfaces_count)}
    return {
        "surfaces": surfaces,
        "entries": len(entries),
        "completed": completed,
        "gaps": gaps,
    }


def _finding_keys_and_states(reports: list[Any]) -> tuple[set[str], dict[str, int]]:
    """Active finding identity keys + per-state counts.

    Key = deterministic finding fingerprint when the record carries identity
    fields, else its CWE token — enough to compare which findings a run kept.
    """
    keys: set[str] = set()
    states: dict[str, int] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        state = state_of(report)
        states[state] = states.get(state, 0) + 1
        if state in INACTIVE_STATES:
            continue
        fingerprint = finding_fingerprint(report)
        key = fingerprint or f"cwe:{str(report.get('cwe') or 'untyped').strip().upper()}"
        keys.add(key)
    return keys, states


def _finding_records(run_dir: Path) -> list[Any]:
    path = run_dir / "vulnerabilities.json"
    if not path.exists():
        return []
    return _list_json(path)


def compare_runs(
    baseline_dir: Path,
    wide_dir: Path,
    *,
    request_improvement: float = DEFAULT_REQUEST_IMPROVEMENT,
    wall_improvement: float = DEFAULT_WALL_IMPROVEMENT,
    cache_ratio_regression: float = DEFAULT_CACHE_RATIO_REGRESSION,
) -> dict[str, Any]:
    """Compare two completed runs and evaluate the WideTurn acceptance gate."""
    baseline = _read_json(baseline_dir / "run.json")
    wide = _read_json(wide_dir / "run.json")
    for name, run in (("baseline", baseline), ("wide", wide)):
        if run.get("status") != "completed":
            raise ValueError(f"{name} run has status {run.get('status')!r}, not completed")

    def _usage(run: dict[str, Any]) -> dict[str, Any]:
        usage = run.get("llm_usage") or {}
        return usage if isinstance(usage, dict) else {}

    base_usage, wide_usage = _usage(baseline), _usage(wide)
    base_requests = max(0, int(base_usage.get("requests") or 0))
    wide_requests = max(0, int(wide_usage.get("requests") or 0))
    base_wall = _wall_seconds(baseline)
    wide_wall = _wall_seconds(wide)

    base_coverage = _coverage_breadth(baseline_dir)
    wide_coverage = _coverage_breadth(wide_dir)
    dropped_surfaces = sorted(base_coverage["surfaces"] - wide_coverage["surfaces"])
    added_surfaces = sorted(wide_coverage["surfaces"] - base_coverage["surfaces"])

    base_keys, base_states = _finding_keys_and_states(_finding_records(baseline_dir))
    wide_keys, wide_states = _finding_keys_and_states(_finding_records(wide_dir))
    lost_verified = sorted(base_keys - wide_keys)
    new_verified = sorted(wide_keys - base_keys)

    def _pct(new: float, old: float) -> float | None:
        if old <= 0:
            return None
        return (new - old) / old

    request_delta = _pct(wide_requests, base_requests)
    wall_delta = _pct(wide_wall, base_wall)
    base_ratio = float(base_usage.get("cache_ratio") or 0.0)
    wide_ratio = float(wide_usage.get("cache_ratio") or 0.0)
    cache_delta = wide_ratio - base_ratio

    checks = {
        "requests": bool(request_delta is not None and request_delta <= -request_improvement),
        "wall_clock": bool(wall_delta is not None and wall_delta <= -wall_improvement),
        "coverage_not_dropped": not dropped_surfaces,
        "verified_not_lost": not lost_verified,
        "cache_ratio": cache_delta >= -cache_ratio_regression,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": {
            "baseline_requests": base_requests,
            "wide_requests": wide_requests,
            "request_delta": request_delta,
            "baseline_wall_seconds": round(base_wall, 1),
            "wide_wall_seconds": round(wide_wall, 1),
            "wall_delta": wall_delta,
            "baseline_cache_ratio": base_ratio,
            "wide_cache_ratio": wide_ratio,
            "cache_delta": cache_delta,
            "baseline_cost": float(base_usage.get("cost") or 0.0),
            "wide_cost": float(wide_usage.get("cost") or 0.0),
            "baseline_surfaces": len(base_coverage["surfaces"]),
            "wide_surfaces": len(wide_coverage["surfaces"]),
            "baseline_coverage_entries": base_coverage["entries"],
            "wide_coverage_entries": wide_coverage["entries"],
            "baseline_findings": len(_finding_records(baseline_dir)),
            "wide_findings": len(_finding_records(wide_dir)),
        },
        "coverage": {
            "dropped_surfaces": dropped_surfaces,
            "added_surfaces": added_surfaces,
            "baseline_completed": base_coverage["completed"],
            "wide_completed": wide_coverage["completed"],
            "baseline_gaps": base_coverage["gaps"],
            "wide_gaps": wide_coverage["gaps"],
        },
        "findings": {
            "lost_verified": lost_verified,
            "new_verified": new_verified,
            "baseline_states": base_states,
            "wide_states": wide_states,
        },
    }


def _fmt(delta: float | None, *, inverse: bool = False) -> str:
    if delta is None:
        return "n/a"
    value = -delta if inverse else delta
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Pre-WideTurn run directory")
    parser.add_argument("--wide", required=True, help="WideTurn run directory")
    parser.add_argument("--request-improvement", type=float, default=DEFAULT_REQUEST_IMPROVEMENT)
    parser.add_argument("--wall-improvement", type=float, default=DEFAULT_WALL_IMPROVEMENT)
    parser.add_argument(
        "--cache-ratio-regression",
        type=float,
        default=DEFAULT_CACHE_RATIO_REGRESSION,
    )
    args = parser.parse_args(argv)

    result = compare_runs(
        Path(args.baseline),
        Path(args.wide),
        request_improvement=args.request_improvement,
        wall_improvement=args.wall_improvement,
        cache_ratio_regression=args.cache_ratio_regression,
    )
    m = result["metrics"]
    print("WideTurn A/B comparison")
    print(f"  LLM requests:      {m['baseline_requests']} -> {m['wide_requests']} "
          f"({_fmt(m['request_delta'], inverse=True)} requests)")
    print(f"  Wall clock:        {m['baseline_wall_seconds']}s -> {m['wide_wall_seconds']}s "
          f"({_fmt(m['wall_delta'], inverse=True)} time)")
    print(f"  Cost:              ${m['baseline_cost']:.2f} -> ${m['wide_cost']:.2f}")
    print(f"  Cache ratio:       {m['baseline_cache_ratio'] * 100:.0f}% -> "
          f"{m['wide_cache_ratio'] * 100:.0f}%")
    print(f"  Coverage surfaces: {m['baseline_surfaces']} -> {m['wide_surfaces']} "
          f"(entries {m['baseline_coverage_entries']} -> {m['wide_coverage_entries']})")

    cov = result["coverage"]
    if cov["dropped_surfaces"]:
        print(f"  DROPPED surfaces (coverage loss): {', '.join(cov['dropped_surfaces'])}")
    if cov["added_surfaces"]:
        print(f"  added surfaces: {', '.join(cov['added_surfaces'])}")
    if cov["baseline_completed"] or cov["wide_completed"]:
        print(f"  completed outcomes: {cov['baseline_completed']} -> {cov['wide_completed']}")
    if cov["baseline_gaps"] or cov["wide_gaps"]:
        print(f"  coverage gaps: {cov['baseline_gaps']} -> {cov['wide_gaps']}")

    findings = result["findings"]
    if findings["lost_verified"]:
        print(f"  LOST verified findings (regression): {', '.join(findings['lost_verified'])}")
    if findings["new_verified"]:
        print(
            "  NEW verified findings (review, not a failure): "
            f"{', '.join(findings['new_verified'][:10])}"
        )
    for label in ("retracted", "rejected", "proof_gap"):
        before = findings["baseline_states"].get(label, 0)
        after = findings["wide_states"].get(label, 0)
        if before or after:
            print(f"  finding state {label}: {before} -> {after}")

    for check, passed in result["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
    print("ACCEPTANCE GATE:", "PASSED" if result["pass"] else "NOT MET")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
