"""A/B comparison for the WideTurn acceptance gate.

Run the same assessment objective twice — once on the pre-WideTurn harness
(``STRIX_PARALLEL_TOOLS=0``) and once with wide turns enabled — then compare the
two run directories:

    python -m strix.dev.ab_compare --baseline <run_dir> --wide <run_dir>

The gate (all must hold, thresholds configurable):

  - >= 20% fewer LLM requests
  - >= 20% lower wall-clock time
  - no coverage drop
  - no new false-positive *class* (compare CWE/CVE classes of filed findings)
  - prompt-cache ratio does not materially regress (>= -5 points)

This compares artifacts, not in-memory state, so both runs must have completed
normally (run.json present with status ``completed``).
"""

from __future__ import annotations

# ruff: noqa: T201 - CLI module; prints the comparison summary to stdout.
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


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


def _finding_classes(reports: list[Any]) -> set[str]:
    classes: set[str] = set()
    for report in reports:
        if not isinstance(report, dict):
            continue
        cwe = str(report.get("cwe") or "").strip()
        cve = str(report.get("cve") or "").strip()
        classes.add(cwe or cve or "unclassified")
    return classes


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

    base_reports = _list_json(baseline_dir / "vulnerabilities.json") if (
        baseline_dir / "vulnerabilities.json"
    ).exists() else []
    wide_reports = _list_json(wide_dir / "vulnerabilities.json") if (
        wide_dir / "vulnerabilities.json"
    ).exists() else []
    base_classes = _finding_classes(base_reports)
    wide_classes = _finding_classes(wide_reports)
    new_classes = sorted(wide_classes - base_classes)
    missing_classes = sorted(base_classes - wide_classes)

    def _coverage_count(run_dir: Path) -> int:
        coverage_path = run_dir / "coverage.json"
        if not coverage_path.exists():
            return 0
        coverage = _read_json(coverage_path)
        return int(coverage.get("findings_filed") or 0)

    base_coverage = _coverage_count(baseline_dir)
    wide_coverage = _coverage_count(wide_dir)

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
        "coverage_not_dropped": wide_coverage >= base_coverage,
        "no_new_fp_class": not new_classes,
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
            "baseline_coverage_findings": base_coverage,
            "wide_coverage_findings": wide_coverage,
            "baseline_findings": len(base_reports),
            "wide_findings": len(wide_reports),
        },
        "classes": {
            "baseline_only": missing_classes,
            "wide_only": new_classes,
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
    print(f"  Findings/coverage: {m['baseline_findings']}/{m['baseline_coverage_findings']} -> "
          f"{m['wide_findings']}/{m['wide_coverage_findings']}")
    classes = result["classes"]
    if classes["wide_only"]:
        print(f"  NEW finding classes (investigate): {', '.join(classes['wide_only'])}")
    if classes["baseline_only"]:
        print(f"  classes lost vs baseline: {', '.join(classes['baseline_only'])}")

    for check, passed in result["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
    print("ACCEPTANCE GATE:", "PASSED" if result["pass"] else "NOT MET")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
