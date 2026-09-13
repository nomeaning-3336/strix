"""Cost display: a non-zero spend must never render as "$0.00".

`format_cost_usd` / `format_spend_percent` are shared by the CLI stats, the
agent-facing budget notices and the web viewer, which keeps the same thresholds
in frontend/src/lib/display-number.ts. These tests pin the table itself, since
the whole point is that the *rendered string* is honest about micro-spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strix.report.usage import format_cost_usd, format_spend_percent


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        # Nothing spent is the only case that may read as zero.
        (0.0, "$0.00"),
        (-1.0, "$0.00"),
        # The regression this helper exists for: a cheap, mostly-cached run.
        (0.0014, "$0.0014"),
        # Just under a cent still uses the sub-cent tier, so it rounds up to 4dp.
        (0.00999, "$0.0100"),
        (0.0001, "$0.0001"),
        (0.000012, "$0.000012"),
        # Normal magnitudes keep two decimals.
        (0.01, "$0.01"),
        (1.234, "$1.23"),
        (5.87559595, "$5.88"),
        (1234.5, "$1234.50"),
        # Positive but below the smallest printed unit: still not "$0.00".
        (0.0000001, "<$0.000001"),
        (0.0000009, "<$0.000001"),
    ],
)
def test_cost_never_renders_nonzero_spend_as_zero(cost: float, expected: str) -> None:
    rendered = format_cost_usd(cost)
    assert rendered == expected
    if cost > 0:
        assert rendered != "$0.00", "a non-zero spend must not render as $0.00"


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        (0.0099, "$0.0099"),
        (0.01, "$0.01"),
        (0.000012, "$0.000012"),
        (0.0000004, "<$0.000001"),
    ],
)
def test_cost_boundary_values(cost: float, expected: str) -> None:
    assert format_cost_usd(cost) == expected


def test_cost_handles_non_finite_values() -> None:
    assert format_cost_usd(float("nan")) == "$0.00"
    assert format_cost_usd(float("inf")) == "$0.00"


@pytest.mark.parametrize(
    ("spent", "ceiling", "expected"),
    [
        (0.0, 20.0, "0%"),
        # A real spend against a large ceiling is "<0.1%", never a bare "0%".
        (0.0014, 20.0, "<0.1%"),
        (0.019, 20.0, "<0.1%"),
        # Small but visible percentages keep one decimal (half-up, so the
        # viewer's Math.round and this agree: 0.25% -> "0.3%").
        (0.04, 20.0, "0.2%"),
        (0.05, 20.0, "0.3%"),
        (0.5, 20.0, "3%"),
        (5.87559595, 20.0, "29%"),
        (20.0, 20.0, "100%"),
        (30.0, 20.0, "150%"),
    ],
)
def test_spend_percent_never_collapses_real_spend(
    spent: float, ceiling: float, expected: str
) -> None:
    assert format_spend_percent(spent, ceiling) == expected


def test_spend_percent_without_a_usable_ceiling() -> None:
    assert format_spend_percent(1.0, 0.0) == "0%"
    assert format_spend_percent(1.0, -5.0) == "0%"
    assert format_spend_percent(float("nan"), 20.0) == "0%"


def test_viewer_bundle_uses_the_same_thresholds() -> None:
    """The TS twin must keep the same tiers, or the CLI and viewer disagree.

    A behavioural cross-language test is not practical here, so this pins the
    thresholds that the two implementations document in common: sub-cent keeps
    four decimals, one cent and above keeps two.
    """
    ts_path = (
        Path(__file__).resolve().parents[1]
        / "strix/interface/viewer/frontend/src/lib/display-number.ts"
    )
    source = ts_path.read_text(encoding="utf-8")
    assert "cost.toFixed(2)" in source
    assert "cost.toFixed(4)" in source
    assert "cost.toFixed(6)" in source
    assert '"<$0.000001"' in source
    assert '"<0.1%"' in source
