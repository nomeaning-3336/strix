// Slim, dependency-free extract of strix-app's display-number helper. The full
// version queries Supabase to compute org-wide finding numbers; the local viewer
// only ever needs the pure formatter, so the supabase-backed functions are
// intentionally omitted (a local run has no org context).
export function formatStrixId(num: number): string {
  return `STRIX-${num}`;
}

/** Format an integer with locale thousands separators (e.g. 68339486 -> "68,339,486"). */
export function formatNumber(num: number): string {
  return new Intl.NumberFormat("en-US").format(num);
}

/**
 * Format a USD amount with adaptive precision so a cheap run never *looks* free.
 *
 * A fixed 2-decimal format is wrong for micro-spend: a run that really cost
 * $0.0014 (a few thousand mostly-cached tokens on an inexpensive model) renders
 * as "$0.00", which reads as "this run was free" or "the cost tracker is
 * broken". Precision therefore scales with magnitude:
 *
 *   0                 -> "$0.00"        (genuinely nothing spent)
 *   0 < cost < 0.01   -> "$0.0014"      (sub-cent: 4 decimals)
 *   cost < 0.0001     -> "$0.000012"    (6 decimals)
 *   cost >= 0.01      -> "$1.23"        (2 decimals)
 *   cost > 0 but rounds away even at 6 decimals -> "<$0.000001"
 */
export function formatCostUsd(cost: number): string {
  if (!Number.isFinite(cost) || cost <= 0) return "$0.00";
  if (cost >= 0.01) return `$${cost.toFixed(2)}`;
  if (cost >= 0.0001) return `$${cost.toFixed(4)}`;
  if (cost >= 0.000001) return `$${cost.toFixed(6)}`;
  return "<$0.000001";
}

/**
 * Percentage of the scan budget spent, without collapsing small spend to "0%".
 *
 * "0.06%" is not information anyone needs, but "0%" for a run that has spent
 * money is actively misleading, so anything under a tenth of a percent is
 * reported as "<0.1%" rather than rounded down to zero. Rounding is half-up
 * (``Math.round``), matching the Python twin in strix/report/usage.py, which
 * implements half-up explicitly instead of Python's default half-to-even.
 */
export function formatSpendPercent(spent: number, budget: number): string {
  if (!Number.isFinite(spent) || !Number.isFinite(budget) || budget <= 0) return "0%";
  if (spent <= 0) return "0%";
  const pct = (spent / budget) * 100;
  if (pct < 0.1) return "<0.1%";
  if (pct < 1) return `${(Math.round(pct * 10) / 10).toFixed(1)}%`;
  return `${Math.round(pct)}%`;
}
