"""CI quality gate: fail the build if the offline strategy regresses.

Floors are set comfortably below the observed offline numbers so that ordinary
noise does not flake CI, while a genuine regression (broken timing, dead signal,
leakage that inflates the null) trips the gate.
"""

from __future__ import annotations

import sys

from evals.harness import run_eval

# (key, comparison, threshold, human description)
FLOORS = [
    ("information_coefficient", "ge", 0.05, "rank IC of signal vs next-day return"),
    ("sharpe", "ge", 0.80, "annualised in-sample Sharpe"),
    ("walk_forward_oos_sharpe", "ge", 0.40, "purged walk-forward OOS Sharpe"),
    ("psr", "ge", 0.90, "probabilistic Sharpe"),
    ("deflated_sharpe", "ge", 0.60, "deflated Sharpe (search-adjusted)"),
    ("sentiment_recovery_corr", "ge", 0.40, "lexical sentiment recovery corr"),
    ("event_t_stat", "ge", 2.0, "event-study long-short t-stat"),
    # The null must stay quiet -- this is the anti-overfitting check.
    ("null_deflated_sharpe", "le", 0.50, "deflated Sharpe on scrambled signal"),
    ("null_sharpe", "le", 1.00, "Sharpe on scrambled signal"),
]


def main() -> int:
    res = run_eval()
    failures: list[str] = []
    print("FinSight eval gate")
    for key, cmp, thr, desc in FLOORS:
        val = res[key]
        ok = val >= thr if cmp == "ge" else val <= thr
        sign = ">=" if cmp == "ge" else "<="
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: {val} {sign} {thr}")
        if not ok:
            failures.append(f"{desc}: {val} not {sign} {thr}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
