"""Minimal end-to-end example: run the offline strategy and deflate it honestly.

    python examples/run_strategy.py

Mirrors what ``evals.harness`` does, in miniature: search a few smoothing
windows, then report the winner's Deflated Sharpe using the *actual* variance of
Sharpes across the configurations searched (not an arbitrary number).
"""

from __future__ import annotations

import numpy as np

from finsight.backtest.engine import run_backtest
from finsight.config import Settings
from finsight.data.factory import load_panel
from finsight.sentiment.factory import get_backend
from finsight.signals.factor import build_signal, score_news
from finsight.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from finsight.stats.metrics import sharpe_ratio

GRID = [1, 2, 3, 5, 8, 13, 21]

if __name__ == "__main__":
    cfg = Settings.from_env()
    panel = load_panel(cfg)
    sentiment = score_news(panel, get_backend(cfg))

    trials = []
    for w in GRID:
        smoothed = sentiment.rolling(w, min_periods=1).mean()
        signal = build_signal(smoothed, winsorize=cfg.winsorize)
        rets = run_backtest(signal, panel.returns, cfg.cost_per_turn).returns
        per_period = rets.mean() / rets.std(ddof=1)
        trials.append((w, rets, float(per_period)))

    best_w, best_rets, _ = max(trials, key=lambda t: t[2])
    sr_variance = float(np.var([t[2] for t in trials], ddof=1))

    print(f"Searched smoothing windows {GRID} -> selected window {best_w}")
    print(f"  Sharpe (annualised) : {sharpe_ratio(best_rets):.2f}")
    print(f"  PSR (vs 0)          : {probabilistic_sharpe_ratio(best_rets):.3f}")
    print(
        f"  Deflated Sharpe     : "
        f"{deflated_sharpe_ratio(best_rets, sr_variance, len(GRID)):.3f}"
        f"  (adjusted for {len(GRID)} trials)"
    )
