"""Vectorised long/short backtest with explicit, leakage-free timing.

Timing contract (the bit that kills most naive backtests):
    weights computed from news on day t are *held* on day t+1 and earn the
    day-t+1 return. We implement that by shifting weights forward one day, so a
    position can never see the return that justifies it.
"""

from __future__ import annotations

import pandas as pd

from finsight.types import Backtest, Signal


def run_backtest(signal: Signal, returns: pd.DataFrame, cost_per_turn: float = 0.0010) -> Backtest:
    """Run the backtest and return gross/net returns, turnover and equity curve."""
    weights = signal.weights.reindex_like(returns).fillna(0.0)

    # Hold today's positions = yesterday's decision (no look-ahead).
    positions = weights.shift(1).fillna(0.0)

    gross = (positions * returns).sum(axis=1)

    # Turnover = L1 change in positions day over day (round-trip captured by cost).
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    cost = cost_per_turn * turnover

    net = gross - cost
    equity = (1.0 + net).cumprod()

    return Backtest(
        returns=net,
        gross_returns=gross,
        turnover=turnover,
        equity=equity,
        cost_per_turn=cost_per_turn,
    )
