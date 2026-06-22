"""Performance statistics, including overfitting-aware Sharpe diagnostics."""

from finsight.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from finsight.stats.metrics import (
    annualized_return,
    annualized_volatility,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
)

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "information_coefficient",
    "summarize",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
]
