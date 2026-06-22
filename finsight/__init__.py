"""FinSight: offline-first news-sentiment alpha with overfitting-aware backtesting."""

from finsight.config import Settings
from finsight.pipeline import StrategyResult, run_strategy
from finsight.types import Backtest, NewsItem, PricePanel, Signal

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "run_strategy",
    "StrategyResult",
    "Backtest",
    "NewsItem",
    "PricePanel",
    "Signal",
    "__version__",
]
