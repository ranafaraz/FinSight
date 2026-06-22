"""End-to-end strategy pipeline: data -> sentiment -> signal -> backtest -> stats."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finsight.backtest.engine import run_backtest
from finsight.config import Settings
from finsight.data.factory import load_panel
from finsight.sentiment.factory import get_backend
from finsight.signals.factor import build_signal, score_news
from finsight.stats.metrics import information_coefficient, summarize
from finsight.types import Backtest, PricePanel, Signal


@dataclass
class StrategyResult:
    settings: Settings
    panel: PricePanel
    sentiment: pd.DataFrame
    signal: Signal
    backtest: Backtest
    metrics: dict[str, float]
    information_coefficient: float

    def report(self) -> str:
        m = self.metrics
        return (
            f"FinSight strategy [{self.settings.sentiment_backend} sentiment / "
            f"{self.settings.data_backend} data]\n"
            f"  IC (rank)      : {self.information_coefficient:+.3f}\n"
            f"  Ann. return    : {m['ann_return']:+.2%}\n"
            f"  Ann. vol       : {m['ann_vol']:.2%}\n"
            f"  Sharpe         : {m['sharpe']:.2f}\n"
            f"  Sortino        : {m['sortino']:.2f}\n"
            f"  Max drawdown   : {m['max_drawdown']:.2%}\n"
            f"  Avg turnover   : {self.backtest.turnover.mean():.3f}/day"
        )


def run_strategy(
    settings: Settings | None = None, panel: PricePanel | None = None
) -> StrategyResult:
    """Run the full strategy and return a populated ``StrategyResult``."""
    cfg = settings or Settings.from_env()
    panel = panel if panel is not None else load_panel(cfg)

    backend = get_backend(cfg)
    sentiment = score_news(panel, backend)
    signal = build_signal(sentiment, winsorize=cfg.winsorize)
    backtest = run_backtest(signal, panel.returns, cost_per_turn=cfg.cost_per_turn)

    metrics = summarize(backtest.returns)
    ic = information_coefficient(signal.scores, panel.returns)

    return StrategyResult(
        settings=cfg,
        panel=panel,
        sentiment=sentiment,
        signal=signal,
        backtest=backtest,
        metrics=metrics,
        information_coefficient=ic,
    )
