"""Core data structures shared across FinSight.

These are deliberately thin dataclasses over pandas objects so that every stage of
the pipeline (data -> sentiment -> signal -> backtest -> stats) speaks the same
vocabulary and the offline/real backends are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TRADING_DAYS = 252  # annualisation convention used throughout


@dataclass
class NewsItem:
    """A single dated headline attached to one asset."""

    date: pd.Timestamp
    asset: str
    headline: str
    # Latent sentiment in [-1, 1] used by the synthetic generator. Real feeds leave this None.
    true_sentiment: float | None = None


@dataclass
class PricePanel:
    """A panel of daily prices/returns indexed by date, columns = assets.

    ``returns`` are simple daily returns. ``market`` is the cross-sectional mean
    return per day (an equal-weight proxy for the market factor), used by the
    event study and for neutralisation.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    news: list[NewsItem] = field(default_factory=list)

    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def market(self) -> pd.Series:
        return self.returns.mean(axis=1)


@dataclass
class Signal:
    """A dated, cross-sectional alpha signal: index = date, columns = assets.

    ``scores`` are the raw (e.g. z-scored sentiment) values; ``weights`` are the
    dollar-neutral portfolio weights derived from them.
    """

    scores: pd.DataFrame
    weights: pd.DataFrame


@dataclass
class Backtest:
    """Output of the backtest engine."""

    returns: pd.Series  # net daily portfolio returns
    gross_returns: pd.Series  # before transaction costs
    turnover: pd.Series  # per-day turnover (sum of |Δweight|)
    equity: pd.Series  # cumulative net equity curve (starts at 1.0)
    cost_per_turn: float
