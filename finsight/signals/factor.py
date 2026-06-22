"""Turn scored news into a cross-sectional, dollar-neutral alpha signal.

Pipeline: aggregate same-day headlines per asset -> winsorise -> cross-sectional
z-score (this demeans each day, neutralising the market factor) -> convert to
dollar-neutral long/short weights with unit gross exposure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.sentiment.base import SentimentBackend
from finsight.types import PricePanel, Signal


def score_news(panel: PricePanel, backend: SentimentBackend) -> pd.DataFrame:
    """Score every headline and aggregate to a (date x asset) sentiment frame.

    Cells with no news are left as NaN (no opinion). The frame is reindexed onto
    the full trading calendar so it aligns with returns.
    """
    if not panel.news:
        return pd.DataFrame(0.0, index=panel.returns.index, columns=panel.returns.columns)

    headlines = [n.headline for n in panel.news]
    scores = backend.score_many(headlines)
    raw = pd.DataFrame(
        {
            "date": [n.date for n in panel.news],
            "asset": [n.asset for n in panel.news],
            "score": scores,
        }
    )
    daily = raw.groupby(["date", "asset"])["score"].mean().unstack("asset")
    return daily.reindex(index=panel.returns.index, columns=panel.returns.columns)


def _winsorize_rows(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if frac <= 0:
        return df
    lo = df.quantile(frac, axis=1)
    hi = df.quantile(1.0 - frac, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def build_signal(sentiment: pd.DataFrame, winsorize: float = 0.02) -> Signal:
    """Cross-sectionally standardise sentiment and form dollar-neutral weights."""
    # A day with no news at all carries no position.
    filled = sentiment.copy()

    wins = _winsorize_rows(filled, winsorize)
    mu = wins.mean(axis=1)
    sigma = wins.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = wins.sub(mu, axis=0).div(sigma, axis=0)
    z = z.fillna(0.0)

    gross = z.abs().sum(axis=1).replace(0.0, np.nan)
    weights = z.div(gross, axis=0).fillna(0.0)

    return Signal(scores=z, weights=weights)
