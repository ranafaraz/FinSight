"""Classic event-study: cumulative abnormal returns (CAR) around news events.

For each event we estimate a market model (alpha, beta) of the asset on the
equal-weight market over a clean estimation window *before* the event, then
measure abnormal returns AR = r_asset - (alpha + beta*r_market) across an event
window and accumulate them into CAR. Averaging CAR across positive vs negative
news, with a cross-sectional t-stat, is the textbook test that the news carries
information -- here it cross-checks the backtest from an independent angle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finsight.types import PricePanel


@dataclass
class EventStudyResult:
    taus: np.ndarray  # event-time offsets, e.g. -5..+5
    car_pos: np.ndarray  # mean CAR path for positive-news events
    car_neg: np.ndarray  # mean CAR path for negative-news events
    n_pos: int
    n_neg: int
    t_long_short: float  # t-stat of (CAR_pos - CAR_neg) at the final horizon

    @property
    def car_long_short_final(self) -> float:
        return float(self.car_pos[-1] - self.car_neg[-1])


def _market_model_ar(
    asset_ret: pd.Series, mkt_ret: pd.Series, loc: int, pre: int, post: int,
    est_window: int, est_gap: int,
) -> np.ndarray | None:
    """Abnormal-return path over [-pre, +post] for one event at integer ``loc``."""
    est_end = loc - pre - est_gap
    est_start = est_end - est_window
    if est_start < 0 or loc + post >= len(asset_ret):
        return None

    y = asset_ret.iloc[est_start:est_end].to_numpy()
    x = mkt_ret.iloc[est_start:est_end].to_numpy()
    if len(y) < 10 or np.std(x) == 0:
        return None

    beta, alpha = np.polyfit(x, y, 1)  # y ~ alpha + beta*x

    win = slice(loc - pre, loc + post + 1)
    ar = asset_ret.iloc[win].to_numpy() - (alpha + beta * mkt_ret.iloc[win].to_numpy())
    return ar


def event_study(
    panel: PricePanel,
    events: list[tuple[pd.Timestamp, str, float]],
    pre: int = 5,
    post: int = 5,
    threshold: float = 0.4,
    est_window: int = 60,
    est_gap: int = 10,
) -> EventStudyResult:
    """Run the event study over ``events`` = list of (date, asset, sentiment)."""
    returns = panel.returns
    market = panel.market
    idx = returns.index
    pos_paths: list[np.ndarray] = []
    neg_paths: list[np.ndarray] = []

    for date, asset, sentiment in events:
        if abs(sentiment) < threshold or asset not in returns.columns:
            continue
        if date not in idx:
            continue
        loc = idx.get_loc(date)
        if not isinstance(loc, int):
            continue
        ar = _market_model_ar(
            returns[asset], market, loc, pre, post, est_window, est_gap
        )
        if ar is None:
            continue
        car = np.cumsum(ar)
        (pos_paths if sentiment > 0 else neg_paths).append(car)

    taus = np.arange(-pre, post + 1)
    car_pos = np.mean(pos_paths, axis=0) if pos_paths else np.zeros(len(taus))
    car_neg = np.mean(neg_paths, axis=0) if neg_paths else np.zeros(len(taus))

    # Two-sample t-stat on the final-horizon CAR (positive vs negative events).
    t_ls = 0.0
    if len(pos_paths) > 1 and len(neg_paths) > 1:
        fp = np.array([p[-1] for p in pos_paths])
        fn = np.array([p[-1] for p in neg_paths])
        sp, sn = fp.var(ddof=1), fn.var(ddof=1)
        se = np.sqrt(sp / len(fp) + sn / len(fn))
        if se > 0:
            t_ls = float((fp.mean() - fn.mean()) / se)

    return EventStudyResult(
        taus=taus,
        car_pos=car_pos,
        car_neg=car_neg,
        n_pos=len(pos_paths),
        n_neg=len(neg_paths),
        t_long_short=t_ls,
    )
