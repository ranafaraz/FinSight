"""Standard performance metrics (annualised by the 252-day convention)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.types import TRADING_DAYS


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of a daily return series (rf = 0)."""
    r = returns.dropna()
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised Sortino ratio (downside deviation in the denominator)."""
    r = returns.dropna()
    downside = r[r < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods))


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float(r.mean() * periods)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the equity curve (negative number)."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def information_coefficient(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """Mean daily cross-sectional Spearman rank IC between signal and next return.

    Signal on day t is ranked against the return realised on day t+1.
    """
    sig = signal.shift(1)  # align signal[t] with return[t] -> predicts current return
    common = sig.index.intersection(forward_returns.index)
    sig = sig.loc[common]
    fwd = forward_returns.loc[common]

    ics: list[float] = []
    for date in common:
        s = sig.loc[date]
        f = fwd.loc[date]
        mask = s.notna() & f.notna() & (s != 0)
        if mask.sum() < 3:
            continue
        ic = s[mask].rank().corr(f[mask].rank())
        if pd.notna(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def summarize(returns: pd.Series) -> dict[str, float]:
    """Bundle the headline metrics for reporting."""
    return {
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }
