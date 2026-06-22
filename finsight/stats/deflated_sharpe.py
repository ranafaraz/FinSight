"""Overfitting-aware Sharpe diagnostics (Bailey & Lopez de Prado).

The Sharpe ratio is a *noisy, biased-upward* statistic: run enough backtests and
one will look great by luck. These functions quantify and correct for that.

- Probabilistic Sharpe Ratio (PSR): probability the true Sharpe exceeds a
  benchmark, given the track record's length, skew and kurtosis.
- Deflated Sharpe Ratio (DSR): PSR evaluated against the *expected maximum*
  Sharpe from N independent trials -- i.e. it deflates the benchmark for the
  number of configurations you tried, the core defence against backtest overfit.

All Sharpe quantities here are in *per-period* (e.g. daily) units, not annualised
-- mixing the two silently breaks the formulas.

References
----------
Bailey, D. & Lopez de Prado, M. (2012) "The Sharpe Ratio Efficient Frontier",
Journal of Risk. Lopez de Prado, M. (2018) "Advances in Financial ML", ch. 8.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import kurtosis, norm, skew

_EULER_GAMMA = 0.5772156649015329


def _period_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    if sd == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / sd)


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``sr_benchmark``) given the sample moments.

    ``sr_benchmark`` is also in per-period units.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3:
        return 0.0
    sr = _period_sharpe(r)
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))  # non-excess kurtosis

    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2
    if denom <= 0:
        # Heavy tails make the estimator unstable; report a degenerate result.
        return 0.0 if sr <= sr_benchmark else 1.0
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum of N i.i.d. trial Sharpes (per-period units).

    ``sr_variance`` is the variance of the per-period Sharpe estimates across the
    ``n_trials`` configurations that were tried.
    """
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    sigma = math.sqrt(sr_variance)
    n = float(n_trials)
    term = (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n) + _EULER_GAMMA * norm.ppf(
        1.0 - 1.0 / (n * math.e)
    )
    return float(sigma * term)


def deflated_sharpe_ratio(
    returns: pd.Series, sr_variance: float, n_trials: int
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max Sharpe of N trials.

    ``sr_variance`` is the variance of per-period Sharpes across the strategies/
    configurations you searched; ``n_trials`` is how many you tried.
    """
    sr_star = expected_max_sharpe(sr_variance, n_trials)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)


def min_track_record_length(
    returns: pd.Series, sr_benchmark: float = 0.0, confidence: float = 0.95
) -> float:
    """Minimum number of observations for PSR to clear ``confidence`` (in periods)."""
    r = returns.dropna()
    if len(r) < 3:
        return float("inf")
    sr = _period_sharpe(r)
    if sr <= sr_benchmark:
        return float("inf")
    g3 = float(skew(r, bias=False))
    g4 = float(kurtosis(r, fisher=False, bias=False))
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr**2
    z = norm.ppf(confidence)
    return float(1.0 + denom * (z / (sr - sr_benchmark)) ** 2)
