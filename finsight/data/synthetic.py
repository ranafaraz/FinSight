"""Deterministic synthetic market with a *known* sentiment -> return signal.

Why this exists: an offline portfolio repo needs ground truth. We embed a real,
known relationship between each asset's daily news sentiment and its *next-day*
return, so the whole pipeline can be validated end to end:

    latent sentiment  s[t,i]
        |  (generates)            |  (drives, with lag)
        v                         v
    headline text            return r[t+1,i] = beta * s[t,i] + market[t+1] + noise

A correct, leakage-free backtest must recover a positive Sharpe; a shuffled
(null) sentiment must not. The lexical scorer recovers ``s`` only imperfectly
from the text, which is exactly the realistic gap a better model would close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.config import Settings
from finsight.sentiment.lexicon import BOOSTERS, NEGATIVE, POSITIVE
from finsight.types import NewsItem, PricePanel

_POS = sorted(POSITIVE)
_NEG = sorted(NEGATIVE)
_BOOST = sorted(BOOSTERS)
_FILLER = ["the", "company", "reported", "today", "this", "quarter", "as", "amid", "after"]
_NEUTRAL_THRESHOLD = 0.10  # |s| below this -> a no-signal-word headline


def _ticker(i: int) -> str:
    """Stable fake ticker like SYM01."""
    return f"SYM{i + 1:02d}"


def _headline(rng: np.random.Generator, name: str, s: float) -> str:
    """Render a headline whose sentiment words reflect latent sentiment ``s``."""
    mag = abs(float(s))
    if mag < _NEUTRAL_THRESHOLD:
        body = list(rng.choice(_FILLER, size=4, replace=True))
        return f"{name} " + " ".join(body)

    n_words = 1 + int(round(mag * 2))  # 1..3 sentiment words
    pool = _POS if s > 0 else _NEG
    opposite = _NEG if s > 0 else _POS
    words = list(rng.choice(pool, size=n_words, replace=True))

    # ~15% of the time inject an opposing word -> realistic lexical-recovery noise.
    if rng.random() < 0.15:
        words.append(str(rng.choice(opposite)))

    # Occasionally prepend an intensity booster to exercise that handling.
    if mag > 0.6 and rng.random() < 0.4:
        words.insert(0, str(rng.choice(_BOOST)))

    filler = list(rng.choice(_FILLER, size=3, replace=True))
    tokens = [name] + words[:1] + filler[:1] + words[1:] + filler[1:]
    return " ".join(tokens)


def generate_synthetic_panel(settings: Settings | None = None) -> PricePanel:
    """Build a deterministic ``PricePanel`` with embedded sentiment alpha."""
    cfg = settings or Settings.from_env()
    rng = np.random.default_rng(cfg.seed)

    n_assets, n_days = cfg.n_assets, cfg.n_days
    assets = [_ticker(i) for i in range(n_assets)]
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    # Latent daily sentiment per (day, asset) as an AR(1) process: real news
    # sentiment drifts (a story persists for days), which is what makes signal
    # smoothing and turnover control matter.
    rho = cfg.sentiment_persistence
    innov = rng.normal(0.0, cfg.sentiment_innov, size=(n_days, n_assets))
    latent = np.empty((n_days, n_assets))
    latent[0] = innov[0]
    for t in range(1, n_days):
        latent[t] = rho * latent[t - 1] + innov[t]
    latent = latent.clip(-1.0, 1.0)

    # Common market factor and idiosyncratic noise.
    market = rng.normal(0.0003, cfg.market_vol, size=n_days)
    idio = rng.normal(0.0, cfg.noise_vol, size=(n_days, n_assets))

    # return[t] is driven by *yesterday's* sentiment (predictive, no look-ahead).
    lagged = np.vstack([np.zeros((1, n_assets)), latent[:-1]])
    rets = cfg.signal_beta * lagged + market[:, None] + idio

    returns = pd.DataFrame(rets, index=dates, columns=assets)
    prices = 100.0 * (1.0 + returns).cumprod()

    news: list[NewsItem] = []
    for t, date in enumerate(dates):
        for j, asset in enumerate(assets):
            s = float(latent[t, j])
            if abs(s) < _NEUTRAL_THRESHOLD and rng.random() < 0.5:
                continue  # not every quiet day has a headline
            news.append(
                NewsItem(
                    date=date,
                    asset=asset,
                    headline=_headline(rng, asset, s),
                    true_sentiment=s,
                )
            )

    return PricePanel(prices=prices, returns=returns, news=news)
