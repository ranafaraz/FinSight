"""Environment-driven configuration.

Every heavy component has a deterministic *offline* default so that tests and CI
run green with no API keys and no model downloads. Real backends are opt-in via
environment variables (and the matching pip extras).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    """Resolved runtime settings.

    Backends
    --------
    sentiment_backend : ``lexical`` (offline default) | ``finbert`` (extra ``[finbert]``)
    data_backend      : ``synthetic`` (offline default) | ``csv`` | ``yfinance`` (``[market]``)
    """

    sentiment_backend: str = "lexical"
    data_backend: str = "synthetic"

    # Synthetic-market controls (deterministic).
    seed: int = 7
    n_assets: int = 12
    n_days: int = 756  # ~3 trading years
    signal_beta: float = 0.0032  # strength of true sentiment -> next-day return link
    sentiment_persistence: float = 0.6  # AR(1) coef of latent sentiment (news drifts)
    sentiment_innov: float = 0.40  # innovation std of the AR(1) latent process
    noise_vol: float = 0.013  # idiosyncratic daily vol
    market_vol: float = 0.008  # common market-factor daily vol

    # Portfolio / backtest controls.
    cost_per_turn: float = 0.0010  # 10 bps round-trip cost per unit turnover
    winsorize: float = 0.02  # two-sided winsorisation fraction for raw scores

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            sentiment_backend=_get("FINSIGHT_SENTIMENT_BACKEND", cls.sentiment_backend),
            data_backend=_get("FINSIGHT_DATA_BACKEND", cls.data_backend),
            seed=_get_int("FINSIGHT_SEED", cls.seed),
            n_assets=_get_int("FINSIGHT_N_ASSETS", cls.n_assets),
            n_days=_get_int("FINSIGHT_N_DAYS", cls.n_days),
            signal_beta=_get_float("FINSIGHT_SIGNAL_BETA", cls.signal_beta),
            sentiment_persistence=_get_float(
                "FINSIGHT_SENTIMENT_PERSISTENCE", cls.sentiment_persistence
            ),
            sentiment_innov=_get_float("FINSIGHT_SENTIMENT_INNOV", cls.sentiment_innov),
            noise_vol=_get_float("FINSIGHT_NOISE_VOL", cls.noise_vol),
            market_vol=_get_float("FINSIGHT_MARKET_VOL", cls.market_vol),
            cost_per_turn=_get_float("FINSIGHT_COST_PER_TURN", cls.cost_per_turn),
            winsorize=_get_float("FINSIGHT_WINSORIZE", cls.winsorize),
        )
