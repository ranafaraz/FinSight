"""Resolve a data backend from settings."""

from __future__ import annotations

from finsight.config import Settings
from finsight.data.synthetic import generate_synthetic_panel
from finsight.types import PricePanel


def load_panel(settings: Settings | None = None, **kwargs) -> PricePanel:
    cfg = settings or Settings.from_env()
    backend = cfg.data_backend.lower()

    if backend == "synthetic":
        return generate_synthetic_panel(cfg)

    if backend == "csv":
        from finsight.data.loaders import load_csv_panel

        return load_csv_panel(**kwargs)

    if backend in ("yfinance", "market"):
        from finsight.data.loaders import load_yfinance_panel

        return load_yfinance_panel(**kwargs)

    raise ValueError(f"Unknown data backend: {cfg.data_backend!r}")
