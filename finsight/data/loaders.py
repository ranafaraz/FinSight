"""CSV / yfinance data loaders (optional, real-data path)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finsight.types import NewsItem, PricePanel


def load_csv_panel(prices_csv: str | Path, news_csv: str | Path | None = None) -> PricePanel:
    """Load a wide price CSV (index = date, columns = tickers) and optional news.

    News CSV columns: ``date,asset,headline`` (sentiment is scored, not supplied).
    """
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True).sort_index()
    returns = prices.pct_change().fillna(0.0)

    news: list[NewsItem] = []
    if news_csv is not None:
        ndf = pd.read_csv(news_csv, parse_dates=["date"])
        for row in ndf.itertuples(index=False):
            news.append(NewsItem(date=row.date, asset=str(row.asset), headline=str(row.headline)))

    return PricePanel(prices=prices, returns=returns, news=news)


def load_yfinance_panel(
    tickers: list[str], start: str, end: str | None = None
) -> PricePanel:  # pragma: no cover - network/optional
    """Download adjusted-close prices via yfinance (requires the ``[market]`` extra)."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "yfinance backend needs the 'market' extra: pip install 'finsight[market]'"
        ) from exc

    raw = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    prices = raw.sort_index().dropna(how="all").ffill()
    returns = prices.pct_change().fillna(0.0)
    return PricePanel(prices=prices, returns=returns, news=[])
