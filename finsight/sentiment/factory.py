"""Resolve a sentiment backend from settings, degrading to offline on failure."""

from __future__ import annotations

import warnings

from finsight.config import Settings
from finsight.sentiment.base import SentimentBackend
from finsight.sentiment.lexical import LexicalSentiment


def get_backend(settings: Settings | None = None) -> SentimentBackend:
    settings = settings or Settings.from_env()
    backend = settings.sentiment_backend.lower()

    if backend in ("lexical", "offline"):
        return LexicalSentiment()

    if backend == "finbert":
        try:
            from finsight.sentiment.finbert import FinBertSentiment

            return FinBertSentiment()
        except ImportError as exc:
            warnings.warn(
                f"FinBERT unavailable ({exc}); falling back to offline lexical backend.",
                stacklevel=2,
            )
            return LexicalSentiment()

    raise ValueError(f"Unknown sentiment backend: {settings.sentiment_backend!r}")
