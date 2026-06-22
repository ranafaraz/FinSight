"""Sentiment scoring backends (offline lexical default, optional FinBERT)."""

from finsight.sentiment.base import SentimentBackend
from finsight.sentiment.factory import get_backend
from finsight.sentiment.lexical import LexicalSentiment

__all__ = ["SentimentBackend", "get_backend", "LexicalSentiment"]
