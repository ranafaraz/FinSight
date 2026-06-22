"""Sentiment backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SentimentBackend(Protocol):
    """Maps a headline to a polarity score in [-1, 1] (negative .. positive)."""

    name: str

    def score(self, text: str) -> float: ...

    def score_many(self, texts: list[str]) -> list[float]: ...
