"""Optional FinBERT sentiment backend (requires the ``[finbert]`` extra).

Lazily imports ``transformers``/``torch`` so that importing FinSight never pulls
in heavy deps. If they are missing, instantiating this backend raises a clear
error and the factory falls back to the offline lexical backend.
"""

from __future__ import annotations


class FinBertSentiment:
    """Wraps ProsusAI/finbert and maps its 3-way label to a [-1, 1] polarity."""

    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise ImportError(
                "FinBERT backend needs the 'finbert' extra: pip install 'finsight[finbert]'"
            ) from exc
        self._pipe = pipeline("sentiment-analysis", model=model_name, top_k=None)

    @staticmethod
    def _to_polarity(scores: list[dict]) -> float:
        by_label = {d["label"].lower(): d["score"] for d in scores}
        return float(by_label.get("positive", 0.0) - by_label.get("negative", 0.0))

    def score(self, text: str) -> float:  # pragma: no cover - requires model download
        out = self._pipe(text)[0]
        return self._to_polarity(out)

    def score_many(self, texts: list[str]) -> list[float]:  # pragma: no cover
        return [self._to_polarity(o) for o in self._pipe(texts)]
