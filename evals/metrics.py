"""Eval-specific measurements that exploit the synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.sentiment.base import SentimentBackend
from finsight.types import PricePanel


def sentiment_recovery(panel: PricePanel, backend: SentimentBackend) -> float:
    """Pearson corr between latent (true) sentiment and the backend's score.

    Only defined for the synthetic backend, where ``true_sentiment`` is known.
    Measures how well the offline lexical model reads the generated headlines.
    """
    pairs = [
        (n.true_sentiment, backend.score(n.headline))
        for n in panel.news
        if n.true_sentiment is not None
    ]
    if len(pairs) < 3:
        return float("nan")
    true, pred = zip(*pairs, strict=True)
    return float(np.corrcoef(true, pred)[0, 1])


def shuffled_sentiment(sentiment: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute sentiment within each day across assets -> destroys the alpha.

    Used for the null test: a leakage-free, correctly-deflated strategy must show
    ~0 Sharpe and a near-zero Deflated Sharpe on this scrambled signal.
    """
    rng = np.random.default_rng(seed)
    out = sentiment.copy()
    vals = out.to_numpy(copy=True)
    for i in range(vals.shape[0]):
        rng.shuffle(vals[i])
    return pd.DataFrame(vals, index=out.index, columns=out.columns)
