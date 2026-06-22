import numpy as np
import pandas as pd

from finsight.backtest.engine import run_backtest
from finsight.config import Settings
from finsight.data.synthetic import generate_synthetic_panel
from finsight.pipeline import run_strategy
from finsight.sentiment.lexical import LexicalSentiment
from finsight.signals.factor import build_signal, score_news


def test_synthetic_panel_is_deterministic():
    a = generate_synthetic_panel(Settings(n_days=120))
    b = generate_synthetic_panel(Settings(n_days=120))
    pd.testing.assert_frame_equal(a.returns, b.returns)
    assert len(a.news) == len(b.news)


def test_signal_is_dollar_neutral():
    panel = generate_synthetic_panel(Settings(n_days=200))
    sentiment = score_news(panel, LexicalSentiment())
    signal = build_signal(sentiment)
    active = signal.weights.loc[signal.weights.abs().sum(axis=1) > 0]
    # Net exposure ~ 0, gross exposure ~ 1 on active days.
    assert active.sum(axis=1).abs().max() < 1e-9
    assert np.allclose(active.abs().sum(axis=1), 1.0, atol=1e-9)


def test_no_lookahead_shift():
    # Positions on day t must equal weights from day t-1.
    panel = generate_synthetic_panel(Settings(n_days=150))
    sentiment = score_news(panel, LexicalSentiment())
    signal = build_signal(sentiment)
    bt = run_backtest(signal, panel.returns)
    expected = (signal.weights.shift(1).fillna(0.0) * panel.returns).sum(axis=1)
    pd.testing.assert_series_equal(bt.gross_returns, expected, check_names=False)


def test_strategy_beats_null():
    res = run_strategy(Settings(n_days=400))
    assert res.information_coefficient > 0
    assert res.metrics["sharpe"] > 0
    assert res.report()  # smoke
