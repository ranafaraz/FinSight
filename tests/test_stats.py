import numpy as np
import pandas as pd

from finsight.stats.cpcv import PurgedWalkForwardCV
from finsight.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from finsight.stats.metrics import max_drawdown, sharpe_ratio


def _series(mean, sd, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n))


def test_sharpe_sign_and_zero():
    assert sharpe_ratio(_series(0.001, 0.01)) > 0
    assert sharpe_ratio(_series(-0.001, 0.01)) < 0
    assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_max_drawdown_negative():
    r = pd.Series([0.1, -0.5, 0.1])
    assert max_drawdown(r) < 0


def test_psr_in_unit_interval_and_monotone():
    weak = probabilistic_sharpe_ratio(_series(0.0005, 0.01, seed=1))
    strong = probabilistic_sharpe_ratio(_series(0.003, 0.01, seed=1))
    assert 0.0 <= weak <= 1.0
    assert 0.0 <= strong <= 1.0
    assert strong > weak


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(0.01, 1) == 0.0
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.01, 10) > 0


def test_deflated_below_psr_when_searching():
    r = _series(0.002, 0.01, seed=3)
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(r, sr_variance=0.02, n_trials=50)
    assert dsr <= psr


def test_walk_forward_is_purged_and_ordered():
    cv = PurgedWalkForwardCV(n_splits=4, horizon=1, embargo=5, min_train=60)
    splits = list(cv.split(500))
    assert len(splits) == 4
    for train, test in splits:
        # No overlap and a real gap (purge+embargo) between train end and test start.
        assert train.max() < test.min()
        assert test.min() - train.max() >= 5
        assert len(set(train) & set(test)) == 0
