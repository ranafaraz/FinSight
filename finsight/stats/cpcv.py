"""Purged, embargoed walk-forward cross-validation for time-series alpha.

Naive k-fold leaks: in finance the label at time ``t`` depends on a *forward*
return window, so a training point right before the test set sees information
that overlaps the test labels. Lopez de Prado's fix is to (a) **purge** training
observations whose label horizon overlaps the test set and (b) add an
**embargo** gap after the test set before training resumes.

This module implements an expanding-window walk-forward variant: train only on
the past, with a ``horizon + embargo`` gap separating train from test. That gap
is what makes the out-of-sample numbers trustworthy.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PurgedWalkForwardCV:
    """Expanding-window, purged + embargoed walk-forward splitter.

    Parameters
    ----------
    n_splits : number of out-of-sample test blocks.
    horizon  : label horizon in periods (forward window the label depends on);
               this many training points immediately before each test block are
               purged.
    embargo  : extra gap (in periods) inserted before the test block on top of
               ``horizon`` to absorb serial correlation.
    min_train : minimum training observations required to emit a fold.
    """

    def __init__(
        self, n_splits: int = 5, horizon: int = 1, embargo: int = 5, min_train: int = 60
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        self.min_train = min_train

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer-position arrays."""
        if n_samples <= self.min_train + self.n_splits:
            raise ValueError("Not enough samples for the requested splits.")

        test_size = (n_samples - self.min_train) // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError("Test blocks would be empty; reduce n_splits.")

        gap = self.horizon + self.embargo
        for i in range(self.n_splits):
            test_start = self.min_train + (i + 1) * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            test_start = min(test_start, n_samples)
            test_end = min(test_end, n_samples)
            if test_start >= test_end:
                continue

            train_end = test_start - gap
            if train_end < self.min_train:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
