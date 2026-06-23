# Architecture

## Overview

FinSight is structured as a pipeline of composable components. Every component has a deterministic offline backend so the full pipeline — including backtesting and statistical correction — runs without external services.

```
src/finsight/
    sentiment/      lexical.py, finbert.py, lexicon.py
    data/           synthetic.py, loaders.py
    signals/        factor.py
    backtest/       engine.py
    stats/          metrics.py, deflated_sharpe.py, cpcv.py
    eventstudy/     study.py
    pipeline.py     orchestrates end-to-end
    cli.py
```

---

## SentimentScorer

The scorer converts a news headline into a scalar in [-1, +1].

**Why log-odds over raw counts?** The lexical backend counts positive and negative finance-lexicon matches and converts to log-odds: `log((pos + 0.5) / (neg + 0.5))`. Raw counts are asymmetric — a headline with 10 positive words and 0 negative words scores differently than one with 1 positive and 1 negative even though the ratio is the same. Log-odds are symmetric and better calibrated for cross-asset comparison.

**Offline synthetic backend:** The synthetic market generates AR(1) latent sentiment for each asset, then perturbs it to produce lexical-model scores. The lexical model recovers the latent sentiment at ~0.84 correlation — realistic for a bag-of-words approach, not a perfect oracle. The gap of ~0.16 is what a stronger model (FinBERT) closes.

**Backend selection:** Set `FINSIGHT_SENTIMENT_BACKEND=lexical` (default) or `finbert`. FinBERT requires `pip install "finsight[finbert]"`. If the extra is not installed, FinSight warns and falls back to lexical rather than crashing.

---

## AlphaSignal construction

1. Aggregate daily per-asset sentiment scores (ewm smoothed over a configurable window).
2. Cross-sectionally z-score across all assets on each day. This makes the signal dollar-neutral by construction — market-wide sentiment moves cancel out.
3. Map z-scores to portfolio weights: long the top decile, short the bottom decile.
4. Apply `weights.shift(1)`: positions are based on yesterday's signal and today's return. This is enforced by a unit test — the backtest cannot see the return it is betting on.

---

## Backtester design

The backtest engine multiplies `weights.shift(1)` by returns, subtracts turnover costs (10 bps per unit of daily turnover), and accumulates PnL.

**Why walk-forward CV over a rolling window?** A fixed rolling window has no mechanism to prevent lookahead leakage: if the same return observation appears near the end of a training window and near the start of the following validation window, the model has effectively seen the future. Walk-forward CV assigns each observation to exactly one validation fold.

**Purging logic:** After each fold split, observations within `PURGE_DAYS` (default: 5) of the boundary are removed from the training set. This prevents the model from training on returns that are temporally correlated with validation returns. An embargo gap is also added: the model does not trade immediately after the purge boundary.

**Why 7 smoothing windows?** The signal requires a smoothing parameter (EWM half-life). Searching over 7 windows is realistic — it's what a practitioner would do. The point of DSR is to correct for exactly this search, not to pretend it didn't happen.

---

## Deflated Sharpe Ratio

The naive Sharpe is the mean return divided by the standard deviation. It has a well-known flaw: if you test N strategies, the expected maximum Sharpe across them is not zero even if all strategies are pure noise. This is the multiple-testing problem.

**The PSR** (Probabilistic Sharpe Ratio) calculates P(true Sharpe > 0) for a single strategy, accounting for non-normality (skewness, kurtosis) and finite sample size.

**The DSR** extends this with a correction for the number of strategies tried: it estimates what the PSR would be if the observed Sharpe were merely the expected maximum across `N` independent trials drawn from a null distribution. A DSR near 1.0 means the Sharpe is too high to be explained by the search alone. A DSR near 0.0 means it is not.

**Demonstration:** FinSight runs the same search over 7 windows on both the real signal and a scrambled (shuffled) null:

| | Real signal | Scrambled null |
|---|---:|---:|
| Best Sharpe found | 1.93 | -1.42 |
| **DSR** | **0.997** | **0.000** |

The scrambled run still finds a "winner" — but DSR correctly identifies it as a false discovery.

---

## Why naive Sharpe overfits on synthetic data but DSR doesn't

The synthetic market has a known ground-truth signal embedded. On one hand, the naive Sharpe overstates the edge because it is the maximum across configurations. On the other hand, the DSR-corrected estimate aligns with the true signal strength because it accounts for the search. On the scrambled data, the naive Sharpe of -1.42 is the best the search found (negative — the null worked) but could easily have been positive with different random noise. The DSR collapses to zero regardless, because the correction term dominates whenever the Sharpe is consistent with chance.
