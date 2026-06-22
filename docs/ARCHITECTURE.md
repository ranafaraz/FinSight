# Architecture

FinSight is a thin, composable pipeline. Every stage speaks the dataclasses in
[`finsight/types.py`](../finsight/types.py), and every heavy component is chosen
by an env var so the offline path needs no keys or downloads.

```
data backend ──▶ news headlines ──▶ sentiment backend ──▶ daily asset sentiment
                                                                   │
                                                          signal construction
                                                  (winsorise → cross-sectional
                                                   z-score → dollar-neutral wts)
                                                                   │
                          ┌────────────────────────────────────────┤
                          ▼                                         ▼
                   backtest engine                            event study
              (shift+1 timing, txn costs)                  (market-model CAR)
                          │
                          ▼
        statistics: Sharpe / Sortino / drawdown / turnover,
        Probabilistic & Deflated Sharpe, purged walk-forward CV
```

## Stages

1. **Data** (`finsight/data/`). `synthetic.py` is the offline default: a
   deterministic, seeded market where each asset's daily news sentiment follows
   an AR(1) process and drives its *next-day* return with a known coefficient.
   Because the relationship is known, the eval has ground truth. `loaders.py`
   adds CSV and (optional) yfinance paths.

2. **Sentiment** (`finsight/sentiment/`). `lexical.py` scores headlines with a
   compact Loughran-McDonald-style finance lexicon plus negation and intensity
   handling — fully deterministic, no model. `finbert.py` is the opt-in upgrade
   (lazily imported). The factory degrades to lexical if FinBERT is absent.

3. **Signal** (`finsight/signals/factor.py`). Aggregate same-day headlines per
   asset, winsorise, cross-sectionally z-score (this demeans each day, which
   neutralises the common market factor), then convert to dollar-neutral weights
   with unit gross exposure.

4. **Backtest** (`finsight/backtest/engine.py`). Vectorised. The one rule that
   kills naive backtests is enforced structurally: weights formed from day-`t`
   news are held on day `t+1` (`weights.shift(1)`), and turnover is charged at a
   configurable per-unit cost.

5. **Statistics** (`finsight/stats/`).
   - `metrics.py` — annualised Sharpe, Sortino, max drawdown, rank IC.
   - `deflated_sharpe.py` — Probabilistic Sharpe Ratio, expected maximum Sharpe
     across N trials, and the Deflated Sharpe Ratio.
   - `cpcv.py` — purged, embargoed walk-forward splitter for leakage-free OOS
     evaluation.

6. **Event study** (`finsight/eventstudy/study.py`). Estimates a market model
   per event on a clean pre-event window and accumulates abnormal returns into a
   CAR path, with a cross-sectional t-stat on the positive-vs-negative spread.

## Determinism & CI

The synthetic market is seeded; the lexical model is rule-based. So the eval
numbers are reproducible bit-for-bit, and [`evals/gate.py`](../evals/gate.py)
can enforce hard floors (including that the *scrambled-signal null* stays quiet)
on every push across Python 3.10–3.12.
