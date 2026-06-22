# FinSight

[![CI](https://github.com/ranafaraz/FinSight/actions/workflows/ci.yml/badge.svg)](https://github.com/ranafaraz/FinSight/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Turn financial news sentiment into a tradable factor — and prove it isn't luck.**

FinSight builds a cross-sectional, dollar-neutral long/short signal from news
sentiment, backtests it with realistic transaction costs, and then subjects the
result to the statistics that separate a real edge from backtest overfitting:
the **Deflated Sharpe Ratio**, the **Probabilistic Sharpe Ratio**, and
**purged + embargoed walk-forward cross-validation** (Bailey & López de Prado).
A classic news **event study** cross-checks the signal from an independent angle.

It is **offline-first**: a deterministic synthetic market embeds a *known*
sentiment→return relationship, so the whole pipeline has ground truth and CI runs
green with **no API keys and no model downloads**. Real backends (FinBERT
sentiment, yfinance prices) are opt-in via pip extras.

> Most backtests answer "what was the Sharpe?" FinSight answers the question that
> actually matters: **"given how many configurations I tried, how likely is this
> Sharpe to survive out of sample?"**

## Demo

```console
$ finsight backtest
FinSight strategy [lexical sentiment / synthetic data]
  IC (rank)      : +0.052
  Ann. return    : +13.60%
  Ann. vol       : 7.05%
  Sharpe         : 1.93
  Sortino        : 2.03
  Max drawdown   : -8.21%
  Avg turnover   : 0.305/day

$ finsight score "Shares plunge as the firm warns and cuts guidance"
[lexical] -1.000 (negative)  'Shares plunge as the firm warns and cuts guidance'
```

![demo](docs/demo.gif)

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data backend (env-selectable)"]
        SYN[Synthetic market<br/>known signal + AR(1) news]
        CSV[CSV / yfinance]
    end
    subgraph Sentiment["Sentiment backend"]
        LEX[Lexical<br/>finance lexicon · offline]
        FB[FinBERT<br/>optional extra]
    end
    SYN --> NEWS[Headlines]
    CSV --> NEWS
    NEWS --> LEX --> SCORE[Daily asset sentiment]
    NEWS --> FB --> SCORE
    SCORE --> SIG[Cross-sectional z-score<br/>→ dollar-neutral weights]
    SIG --> BT[Backtest engine<br/>shift+1 timing · txn costs]
    BT --> STATS[Stats]
    SIG -.-> ES[Event study · CAR]
    Data -.-> ES
    subgraph STATS["Overfitting-aware statistics"]
        SR[Sharpe / Sortino / drawdown]
        PSR[Probabilistic Sharpe]
        DSR[Deflated Sharpe<br/>N-trial correction]
        CV[Purged walk-forward CV]
    end
```

## The metric that matters

The strategy is selected from a search over 7 smoothing windows. A naive report
would quote the winner's Sharpe and stop. FinSight instead **deflates** it:

| | Real signal | Scrambled (null) signal |
|---|---:|---:|
| Annualised Sharpe | **1.93** | −1.42 |
| **Deflated Sharpe Ratio** | **0.997** | **0.000** |

When the alpha is shuffled away, the *same search* still finds a "best"
configuration — but the Deflated Sharpe correctly collapses to zero. That gap is
the whole product.

## Full offline eval

Reproduce with `python -m evals.harness` (writes [`evals/RESULTS.md`](evals/RESULTS.md)):

| Metric | Value | Reads as |
|---|---:|---|
| Sentiment recovery corr | 0.84 | lexical model vs. latent truth |
| Information coefficient (rank IC) | 0.052 | signal vs. next-day return |
| Annualised return (net) | +13.6% | after 10 bps/turn costs |
| Sharpe (annualised) | 1.93 | in-sample, selected config |
| Walk-forward OOS Sharpe | 2.10 | purged + embargoed |
| Probabilistic Sharpe (PSR) | 1.000 | P(true Sharpe > 0) |
| **Deflated Sharpe (DSR)** | **0.997** | search-adjusted |
| Event-study long-short CAR (t) | +0.013 (**t = 8.45**) | independent confirmation |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                    # 18 tests
python -m evals.harness      # full eval table -> evals/RESULTS.md
python -m evals.gate         # CI quality gate (floors on every metric)

finsight backtest            # run the strategy
finsight score "..."         # score a headline
finsight eventstudy          # news event study (CAR)
```

One-command Docker run (offline):

```bash
docker build -t finsight . && docker run --rm finsight
```

## Backend matrix

| Component | Offline default | Optional real backend | Env var |
|---|---|---|---|
| Sentiment | `lexical` (finance lexicon) | `finbert` (`pip install "finsight[finbert]"`) | `FINSIGHT_SENTIMENT_BACKEND` |
| Market data | `synthetic` (deterministic) | `csv`, `yfinance` (`[market]`) | `FINSIGHT_DATA_BACKEND` |

Optional deps are imported lazily and degrade gracefully: if FinBERT isn't
installed, FinSight warns and falls back to the offline lexical backend rather
than crashing.

## Why this is honest

- **No look-ahead.** Positions are `weights.shift(1)` — a position can never see
  the return that justifies it (enforced by a test).
- **Costs are real.** Turnover is charged at 10 bps per unit; the high-turnover
  raw signal is deliberately *not* profitable until smoothing controls turnover.
- **The hard cases stay in.** The synthetic lexical model recovers sentiment at
  ~0.84 corr, *not* 1.0 — the gap is exactly what a stronger model (FinBERT)
  would close. The signal IC (~0.05) is in the realistic range for daily news
  alpha, not a fantasy.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the design write-up.

## License

MIT — see [LICENSE](LICENSE).
