# FinSight

[![CI](https://github.com/ranafaraz/FinSight/actions/workflows/ci.yml/badge.svg)](https://github.com/ranafaraz/FinSight/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-finsight.dexdevs.com-brightgreen?logo=rocket)](https://finsight.dexdevs.com)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/ranafaraz/FinSight/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/ranafaraz/FinSight/blob/main/LICENSE)

**Turn financial news sentiment into a tradable factor — and prove it isn't luck.**

FinSight builds a cross-sectional, dollar-neutral long/short signal from news sentiment, backtests it with realistic transaction costs, and then subjects the result to the statistics that separate a real edge from backtest overfitting:

- **Deflated Sharpe Ratio (DSR)** — corrects for multiple-testing across configurations tried
- **Probabilistic Sharpe Ratio (PSR)** — P(true Sharpe > 0) under parameter uncertainty
- **Purged + embargoed walk-forward CV** — eliminates look-ahead leakage across fold boundaries
- **News event study (CAR / t-stat)** — independent confirmation from a second statistical angle

The pipeline is **offline-first**: a deterministic synthetic market embeds a known sentiment-to-return relationship so every metric has ground truth. CI runs green with no API keys and no model downloads.

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data backend (env-selectable)"]
        SYN[Synthetic market\nknown signal + AR(1) news]
        CSV[CSV / yfinance]
    end
    subgraph Sentiment["Sentiment backend"]
        LEX[Lexical\nfinance lexicon · offline]
        FB[FinBERT\noptional extra]
    end
    SYN --> NEWS[Headlines]
    CSV --> NEWS
    NEWS --> LEX --> SCORE[Daily asset sentiment]
    NEWS --> FB --> SCORE
    SCORE --> SIG[Cross-sectional z-score\n→ dollar-neutral weights]
    SIG --> BT[Backtest engine\nshift+1 timing · txn costs]
    BT --> STATS[Overfitting-aware stats\nSharpe · PSR · DSR · purged CV]
    SIG -.-> ES[Event study · CAR]
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                    # 18 tests
finsight backtest            # run the strategy
```

## Wiki pages

| Page | What it covers |
|---|---|
| [Architecture](Architecture) | Component design, sentiment backends, backtester, DSR/PSR math |
| [Evaluation](Evaluation) | Benchmark setup, results table, ablation, reproduce commands |
| [Configuration](Configuration) | Env vars, backend matrix, `.env.example` |
| [Development](Development) | Setup, test commands, adding new backends, code layout |
