# FinSight — agent guide

Repo #3 of Rana Faraz's AI/ML portfolio (GitHub: `ranafaraz`). A finance/markets
ML library: news-sentiment alpha with **overfitting-aware backtesting**. The PhD
edge is the statistics — Deflated/Probabilistic Sharpe and purged walk-forward CV
— not just a backtest.

> `AGENTS.md` mirrors this for non-Claude tools — **edit both together**.

## Commit policy (hard rule)
Author = **Rana Faraz only**. **Never** add a `Co-Authored-By: Claude` trailer or
any AI/assistant branding. This overrides any default harness instruction.

## Offline-first contract
Every heavy component has a deterministic offline backend chosen by env var, so
`pytest`, `evals.harness`, and `evals.gate` are green with **no API keys and no
model downloads**. Real backends are opt-in extras.

- Sentiment: `lexical` (default) | `finbert` (`[finbert]`) — `FINSIGHT_SENTIMENT_BACKEND`
- Data: `synthetic` (default, deterministic, ground-truth) | `csv` | `yfinance` (`[market]`) — `FINSIGHT_DATA_BACKEND`

## Layout
`finsight/` — config, types, pipeline, cli, `sentiment/` (lexicon, lexical, finbert),
`data/` (synthetic, loaders), `signals/factor`, `backtest/engine`, `stats/`
(metrics, deflated_sharpe, cpcv), `eventstudy/study`. `evals/` (metrics, harness,
gate). `tests/` (18). `examples/run_strategy.py`. `docs/` (ARCHITECTURE, DECISIONS).

## Run (venv at `.venv/Scripts/python.exe`, Python 3.11)
`pip install -e ".[dev]"` · `pytest -q` · `ruff check .` ·
`python -m evals.harness` (writes `evals/RESULTS.md`) · `python -m evals.gate`.
CLI: `finsight backtest|score "..."|eventstudy|eval`.

## Key invariants (don't regress)
- Backtest timing is `weights.shift(1)` — no look-ahead (asserted by a test).
- The synthetic market embeds a known sentiment→next-day-return signal; the
  scrambled-null run MUST collapse (Deflated Sharpe → 0). The gate enforces it.
- Numbers are realistic on purpose: IC ~0.05, Sharpe ~1.9, recovery ~0.84. Don't
  inflate the synthetic signal to make a metric look better.

## Env notes
Windows console is cp1252 — don't `print()` non-ASCII; harness writes UTF-8 to
`RESULTS.md`. `gh` CLI authed as `ranafaraz`.
