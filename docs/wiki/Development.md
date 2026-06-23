# Development

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Git
- (Optional) Docker for the one-command run

---

## Setup

```bash
git clone https://github.com/ranafaraz/FinSight.git
cd FinSight

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

The `[dev]` extra includes pytest, ruff, and all test dependencies. No real backend extras are needed for development.

---

## Running tests

```bash
pytest -q                        # 18 tests, all offline
pytest tests/test_no_lookahead.py  # verify timing invariant
pytest tests/ -k "backtest"     # run a subset
```

The test suite is fully offline. It enforces key invariants:
- No lookahead: `weights.shift(1)` is unit-tested to ensure positions cannot see their own return
- Heuristic admissibility: each domain's heuristic never overestimates (checked against known synthetic costs)
- Scrambled null: DSR must collapse to zero when returns are shuffled

---

## Linting

```bash
ruff check .
ruff format .
```

---

## Full evaluation

```bash
python -m evals.harness          # runs full benchmark, writes evals/RESULTS.md
python -m evals.gate             # enforces quality floors; exits non-zero on failure
```

CI runs both on every push.

---

## Code layout

```
src/finsight/
    __init__.py
    config.py           env-var loading, defaults
    types.py            dataclasses (PriceData, Signal, BacktestResult, ...)
    pipeline.py         end-to-end orchestration
    cli.py              Click commands (backtest, score, eventstudy, eval)

    sentiment/
        lexicon.py      finance word lists (positive/negative)
        lexical.py      LexicalSentimentScorer (log-odds over lexicon)
        finbert.py      FinBERTSentimentScorer (lazy import, optional)

    data/
        synthetic.py    SyntheticMarket: deterministic AR(1) + known signal
        loaders.py      CSV and yfinance loaders

    signals/
        factor.py       cross-sectional z-score, EWM smoothing, weight construction

    backtest/
        engine.py       BacktestEngine: shift+1 PnL, txn cost, walk-forward CV

    stats/
        metrics.py      sharpe, sortino, drawdown, PSR
        deflated_sharpe.py  DSR: N-trial correction
        cpcv.py         combinatorial purged CV (CPCV)

    eventstudy/
        study.py        event windows, CAR aggregation, t-statistic

evals/
    harness.py          orchestrates all experiments, writes RESULTS.md
    gate.py             quality gate: floors on every key metric
    RESULTS.md          (generated) — do not edit manually

tests/
    test_no_lookahead.py
    test_synthetic.py
    test_lexical.py
    test_backtest.py
    test_stats.py
    test_eventstudy.py
    ... (18 total)

docs/
    ARCHITECTURE.md     detailed design notes
    DECISIONS.md        "what I built and why"
    demo.gif            animated demo
```

---

## Adding a new sentiment backend

1. Create `src/finsight/sentiment/mybackend.py` with a class that implements:

```python
class MyBackendSentimentScorer:
    def score(self, headlines: list[str]) -> list[float]:
        """Return sentiment in [-1, +1] for each headline."""
        ...
```

2. Register it in `src/finsight/sentiment/__init__.py`:

```python
def get_scorer(backend: str):
    if backend == "mybackend":
        from .mybackend import MyBackendSentimentScorer
        return MyBackendSentimentScorer()
    ...
```

3. Add it to the backend matrix in `config.py` and document it in `Configuration`.

4. Add a test in `tests/test_mybackend.py` verifying the score range and offline behaviour.

---

## Adding a new data backend

1. Create `src/finsight/data/myloader.py` implementing:

```python
class MyLoader:
    def load(self) -> PriceData:
        """Return PriceData with prices, returns, and headlines."""
        ...
```

2. Register in `src/finsight/data/__init__.py` under the `FINSIGHT_DATA_BACKEND` key.

3. Add a lazy import guard so missing optional dependencies fall back to `synthetic` with a warning.
