# Configuration

## Environment variables

All configuration is via environment variables. Defaults are the offline backends; no `.env` file is needed to run the pipeline or tests.

| Variable | Default | Options | Description |
|---|---|---|---|
| `FINSIGHT_SENTIMENT_BACKEND` | `lexical` | `lexical`, `finbert` | Sentiment scoring backend. `finbert` requires `pip install "finsight[finbert]"` |
| `FINSIGHT_DATA_BACKEND` | `synthetic` | `synthetic`, `csv`, `yfinance` | Market data source. `yfinance` requires `pip install "finsight[market]"` |
| `FINSIGHT_SEED` | `42` | integer | Random seed for synthetic data generation |
| `FINSIGHT_N_ASSETS` | `20` | integer | Number of synthetic assets |
| `FINSIGHT_N_DAYS` | `756` | integer | Number of trading days in synthetic history |
| `FINSIGHT_LOOKBACK_WINDOW` | `5` | integer | EWM smoothing half-life for signal (days) |
| `FINSIGHT_PURGE_DAYS` | `5` | integer | Walk-forward CV purge window (days around fold boundary to exclude) |
| `FINSIGHT_TXN_COST_BPS` | `10` | float | Transaction cost per unit turnover (basis points) |
| `FINSIGHT_SCRAMBLE` | `0` | `0`, `1` | Set to 1 to scramble returns (null test) |
| `FINSIGHT_CSV_PATH` | — | file path | Path to CSV price data (required when `DATA_BACKEND=csv`) |

---

## Backend matrix

| Component | Offline default | Optional real backend | Install |
|---|---|---|---|
| Sentiment | `lexical` (finance lexicon, no download) | `finbert` (HuggingFace FinBERT) | `pip install "finsight[finbert]"` |
| Market data | `synthetic` (deterministic, ground-truth signal) | `csv` (user-supplied), `yfinance` (live fetch) | `pip install "finsight[market]"` |

Optional backends degrade gracefully: if FinBERT is not installed, FinSight warns and falls back to `lexical`. If `yfinance` is not installed, it falls back to `synthetic`.

---

## .env.example

```bash
# Copy to .env and fill in values for real backends.
# Everything works offline without this file.

# Sentiment backend (default: lexical)
# FINSIGHT_SENTIMENT_BACKEND=finbert

# Market data backend (default: synthetic)
# FINSIGHT_DATA_BACKEND=yfinance
# FINSIGHT_CSV_PATH=/path/to/prices.csv

# Synthetic market parameters
FINSIGHT_SEED=42
FINSIGHT_N_ASSETS=20
FINSIGHT_N_DAYS=756
FINSIGHT_LOOKBACK_WINDOW=5
FINSIGHT_PURGE_DAYS=5
FINSIGHT_TXN_COST_BPS=10
```

---

## Docker

```bash
# Offline run (synthetic data, lexical sentiment)
docker build -t finsight .
docker run --rm finsight

# With real backends via env var
docker run --rm \
  -e FINSIGHT_SENTIMENT_BACKEND=finbert \
  -e FINSIGHT_DATA_BACKEND=yfinance \
  finsight
```
