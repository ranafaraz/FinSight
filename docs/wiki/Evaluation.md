# Evaluation

## Benchmark setup

The benchmark uses the fully offline synthetic backend so results are deterministic and reproducible without any API keys or downloads.

**Synthetic market:**
- 20 assets, 756 trading days (~3 years)
- Latent per-asset sentiment drawn from AR(1) process; next-day return = alpha * sentiment + noise
- Lexical model recovers latent sentiment at ~0.84 correlation (realistically imperfect)
- Signal IC (rank correlation of signal with next-day return) ~0.052

**Backtest:**
- Cross-sectional z-score weights, top/bottom decile long/short
- Transaction costs: 10 bps per unit of daily turnover
- 7 smoothing window configurations searched; best selected

**Statistical correction:**
- Deflated Sharpe Ratio applied with N=7 (number of configs tried)
- Probabilistic Sharpe with standard benchmark Sharpe of 0.0
- Purged walk-forward CV: 5-fold, purge window 5 days, embargo 2 days

---

## Results

Full table from `python -m evals.harness` (writes `evals/RESULTS.md`):

| Metric | Value | Reads as |
|---|---:|---|
| Sentiment recovery corr | 0.84 | lexical model vs. latent truth |
| Information coefficient (rank IC) | 0.052 | signal vs. next-day return |
| Annualised return (net) | +13.6% | after 10 bps/turn costs |
| Annualised volatility | 7.05% | |
| Sharpe (annualised) | 1.93 | in-sample, selected config |
| Sortino ratio | 2.03 | |
| Max drawdown | -8.21% | |
| Average daily turnover | 0.305 | fraction of portfolio turned over |
| Walk-forward OOS Sharpe | 2.10 | purged + embargoed CV |
| Probabilistic Sharpe (PSR) | 1.000 | P(true Sharpe > 0) |
| **Deflated Sharpe (DSR)** | **0.997** | search-adjusted |
| Event-study long-short CAR | +0.013 | 5-day window |
| Event-study t-statistic | 8.45 | independent confirmation |

---

## Ablation: naive vs. rigorous estimates

The optimism gap shows how much a naive backtest overstates the edge when it ignores selection bias and look-ahead:

| Method | Reported Sharpe | Honest? |
|---|---:|---|
| Naive (best config, in-sample) | 1.93 | No — selection bias not corrected |
| Purged walk-forward CV (OOS) | 2.10 | Yes — OOS (higher here because train set smaller) |
| Deflated Sharpe (DSR) | 0.997 probability | Yes — corrects for N=7 configs tried |
| Scrambled null — naive Sharpe | -1.42 (best found) | — |
| **Scrambled null — DSR** | **0.000** | Correctly identifies false discovery |

The scrambled-null DSR of 0.000 is the key result: the same search procedure applied to random data still finds an apparent winner, but the DSR collapses. This demonstrates the correction is doing real work.

---

## How to reproduce

```bash
# Activate venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Run the full benchmark (writes evals/RESULTS.md)
python -m evals.harness

# Check quality gates (fails if any metric floor is violated)
python -m evals.gate

# Run the scrambled-null test explicitly
FINSIGHT_SCRAMBLE=1 python -m evals.harness

# Run with FinBERT sentiment backend (requires pip install "finsight[finbert]")
FINSIGHT_SENTIMENT_BACKEND=finbert python -m evals.harness
```

CI runs `evals.gate` on every push; it enforces floors on IC, DSR, PSR, and the scrambled-null collapse.
