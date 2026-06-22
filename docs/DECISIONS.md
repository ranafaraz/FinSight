# Design decisions / what I built

A short record of the non-obvious choices and *why*.

### Deflated Sharpe Ratio as the headline metric
A single backtest Sharpe is nearly meaningless without knowing how many
configurations were tried to get it. I built the eval to *explicitly* search a
grid of smoothing windows, then report the winner's **Deflated Sharpe** (PSR
against the expected maximum Sharpe of N independent trials). The scrambled-null
run, where DSR collapses to 0, demonstrates the deflation actually bites. This is
the core differentiator vs. a hobby backtest.

### Ground-truth synthetic market
An offline portfolio repo can't ship a paid data feed, but it also shouldn't fake
its numbers. The synthetic market embeds a *known* sentiment→return link, so
every claim (IC, Sharpe, event-study CAR, null collapse) is checkable against
ground truth, and the tests assert the pipeline recovers it.

### AR(1) sentiment, and why turnover matters
Early versions used i.i.d. daily sentiment. The signal had positive IC but a
*negative* net Sharpe — transaction costs ate a strategy that fully rebalanced
every day. Rather than hide that, I made latent sentiment persistent (AR(1)),
which is realistic (a news story lasts days) and makes the smoothing-window
search meaningful: smoothing trades signal decay against turnover cost. The
selected window (8 days) is the genuine optimum of that trade-off.

### Leakage control is structural, not aspirational
- Backtest timing is `weights.shift(1)`, asserted by a test.
- Cross-validation is purged + embargoed walk-forward, never plain k-fold.
- The market factor is removed by cross-sectional demeaning, so the signal is a
  pure relative-value bet, not a disguised market beta.

### Offline-first, real backends optional
Lexical sentiment and a synthetic market keep CI green with zero keys/downloads.
FinBERT and yfinance are pip extras, lazily imported, with graceful fallback —
importing FinSight never pulls in `torch`.

### Honest, realistic magnitudes
Signal IC is tuned to ~0.05 (realistic for daily news alpha), sentiment recovery
to ~0.84 (not 1.0 — the gap is what a better model would close), and costs are
charged at 10 bps/turn. The point is to look like a real research result, not a
too-good-to-be-true demo.

## Roadmap
- FinBERT results column alongside the lexical baseline.
- Combinatorial purged cross-validation (CPCV) for a full PBO estimate.
- A real-data adapter walkthrough (yfinance + a public headline set).
- Recorded `docs/demo.gif`.
