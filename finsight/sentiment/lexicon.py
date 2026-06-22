"""A compact finance sentiment lexicon.

Inspired by the Loughran-McDonald financial sentiment word lists: general-purpose
sentiment dictionaries mislabel finance text (e.g. "liability", "tax", "cost" are
neutral in 10-Ks), so finance NLP uses domain word lists. This is a small,
hand-curated subset sufficient for the synthetic headlines and as an honest
offline baseline. The optional FinBERT backend is the upgrade path.
"""

from __future__ import annotations

POSITIVE = {
    "beat", "beats", "surge", "surges", "surged", "jump", "jumps", "jumped",
    "rally", "rallies", "soar", "soars", "soared", "gain", "gains", "gained",
    "rise", "rises", "rose", "upgrade", "upgraded", "outperform", "outperforms",
    "strong", "stronger", "record", "growth", "grow", "profit", "profitable",
    "bullish", "boost", "boosts", "boosted", "exceed", "exceeds", "exceeded",
    "robust", "upbeat", "expansion", "win", "wins", "won", "approval", "approved",
    "raises", "raised", "higher", "optimistic", "rebound", "rebounds", "topped",
}

NEGATIVE = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "drop", "drops",
    "dropped", "fall", "falls", "fell", "decline", "declines", "declined",
    "slump", "slumps", "slumped", "downgrade", "downgraded", "underperform",
    "weak", "weaker", "loss", "losses", "lawsuit", "probe", "fraud", "bearish",
    "cut", "cuts", "slash", "slashes", "slashed", "warn", "warns", "warning",
    "recall", "recalls", "default", "bankruptcy", "layoff", "layoffs", "lower",
    "pessimistic", "shortfall", "halt", "halts", "halted", "investigation",
    "tumble", "tumbles", "tumbled", "sink", "sinks", "sank", "fears", "slowdown",
}

# Negators flip the polarity of the next sentiment word within a small window.
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "failed", "failing"}

# Intensity boosters scale the magnitude of the next sentiment word.
BOOSTERS = {
    "sharply": 1.5, "significantly": 1.4, "strongly": 1.4, "deeply": 1.4,
    "slightly": 0.6, "marginally": 0.6, "modestly": 0.7,
}
