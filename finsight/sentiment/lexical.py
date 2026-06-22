"""Deterministic, offline lexical sentiment scorer.

Tokenises a headline, counts positive/negative finance terms, and applies simple
negation and intensity handling. Output is a polarity in [-1, 1]. This is the
default backend and requires no model download — CI runs it everywhere.
"""

from __future__ import annotations

import re

from finsight.sentiment.lexicon import BOOSTERS, NEGATIVE, NEGATORS, POSITIVE

_TOKEN_RE = re.compile(r"[a-z]+")
_NEG_WINDOW = 3  # how many following tokens a negator/booster affects


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalSentiment:
    """Lexicon-based polarity scorer."""

    name = "lexical"

    def score(self, text: str) -> float:
        tokens = _tokenize(text)
        total = 0.0
        hits = 0
        neg_countdown = 0
        boost = 1.0
        boost_countdown = 0
        for tok in tokens:
            # Modifiers open a short window over the *following* tokens.
            if tok in NEGATORS:
                neg_countdown = _NEG_WINDOW
                continue
            if tok in BOOSTERS:
                boost = BOOSTERS[tok]
                boost_countdown = _NEG_WINDOW
                continue

            polarity = 0.0
            if tok in POSITIVE:
                polarity = 1.0
            elif tok in NEGATIVE:
                polarity = -1.0

            if polarity != 0.0:
                magnitude = boost if boost_countdown > 0 else 1.0
                if neg_countdown > 0:
                    polarity = -polarity
                total += polarity * magnitude
                hits += 1
                # A modifier is consumed by the first sentiment word it reaches.
                neg_countdown = 0
                boost = 1.0
                boost_countdown = 0
            else:
                if neg_countdown > 0:
                    neg_countdown -= 1
                if boost_countdown > 0:
                    boost_countdown -= 1

        if hits == 0:
            return 0.0
        # Normalise by hit count so long headlines aren't over-weighted, then squash.
        raw = total / hits
        return max(-1.0, min(1.0, raw))

    def score_many(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]
