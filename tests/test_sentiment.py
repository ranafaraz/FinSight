from finsight.sentiment.lexical import LexicalSentiment


def test_polarity_direction():
    s = LexicalSentiment()
    assert s.score("Shares surge as profit beats estimates") > 0.3
    assert s.score("Stock plunges on fraud probe and lawsuit") < -0.3
    assert s.score("The company reported results today") == 0.0


def test_negation_flips():
    s = LexicalSentiment()
    assert s.score("did not beat expectations") < 0
    assert s.score("no growth and weak demand") < 0


def test_score_many_matches_score():
    s = LexicalSentiment()
    texts = ["surge gains", "plunge loss", "neutral filler today"]
    assert s.score_many(texts) == [s.score(t) for t in texts]


def test_bounded():
    s = LexicalSentiment()
    extreme = "surge surge surge beat beat profit rally soar gain win"
    assert -1.0 <= s.score(extreme) <= 1.0
