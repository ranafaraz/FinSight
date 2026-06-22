from finsight.cli import main


def test_cli_score(capsys):
    assert main(["score", "Shares surge as profit beats estimates"]) == 0
    out = capsys.readouterr().out
    assert "positive" in out


def test_cli_backtest(capsys):
    assert main(["backtest"]) == 0
    assert "Sharpe" in capsys.readouterr().out


def test_cli_eventstudy(capsys):
    assert main(["eventstudy", "--threshold", "0.5"]) == 0
    assert "Event study" in capsys.readouterr().out
