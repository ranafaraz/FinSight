"""FinSight command-line interface.

    finsight backtest            run the offline strategy and print metrics
    finsight score "headline"    score a single headline with the active backend
    finsight eventstudy          run the news event study (CAR)
    finsight eval                run the full offline eval harness
"""

from __future__ import annotations

import argparse
import sys

from finsight.config import Settings


def _cmd_backtest(args: argparse.Namespace) -> int:
    from finsight.pipeline import run_strategy

    res = run_strategy(Settings.from_env())
    print(res.report())
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from finsight.sentiment.factory import get_backend

    backend = get_backend(Settings.from_env())
    score = backend.score(args.text)
    label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    print(f"[{backend.name}] {score:+.3f} ({label})  {args.text!r}")
    return 0


def _cmd_eventstudy(args: argparse.Namespace) -> int:
    from finsight.data.factory import load_panel
    from finsight.eventstudy.study import event_study

    cfg = Settings.from_env()
    panel = load_panel(cfg)
    events = [(n.date, n.asset, n.true_sentiment or 0.0) for n in panel.news]
    res = event_study(panel, events, threshold=args.threshold)
    print(
        f"Event study: {res.n_pos} positive / {res.n_neg} negative events\n"
        f"  CAR(+) final : {res.car_pos[-1]:+.4f}\n"
        f"  CAR(-) final : {res.car_neg[-1]:+.4f}\n"
        f"  long-short   : {res.car_long_short_final:+.4f}  (t = {res.t_long_short:.2f})"
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from evals.harness import main as eval_main

    return eval_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="finsight", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("backtest", help="run the offline strategy and print metrics").set_defaults(
        func=_cmd_backtest
    )

    sp = sub.add_parser("score", help="score a single headline")
    sp.add_argument("text", help="headline text to score")
    sp.set_defaults(func=_cmd_score)

    se = sub.add_parser("eventstudy", help="run the news event study")
    se.add_argument("--threshold", type=float, default=0.4, help="min |sentiment| for an event")
    se.set_defaults(func=_cmd_eventstudy)

    sub.add_parser("eval", help="run the full offline eval harness").set_defaults(func=_cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
