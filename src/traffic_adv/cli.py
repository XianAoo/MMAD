from __future__ import annotations

import argparse
import json

from traffic_adv.config import load_experiment
from traffic_adv.defense import run_defense_experiment
from traffic_adv.pipeline import attack, evaluate, evaluate_clean, run, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic adversarial experiment runner")
    parser.add_argument(
        "command", choices=["train", "attack", "evaluate", "evaluate-clean", "run", "defense"]
    )
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "defense":
        result = run_defense_experiment(args.config)
    else:
        config = load_experiment(args.config)
        actions = {
            "train": train,
            "attack": attack,
            "evaluate": evaluate,
            "evaluate-clean": evaluate_clean,
            "run": run,
        }
        result = actions[args.command](config)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    elif result is not None:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

