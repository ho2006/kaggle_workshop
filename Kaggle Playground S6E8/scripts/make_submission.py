from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.submission import best_completed_prediction, make_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a template-validated submission")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prediction")
    group.add_argument("--best", action="store_true")
    args = parser.parse_args()
    config = load_config("configs/baseline.yaml")
    try:
        prediction = best_completed_prediction(config) if args.best else Path(args.prediction)
        submission, metadata = make_submission(config, prediction)
    except (FileNotFoundError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print(f"Submission: {submission}")
    print(f"Metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

