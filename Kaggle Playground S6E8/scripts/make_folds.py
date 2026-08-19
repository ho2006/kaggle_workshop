from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import DataUnavailableError
from src.folds import create_and_save_folds


def main() -> int:
    try:
        folds = create_and_save_folds(load_config("configs/baseline.yaml"))
        print(f"Fold file ready: {len(folds):,} rows")
        return 0
    except DataUnavailableError as exc:
        print(f"BLOCKED_DATA: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

