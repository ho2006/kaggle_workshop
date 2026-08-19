from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit
from src.config import load_config
from src.data import DataUnavailableError


def main() -> int:
    try:
        summary = run_audit(load_config("configs/baseline.yaml"))
        print(f"Audit complete: train={summary['train_shape']}, test={summary['test_shape']}")
        return 0
    except DataUnavailableError as exc:
        print(f"BLOCKED_DATA: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

