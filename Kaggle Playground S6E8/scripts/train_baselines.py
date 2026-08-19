from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.reporting import generate_reports
from src.train import run_all_baselines


def main() -> int:
    results = run_all_baselines()
    generate_reports(load_config("configs/baseline.yaml"))
    for result in results:
        score = result.get("full_oof_auc")
        print(f"{result['planned_experiment_id']}: {result['status']} OOF={score}")
    return 0 if any(result["status"] == "COMPLETE" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

