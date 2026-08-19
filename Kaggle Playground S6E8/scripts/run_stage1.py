from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit import run_audit
from src.config import load_config
from src.data import DataUnavailableError
from src.folds import create_and_save_folds
from src.reporting import generate_reports
from src.submission import best_completed_prediction, make_submission
from src.train import run_all_baselines


def main() -> int:
    config = load_config("configs/baseline.yaml")
    blocked = False
    try:
        run_audit(config)
    except DataUnavailableError as exc:
        print(f"AUDIT BLOCKED_DATA: {exc}")
        blocked = True
    try:
        create_and_save_folds(config)
    except DataUnavailableError as exc:
        print(f"FOLDS BLOCKED_DATA: {exc}")
        blocked = True
    results = run_all_baselines()
    if not any(result["status"] == "COMPLETE" for result in results):
        blocked = True
    try:
        prediction = best_completed_prediction(config)
        submission, _ = make_submission(config, prediction)
        print(f"SUBMISSION COMPLETE: {submission}")
    except (FileNotFoundError, ValueError, DataUnavailableError) as exc:
        print(f"SUBMISSION BLOCKED: {exc}")
        blocked = True
    generate_reports(config)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=False)
    if test.returncode:
        print(f"TESTS FAILED: exit code {test.returncode}")
        return test.returncode
    print("TESTS COMPLETE")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())

