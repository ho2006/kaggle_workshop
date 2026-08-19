from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import project_path
from src.data import load_data
from src.predict import read_prediction_file
from src.utils import ensure_output_dirs, utc_now, write_json


def _experiment_id_from_prediction(path: Path) -> str:
    suffix = "_test"
    return path.stem[:-len(suffix)] if path.stem.endswith(suffix) else path.stem


def best_completed_prediction(config: dict[str, Any]) -> Path:
    paths = ensure_output_dirs(config)
    ledger_path = paths["experiments"] / "experiments.csv"
    if not ledger_path.exists():
        raise FileNotFoundError("No experiment ledger exists")
    ledger = pd.read_csv(ledger_path)
    complete = ledger.loc[ledger["status"] == "COMPLETE"].copy()
    if complete.empty:
        raise ValueError("No completed experiment is available for submission")
    complete["full_oof_auc"] = pd.to_numeric(complete["full_oof_auc"], errors="coerce")
    complete["std_fold_auc"] = pd.to_numeric(complete["std_fold_auc"], errors="coerce")
    best = complete.sort_values(
        ["full_oof_auc", "std_fold_auc"], ascending=[False, True]
    ).iloc[0]
    prediction = paths["predictions"] / f"{best['experiment_id']}_test.csv"
    if not prediction.exists():
        raise FileNotFoundError(f"Best experiment prediction is missing: {prediction}")
    return prediction


def make_submission(config: dict[str, Any], prediction_path: str | Path) -> tuple[Path, Path]:
    paths = ensure_output_dirs(config)
    bundle = load_data(config)
    prediction_path = project_path(prediction_path)
    predictions = read_prediction_file(prediction_path, len(bundle.test))
    identifier = bundle.schema.id_column
    if identifier not in predictions.columns:
        raise ValueError(f"Prediction file must contain ID column {identifier!r}")
    if not predictions[identifier].equals(bundle.test[identifier]):
        raise ValueError("Prediction IDs/order do not match test.csv")
    if not bundle.sample_submission[identifier].equals(bundle.test[identifier]):
        raise ValueError("Sample submission IDs/order do not match test.csv")
    prediction_column = bundle.schema.prediction_columns[0]
    submission = bundle.sample_submission.copy()
    submission[prediction_column] = predictions["prediction"].to_numpy(dtype=float)
    if list(submission.columns) != list(bundle.sample_submission.columns):
        raise ValueError("Submission columns/order changed from the sample template")
    if submission.isna().any().any() or not np.isfinite(
        submission[prediction_column].to_numpy(dtype=float)
    ).all():
        raise ValueError("Submission contains missing or non-finite values")
    if not submission[prediction_column].between(0, 1).all():
        raise ValueError("Submission probabilities are outside [0, 1]")
    experiment_id = _experiment_id_from_prediction(prediction_path)
    output = paths["submissions"] / f"submission_{experiment_id}.csv"
    if output.exists():
        output = paths["submissions"] / (
            f"submission_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    submission.to_csv(output, index=False)
    reread = pd.read_csv(output)
    pd.testing.assert_frame_equal(reread, submission, check_dtype=False)
    ledger_path = paths["experiments"] / "experiments.csv"
    experiment: dict[str, Any] = {}
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        match = ledger.loc[ledger["experiment_id"] == experiment_id]
        if not match.empty:
            experiment = match.iloc[-1].to_dict()
    metadata = {
        "experiment_id": experiment_id,
        "model": experiment.get("model"),
        "feature_version": experiment.get("feature_version"),
        "oof_auc": experiment.get("full_oof_auc"),
        "seed": experiment.get("seed", config["project"]["seed"]),
        "fold_version": experiment.get("fold_version"),
        "created_at": utc_now(),
        "prediction_summary": {
            "min": float(submission[prediction_column].min()),
            "max": float(submission[prediction_column].max()),
            "mean": float(submission[prediction_column].mean()),
            "rows": len(submission),
        },
        "source_prediction": str(prediction_path),
        "submission_file": str(output),
    }
    metadata_path = output.with_name(output.stem + "_metadata.json")
    write_json(metadata_path, metadata)
    return output, metadata_path
