from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.metrics import validate_probabilities


def read_prediction_file(path: str | Path, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"row_index", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Prediction file missing columns: {required - set(frame.columns)}")
    if len(frame) != expected_rows:
        raise ValueError(f"Prediction rows {len(frame)} != expected {expected_rows}")
    if frame["row_index"].duplicated().any() or not frame["row_index"].equals(
        pd.Series(range(expected_rows), name="row_index")
    ):
        raise ValueError("Prediction row_index must be unique and in original order")
    validate_probabilities(frame["prediction"].to_numpy(), expected_rows, "test prediction")
    return frame

