from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def validate_probabilities(values: np.ndarray, expected_length: int, label: str) -> None:
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_length,):
        raise ValueError(f"{label} shape {array.shape} != ({expected_length},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity")
    if ((array < 0) | (array > 1)).any():
        raise ValueError(f"{label} contains values outside [0, 1]")


def auc(y_true: np.ndarray, prediction: np.ndarray) -> float:
    validate_probabilities(np.asarray(prediction), len(y_true), "prediction")
    return float(roc_auc_score(y_true, prediction))

