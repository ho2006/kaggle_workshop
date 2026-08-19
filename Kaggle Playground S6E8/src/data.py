from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, project_path


class DataUnavailableError(FileNotFoundError):
    """Raised when required competition data is not available."""


@dataclass(frozen=True)
class CompetitionSchema:
    target_column: str
    id_column: str
    prediction_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]


@dataclass
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    schema: CompetitionSchema
    paths: dict[str, Path]


def resolve_data_path(configured: str | Path, filename: str) -> Path:
    configured_path = project_path(configured)
    candidates = [
        configured_path,
        PROJECT_ROOT / "data" / "raw" / filename,
        PROJECT_ROOT / filename,
    ]
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return candidate
    return configured_path


def configured_data_paths(config: dict[str, Any]) -> dict[str, Path]:
    data = config["data"]
    return {
        "train.csv": resolve_data_path(data["train_path"], "train.csv"),
        "test.csv": resolve_data_path(data["test_path"], "test.csv"),
        "sample_submission.csv": resolve_data_path(
            data["sample_submission_path"], "sample_submission.csv"
        ),
    }


def require_data_files(config: dict[str, Any]) -> dict[str, Path]:
    paths = configured_data_paths(config)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{name} -> {paths[name]}" for name in missing)
        raise DataUnavailableError(
            f"Missing competition data: {details}. See data/README.md."
        )
    return paths


def _header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def infer_competition_schema(
    train_columns: list[str],
    test_columns: list[str],
    sample_columns: list[str],
    configured_target: str | None = None,
    configured_id: str | None = None,
) -> CompetitionSchema:
    train_only = [column for column in train_columns if column not in test_columns]
    test_only = [column for column in test_columns if column not in train_columns]
    if test_only:
        raise ValueError(f"Unexpected test-only columns: {test_only}")
    if configured_target:
        if configured_target not in train_only:
            raise ValueError(
                f"Configured target {configured_target!r} is not the sole train-only field; "
                f"train-only fields are {train_only}."
            )
        target = configured_target
    elif len(train_only) == 1:
        target = train_only[0]
    else:
        raise ValueError(f"Cannot uniquely infer target from train-only columns: {train_only}")

    common_sample = [column for column in sample_columns if column in test_columns]
    if configured_id:
        if configured_id not in common_sample:
            raise ValueError(f"Configured ID {configured_id!r} is not shared by test and sample")
        id_column = configured_id
    elif len(common_sample) == 1:
        id_column = common_sample[0]
    else:
        raise ValueError(f"Cannot uniquely infer ID; shared test/sample columns: {common_sample}")

    prediction_columns = tuple(column for column in sample_columns if column != id_column)
    if len(prediction_columns) != 1 or prediction_columns[0] != target:
        raise ValueError(
            "Binary submission must contain the inferred target as its only prediction "
            f"column; found {prediction_columns}, target={target!r}."
        )
    expected_test_order = [column for column in train_columns if column != target]
    if expected_test_order != test_columns:
        raise ValueError("Train/test feature column order differs after removing target")
    features = tuple(column for column in test_columns if column != id_column)
    return CompetitionSchema(target, id_column, prediction_columns, features)


def load_data(config: dict[str, Any]) -> DataBundle:
    paths = require_data_files(config)
    train_path, test_path, sample_path = (
        paths["train.csv"], paths["test.csv"], paths["sample_submission.csv"]
    )
    schema = infer_competition_schema(
        _header(train_path),
        _header(test_path),
        _header(sample_path),
        config["data"].get("target_column"),
        config["data"].get("id_column"),
    )
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    sample = pd.read_csv(sample_path, low_memory=False)
    if len(sample) != len(test):
        raise ValueError(f"Sample rows {len(sample)} != test rows {len(test)}")
    if not sample[schema.id_column].equals(test[schema.id_column]):
        raise ValueError("Sample submission IDs are not aligned with test IDs")
    if train[schema.target_column].isna().any():
        raise ValueError("Target contains missing values")
    target_values = pd.to_numeric(train[schema.target_column], errors="raise")
    unique = set(target_values.unique().tolist())
    if unique != {0, 1}:
        raise ValueError(f"Expected binary target values {{0, 1}}, found {sorted(unique)}")
    train[schema.target_column] = target_values.astype("int8")
    return DataBundle(train, test, sample, schema, paths)


def classify_feature_columns(
    frame: pd.DataFrame, low_cardinality_numeric_threshold: int = 20
) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numerical: list[str] = []
    for column in frame.columns:
        series = frame[column]
        unique = series.nunique(dropna=True)
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or unique <= low_cardinality_numeric_threshold
        ):
            categorical.append(column)
        else:
            numerical.append(column)
    return numerical, categorical


def dataframe_memory_mb(frame: pd.DataFrame) -> float:
    return float(frame.memory_usage(index=True, deep=True).sum() / (1024 ** 2))

