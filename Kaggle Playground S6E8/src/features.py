from __future__ import annotations

import pandas as pd


FEATURE_VERSIONS = {"raw_v1", "missing_v1"}


def build_features(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    version: str,
) -> pd.DataFrame:
    if version not in FEATURE_VERSIONS:
        raise ValueError(f"Unknown feature version {version!r}; choose {sorted(FEATURE_VERSIONS)}")
    columns = list(feature_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    result = frame.loc[:, columns].copy()
    if version == "raw_v1":
        return result
    reserved = {"missing_count", "has_any_missing"}
    indicator_names = {f"__missing__{column}" for column in columns}
    collisions = (reserved | indicator_names).intersection(result.columns)
    if collisions:
        raise ValueError(f"Missing feature names collide with source columns: {collisions}")
    missing_mask = result.isna()
    result["missing_count"] = missing_mask.sum(axis=1).astype("int16")
    result["has_any_missing"] = missing_mask.any(axis=1).astype("int8")
    for column in columns:
        result[f"__missing__{column}"] = missing_mask[column].astype("int8")
    return result


def align_feature_frames(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(train.columns) != list(test.columns):
        raise ValueError("Train/test generated feature columns or order differ")
    if len(train.columns) != len(set(train.columns)):
        raise ValueError("Generated features contain duplicate names")

