from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import (
    DataBundle,
    DataUnavailableError,
    classify_feature_columns,
    configured_data_paths,
    dataframe_memory_mb,
    load_data,
)
from src.utils import ensure_output_dirs, environment_markdown, utc_now, write_json


COLUMN_SUMMARY_COLUMNS = [
    "column", "dtype", "non_null_count", "missing_count", "missing_rate",
    "unique_count", "unique_rate", "min", "q01", "q05", "q25", "mean",
    "median", "q75", "q95", "q99", "max", "std", "skew", "top_value",
    "top_frequency", "infinite_count", "negative_count", "iqr_outlier_count",
    "suspected_constant", "suspected_id", "suspected_categorical",
]


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return value
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value


def _profile_column(series: pd.Series) -> dict[str, Any]:
    n = len(series)
    non_null = int(series.notna().sum())
    missing = n - non_null
    unique = int(series.nunique(dropna=True))
    counts = series.value_counts(dropna=True)
    row: dict[str, Any] = {
        "column": series.name,
        "dtype": str(series.dtype),
        "non_null_count": non_null,
        "missing_count": missing,
        "missing_rate": missing / n if n else np.nan,
        "unique_count": unique,
        "unique_rate": unique / non_null if non_null else np.nan,
        "top_value": _safe_scalar(counts.index[0]) if len(counts) else None,
        "top_frequency": int(counts.iloc[0]) if len(counts) else 0,
        "suspected_constant": unique <= 1,
        "suspected_id": bool(non_null and unique / non_null >= 0.98),
        "suspected_categorical": bool(
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or unique <= 50
        ),
    }
    numeric = pd.to_numeric(series, errors="coerce") if pd.api.types.is_numeric_dtype(series) else None
    stats = ["min", "q01", "q05", "q25", "mean", "median", "q75", "q95", "q99", "max", "std", "skew"]
    if numeric is not None:
        finite = numeric[np.isfinite(numeric)]
        quantiles = finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(finite) else pd.Series(dtype=float)
        row.update({
            "min": finite.min() if len(finite) else None,
            "q01": quantiles.get(0.01),
            "q05": quantiles.get(0.05),
            "q25": quantiles.get(0.25),
            "mean": finite.mean() if len(finite) else None,
            "median": quantiles.get(0.5),
            "q75": quantiles.get(0.75),
            "q95": quantiles.get(0.95),
            "q99": quantiles.get(0.99),
            "max": finite.max() if len(finite) else None,
            "std": finite.std() if len(finite) else None,
            "skew": finite.skew() if len(finite) else None,
            "infinite_count": int(np.isinf(numeric).sum()),
            "negative_count": int((finite < 0).sum()),
        })
        q25, q75 = quantiles.get(0.25), quantiles.get(0.75)
        if pd.notna(q25) and pd.notna(q75):
            iqr = q75 - q25
            row["iqr_outlier_count"] = int(
                ((finite < q25 - 3 * iqr) | (finite > q75 + 3 * iqr)).sum()
            )
        else:
            row["iqr_outlier_count"] = 0
    else:
        row.update({key: None for key in stats})
        row.update({"infinite_count": 0, "negative_count": 0, "iqr_outlier_count": 0})
    return row


def _hash_rows(frame: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(frame, index=False, categorize=True)


def _duplicate_audit(bundle: DataBundle) -> dict[str, Any]:
    train = bundle.train
    target = bundle.schema.target_column
    identifier = bundle.schema.id_column
    features = list(bundle.schema.feature_columns)
    exact_duplicate_count = int(train.duplicated(keep="first").sum())
    feature_hash = _hash_rows(train[features])
    feature_counts = feature_hash.value_counts()
    duplicated_signatures = feature_counts[feature_counts > 1]
    feature_duplicate_count = int(train[features].duplicated(keep="first").sum())
    duplicate_group_count = int(len(duplicated_signatures))
    conflict = pd.DataFrame({"signature": feature_hash, "target": train[target]}).groupby(
        "signature", sort=False
    )["target"].nunique()
    conflicting = int((conflict > 1).sum())
    common_with_id = [identifier, *features]
    train_with_id_hash = _hash_rows(train[common_with_id])
    test_with_id_hash = _hash_rows(bundle.test[common_with_id])
    overlap_signatures = np.intersect1d(train_with_id_hash.unique(), test_with_id_hash.unique())
    train_no_id_hash = feature_hash
    test_no_id_hash = _hash_rows(bundle.test[features])
    overlap_no_id = np.intersect1d(train_no_id_hash.unique(), test_no_id_hash.unique())
    return {
        "exact_duplicate_count": exact_duplicate_count,
        "feature_duplicate_count": feature_duplicate_count,
        "duplicate_group_count": duplicate_group_count,
        "conflicting_label_group_count": conflicting,
        "train_test_exact_overlap_count": int(len(overlap_signatures)),
        "train_test_overlap_without_id_count": int(len(overlap_no_id)),
        "definition": "Duplicate counts are rows after the first; overlap counts are unique feature signatures.",
    }


def _missing_summary(bundle: DataBundle) -> tuple[pd.DataFrame, dict[str, Any]]:
    train, test = bundle.train, bundle.test
    target = bundle.schema.target_column
    rows: list[dict[str, Any]] = []
    for column in bundle.schema.feature_columns:
        row = {
            "column": column,
            "train_missing_count": int(train[column].isna().sum()),
            "train_missing_rate": float(train[column].isna().mean()),
            "test_missing_count": int(test[column].isna().sum()),
            "test_missing_rate": float(test[column].isna().mean()),
            "missing_rate_difference": float(test[column].isna().mean() - train[column].isna().mean()),
        }
        for label in (0, 1):
            mask = train[target] == label
            row[f"target_{label}_missing_rate"] = float(train.loc[mask, column].isna().mean())
        rows.append(row)
    missing_frame = pd.DataFrame(rows)
    train_counts = train[list(bundle.schema.feature_columns)].isna().sum(axis=1)
    feature_array = np.asarray(bundle.schema.feature_columns)
    missing_matrix = train[list(bundle.schema.feature_columns)].isna().to_numpy(dtype=np.uint8)
    packed = np.packbits(missing_matrix, axis=1)
    unique_patterns, pattern_counts = np.unique(packed, axis=0, return_counts=True)
    top_indices = np.argsort(pattern_counts)[::-1][:20]
    top_patterns = []
    for index in top_indices:
        unpacked = np.unpackbits(unique_patterns[index])[: len(feature_array)].astype(bool)
        pattern = "|".join(feature_array[unpacked].tolist()) or "<none>"
        count = int(pattern_counts[index])
        top_patterns.append({"pattern": pattern, "count": count, "rate": count / len(train)})
    missing_cols = [column for column in bundle.schema.feature_columns if train[column].isna().any()]
    co_missing: list[dict[str, Any]] = []
    if missing_cols:
        mask = train[missing_cols].isna().astype("int8")
        wide_mask = mask.astype("int64")
        co = wide_mask.T.dot(wide_mask)
        for i, left in enumerate(missing_cols):
            for right in missing_cols[i + 1 :]:
                count = int(co.loc[left, right])
                if count:
                    co_missing.append({"left": left, "right": right, "count": count})
        co_missing.sort(key=lambda item: item["count"], reverse=True)
    relation = float(train_counts.corr(train[target])) if train_counts.nunique() > 1 else None
    details = {
        "train_row_missing_count_summary": train_counts.describe().to_dict(),
        "missing_count_target_correlation": relation,
        "top_missing_patterns": top_patterns,
        "top_co_missing_pairs": co_missing[:30],
    }
    return missing_frame, details


def _train_test_summary(bundle: DataBundle) -> pd.DataFrame:
    train, test = bundle.train, bundle.test
    rows: list[dict[str, Any]] = []
    for column in bundle.schema.feature_columns:
        tr, te = train[column], test[column]
        row: dict[str, Any] = {
            "column": column,
            "train_dtype": str(tr.dtype),
            "test_dtype": str(te.dtype),
            "train_unique": int(tr.nunique(dropna=True)),
            "test_unique": int(te.nunique(dropna=True)),
            "train_missing_rate": float(tr.isna().mean()),
            "test_missing_rate": float(te.isna().mean()),
        }
        treat_as_categorical = (
            not pd.api.types.is_numeric_dtype(tr)
            or pd.api.types.is_bool_dtype(tr)
            or tr.nunique(dropna=True) <= 20
        )
        if not treat_as_categorical and pd.api.types.is_numeric_dtype(te):
            row.update({
                "train_min": tr.min(), "test_min": te.min(),
                "train_max": tr.max(), "test_max": te.max(),
                "train_mean": tr.mean(), "test_mean": te.mean(),
                "test_below_train_min": int((te < tr.min()).sum()) if tr.notna().any() else 0,
                "test_above_train_max": int((te > tr.max()).sum()) if tr.notna().any() else 0,
                "train_only_category_count": None, "test_only_category_count": None,
            })
        else:
            tr_values = set(tr.dropna().astype(str).unique())
            te_values = set(te.dropna().astype(str).unique())
            row.update({
                "train_min": None, "test_min": None, "train_max": None, "test_max": None,
                "train_mean": None, "test_mean": None,
                "test_below_train_min": None, "test_above_train_max": None,
                "train_only_category_count": len(tr_values - te_values),
                "test_only_category_count": len(te_values - tr_values),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _label_audit(bundle: DataBundle) -> dict[str, Any]:
    train = bundle.train
    target, identifier = bundle.schema.target_column, bundle.schema.id_column
    counts = train[target].value_counts().sort_index()
    id_numeric = pd.to_numeric(train[identifier], errors="coerce")
    id_target_corr = (
        float(id_numeric.rank().corr(train[target].rank()))
        if id_numeric.notna().all() else None
    )
    order = np.argsort(id_numeric.to_numpy()) if id_numeric.notna().all() else np.arange(len(train))
    window = min(1000, max(50, len(train) // 100))
    rolling = train[target].iloc[order].rolling(window, min_periods=window).mean()
    return {
        "target_dtype": str(train[target].dtype),
        "class_count": int(counts.size),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "class_rates": {str(key): float(value / len(train)) for key, value in counts.items()},
        "missing_target_count": int(train[target].isna().sum()),
        "id_target_spearman": id_target_corr,
        "id_sorted_rolling_window": window,
        "id_sorted_rolling_positive_rate_min": float(rolling.min()) if rolling.notna().any() else None,
        "id_sorted_rolling_positive_rate_max": float(rolling.max()) if rolling.notna().any() else None,
    }


def write_blocked_audit(config: dict[str, Any], reason: str) -> None:
    paths = ensure_output_dirs(config)
    data_paths = configured_data_paths(config)
    (paths["reports"] / "environment_report.md").write_text(
        environment_markdown(data_paths), encoding="utf-8"
    )
    payload = {
        "status": "BLOCKED_DATA",
        "generated_at": utc_now(),
        "reason": reason,
        "available_files": {name: path.exists() for name, path in data_paths.items()},
    }
    sample_path = data_paths["sample_submission.csv"]
    if sample_path.exists():
        sample = pd.read_csv(sample_path, low_memory=False)
        payload["sample_submission_shape"] = list(sample.shape)
        payload["sample_submission_columns"] = sample.columns.tolist()
    write_json(paths["audit"] / "data_summary.json", payload)
    pd.DataFrame(columns=COLUMN_SUMMARY_COLUMNS).to_csv(
        paths["audit"] / "column_summary.csv", index=False
    )
    pd.DataFrame(columns=[
        "column", "train_missing_count", "train_missing_rate", "test_missing_count",
        "test_missing_rate", "missing_rate_difference", "target_0_missing_rate",
        "target_1_missing_rate",
    ]).to_csv(paths["audit"] / "missing_summary.csv", index=False)
    write_json(paths["audit"] / "duplicate_summary.json", payload)
    pd.DataFrame(columns=["column", "status"]).to_csv(
        paths["audit"] / "train_test_summary.csv", index=False
    )
    (paths["reports"] / "data_audit.md").write_text(
        f"# Data Audit\n\n## Status: BLOCKED_DATA\n\n{reason}\n\n"
        "No row-level audit findings were fabricated. Add `train.csv` and `test.csv` "
        "as described in `data/README.md`, then rerun `python scripts/run_audit.py`.\n",
        encoding="utf-8",
    )


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_output_dirs(config)
    data_paths = configured_data_paths(config)
    (paths["reports"] / "environment_report.md").write_text(
        environment_markdown(data_paths), encoding="utf-8"
    )
    try:
        bundle = load_data(config)
    except DataUnavailableError as exc:
        write_blocked_audit(config, str(exc))
        raise

    train, test = bundle.train, bundle.test
    numerical, categorical = classify_feature_columns(
        train[list(bundle.schema.feature_columns)],
        config["data"].get("low_cardinality_numeric_threshold", 20),
    )
    column_summary = pd.DataFrame([_profile_column(train[column]) for column in train.columns])
    column_summary = column_summary.reindex(columns=COLUMN_SUMMARY_COLUMNS)
    column_summary.to_csv(paths["audit"] / "column_summary.csv", index=False)
    missing_summary, missing_details = _missing_summary(bundle)
    missing_summary.to_csv(paths["audit"] / "missing_summary.csv", index=False)
    duplicate_summary = _duplicate_audit(bundle)
    write_json(paths["audit"] / "duplicate_summary.json", duplicate_summary)
    train_test = _train_test_summary(bundle)
    train_test.to_csv(paths["audit"] / "train_test_summary.csv", index=False)
    label = _label_audit(bundle)
    summary = {
        "status": "COMPLETE",
        "generated_at": utc_now(),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "sample_submission_shape": list(bundle.sample_submission.shape),
        "target_column": bundle.schema.target_column,
        "id_column": bundle.schema.id_column,
        "prediction_columns": list(bundle.schema.prediction_columns),
        "feature_count": len(bundle.schema.feature_columns),
        "numerical_features": numerical,
        "categorical_features": categorical,
        "train_memory_mb": dataframe_memory_mb(train),
        "test_memory_mb": dataframe_memory_mb(test),
        "label_audit": label,
        "duplicate_audit": duplicate_summary,
        "missing_audit": missing_details,
    }
    write_json(paths["audit"] / "data_summary.json", summary)
    duplicate_risk = duplicate_summary["feature_duplicate_count"] > 0
    missing_columns = int((missing_summary["train_missing_count"] > 0).sum())
    distribution_flags = int(
        (train_test["test_missing_rate"] - train_test["train_missing_rate"]).abs().gt(0.05).sum()
    )
    report = f"""# Data Audit

## Status: COMPLETE

Generated: {utc_now()}

## Dataset and grain

- Train: {train.shape[0]:,} rows × {train.shape[1]:,} columns
- Test: {test.shape[0]:,} rows × {test.shape[1]:,} columns
- Unit of analysis: one competition row, keyed by `{bundle.schema.id_column}`
- Target: `{bundle.schema.target_column}` (binary, positive rate {label['class_rates'].get('1', float('nan')):.6f})
- Features: {len(bundle.schema.feature_columns)} ({len(numerical)} numerical, {len(categorical)} categorical by heuristic)
- Missing columns in train: {missing_columns}

## Checks performed

Completeness, dtype/cardinality, robust quantiles and 3×IQR extremes, infinities,
negative values, target balance, ID/target rank relation, ID-sorted rolling target
rate, exact and feature-only duplicates, conflicting labels, train-test feature
overlap, row missing patterns, co-missing pairs, per-class missingness, range and
category support differences.

## Duplicate findings

- Exact duplicate rows after first: {duplicate_summary['exact_duplicate_count']:,}
- Feature duplicate rows after first: {duplicate_summary['feature_duplicate_count']:,}
- Duplicate feature groups: {duplicate_summary['duplicate_group_count']:,}
- Conflicting-label duplicate groups: {duplicate_summary['conflicting_label_group_count']:,}
- Train/test exact overlaps including ID: {duplicate_summary['train_test_exact_overlap_count']:,}
- Train/test overlapping feature signatures after removing ID: {duplicate_summary['train_test_overlap_without_id_count']:,}

{'**Risk:** Random StratifiedKFold may be optimistic because identical feature signatures can cross folds. Keep the standard five-fold baseline, but test GroupKFold in Stage 2.' if duplicate_risk else 'No exact feature duplicates were detected; near-duplicate risk remains untested in Stage 1.'}

## Missingness and train-test differences

- Train columns with missing values: {missing_columns}
- Correlation between row missing count and target: {missing_details['missing_count_target_correlation']}
- Fields with absolute train-test missing-rate gap > 5 pp: {distribution_flags}
- Detailed evidence: `artifacts/audit/missing_summary.csv` and `train_test_summary.csv`

## Leakage review

- Target is excluded from both feature versions by construction.
- Missing indicators use only the current row.
- No target encoding is used.
- Learned imputers, scalers, one-hot/ordinal mappings are fitted inside each fold.
- ID is excluded from model features.

## Severity and next tests

Any conflicting-label duplicate groups or strong ID/target structure are high-risk
for validation. Range/category support differences and missing-rate gaps are
medium-risk distribution warnings, not reasons to delete rows. Stage 2 should add
group-aware CV, near-duplicate hashing, adversarial validation, and ID/batch cuts.
"""
    (paths["reports"] / "data_audit.md").write_text(report, encoding="utf-8")
    return summary
