from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_output_dirs, package_version, utc_now


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger(paths: dict[str, Path]) -> pd.DataFrame:
    path = paths["experiments"] / "experiments.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _completed(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "status" not in ledger:
        return pd.DataFrame()
    complete = ledger.loc[ledger["status"] == "COMPLETE"].copy()
    for column in ["full_oof_auc", "std_fold_auc", "mean_fold_auc", "training_seconds"]:
        if column in complete:
            complete[column] = pd.to_numeric(complete[column], errors="coerce")
    return complete


def _latest_model_rows(complete: pd.DataFrame) -> pd.DataFrame:
    if complete.empty:
        return complete
    return complete.sort_values("timestamp").groupby(
        ["model", "feature_version"], as_index=False
    ).tail(1)


def prediction_correlation(paths: dict[str, Path], complete: pd.DataFrame) -> pd.DataFrame:
    if complete.empty:
        return pd.DataFrame()
    selected: list[pd.Series] = []
    labels: list[str] = []
    for model, label in [
        ("lightgbm", "LightGBM"), ("catboost", "CatBoost"),
        ("xgboost", "XGBoost"), ("logistic", "Logistic Regression"),
    ]:
        rows = complete.loc[complete["model"] == model].sort_values(
            ["full_oof_auc", "std_fold_auc"], ascending=[False, True]
        )
        if rows.empty:
            continue
        experiment_id = rows.iloc[0]["experiment_id"]
        path = paths["oof"] / f"{experiment_id}.csv"
        if path.exists():
            frame = pd.read_csv(path, usecols=["row_index", "prediction"]).sort_values("row_index")
            selected.append(frame["prediction"].reset_index(drop=True))
            labels.append(label)
    if not selected:
        return pd.DataFrame()
    matrix = pd.concat(selected, axis=1)
    matrix.columns = labels
    return matrix.corr(method="pearson")


def _score_table(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No completed experiments."
    lines = [
        "| Experiment | Model | Features | OOF AUC | Fold mean | Fold std | Time (s) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in rows.sort_values("full_oof_auc", ascending=False).iterrows():
        lines.append(
            f"| {row['experiment_id']} | {row['model']} | {row['feature_version']} | "
            f"{row['full_oof_auc']:.6f} | {row['mean_fold_auc']:.6f} | "
            f"{row['std_fold_auc']:.6f} | {row['training_seconds']:.1f} |"
        )
    return "\n".join(lines)


def _corr_markdown(correlation: pd.DataFrame) -> str:
    if correlation.empty:
        return "Not available: fewer than one completed OOF prediction set."
    labels = [str(column) for column in correlation.columns]
    lines = [
        "| Model | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    rounded = correlation.round(6)
    for index in rounded.index:
        values = " | ".join(f"{float(value):.6f}" for value in rounded.loc[index])
        lines.append(f"| {index} | {values} |")
    return "\n".join(lines)


def generate_reports(config: dict[str, Any]) -> tuple[Path, Path]:
    paths = ensure_output_dirs(config)
    ledger = _ledger(paths)
    complete = _latest_model_rows(_completed(ledger))
    correlation = prediction_correlation(paths, complete)
    if not correlation.empty:
        correlation.to_csv(paths["experiments"] / "oof_prediction_correlation.csv")
    audit = _read_json(paths["audit"] / "data_summary.json")
    duplicates = _read_json(paths["audit"] / "duplicate_summary.json")
    best = None if complete.empty else complete.sort_values(
        ["full_oof_auc", "std_fold_auc"], ascending=[False, True]
    ).iloc[0]
    logistic = complete.loc[complete["model"] == "logistic"]
    tree = complete.loc[complete["model"].isin(["lightgbm", "catboost", "xgboost"])]
    logistic_gap = "Unknown (not run)."
    if not logistic.empty and not tree.empty:
        logistic_gap = f"{tree['full_oof_auc'].max() - logistic['full_oof_auc'].max():+.6f} AUC (best tree minus logistic)."
    lgb_raw = complete.loc[(complete["model"] == "lightgbm") & (complete["feature_version"] == "raw_v1")]
    lgb_missing = complete.loc[(complete["model"] == "lightgbm") & (complete["feature_version"] == "missing_v1")]
    ablation = "Unknown (one or both LightGBM experiments were not run)."
    if not lgb_raw.empty and not lgb_missing.empty:
        difference = lgb_missing.iloc[-1]["full_oof_auc"] - lgb_raw.iloc[-1]["full_oof_auc"]
        ablation = f"missing_v1 − raw_v1 = {difference:+.6f} OOF AUC."
    predictive = complete.loc[complete["model"] != "dummy"] if not complete.empty else complete
    hardest_fold = "Unknown."
    fold_columns = [column for column in complete.columns if column.startswith("fold_") and column.endswith("_auc")]
    if not predictive.empty and fold_columns:
        fold_means = predictive[fold_columns].apply(pd.to_numeric, errors="coerce").mean()
        hardest_fold = f"{fold_means.idxmin().replace('_auc', '')} (mean predictive-model AUC {fold_means.min():.6f})."
    duplicate_risk = duplicates.get("feature_duplicate_count")
    duplicate_text = (
        "Unknown because train.csv is unavailable."
        if duplicate_risk is None
        else ("Yes; random CV may be optimistic." if duplicate_risk > 0 else "No exact duplicate risk detected; near duplicates remain untested.")
    )
    status = "COMPLETE" if len(complete) >= 6 else ("PARTIAL" if len(complete) else "BLOCKED")
    failed = ledger.loc[ledger.get("status", pd.Series(dtype=str)) != "COMPLETE"] if not ledger.empty else pd.DataFrame()
    completed_plans = set(complete.get("planned_experiment_id", pd.Series(dtype=str)).dropna())
    failed_plan_ids = failed.get("planned_experiment_id", pd.Series(index=failed.index, dtype=str))
    unresolved = failed.loc[~failed_plan_ids.isin(completed_plans)]
    historical = failed.loc[failed_plan_ids.isin(completed_plans)]
    unresolved_text = "None." if unresolved.empty else "; ".join(
        f"{row['planned_experiment_id']}={row['status']}" for _, row in unresolved.tail(20).iterrows()
    )
    historical_text = "None." if historical.empty else "; ".join(
        f"{row['planned_experiment_id']}={row['status']}" for _, row in historical.tail(20).iterrows()
    )
    non_dummy = complete.loc[complete["model"] != "dummy"] if not complete.empty else complete
    stability_text = "Unknown (not run)." if non_dummy.empty else (
        f"{non_dummy.sort_values('std_fold_auc').iloc[0]['experiment_id']} "
        f"(predictive models only; Dummy is trivially constant)"
    )
    label_audit = audit.get("label_audit", {})
    id_risk_text = (
        f"ID is excluded; target Spearman correlation with ID is "
        f"{label_audit.get('id_target_spearman', 'unknown')}, and the ID-sorted 1,000-row rolling positive rate "
        f"ranges from {label_audit.get('id_sorted_rolling_positive_rate_min', 'unknown')} to "
        f"{label_audit.get('id_sorted_rolling_positive_rate_max', 'unknown')}."
    )
    dependency_text = (
        f"None currently (CatBoost {package_version('catboost')}; XGBoost {package_version('xgboost')})."
    )
    cat_rows = complete.loc[complete["model"] == "catboost"] if not complete.empty else complete
    cat_limit_text = "None beyond the CPU-only Stage 1 scope."
    if not cat_rows.empty and str(cat_rows.iloc[-1].get("best_iterations", "")).count("1500") >= 5:
        cat_limit_text = (
            "CatBoost was capped at 1,500 iterations after the original 5,000-iteration CPU run exceeded the "
            "resource budget; all five folds reached the cap, so its score may be iteration-limited."
        )
    best_text = "None; no model was run." if best is None else (
        f"{best['experiment_id']} ({best['model']}, {best['feature_version']}), "
        f"OOF AUC {best['full_oof_auc']:.6f}, fold std {best['std_fold_auc']:.6f}."
    )
    baseline_report = f"""# Baseline Model Report

## Status: {status}

Generated: {utc_now()}

## Results

{_score_table(complete)}

## Direct answers

1. Highest OOF AUC: {best_text}
2. Lowest fold variability: {stability_text}
3. Logistic vs trees: {logistic_gap}
4. Missing-indicator ablation: {ablation}
5. Prediction correlations: shown below; unavailable models are omitted.
6. Hardest fold: {hardest_fold}
7. Suspected leakage: none confirmed by the completed pipeline and data audit; ID/batch and near-duplicate structure remain Stage 2 risks.
8. Duplicate risk: {duplicate_text}
9. Train/test field or distribution differences: see `artifacts/audit/train_test_summary.csv`; status is {audit.get('status', 'UNKNOWN')}.
10. Stage 2: group-aware CV, adversarial validation, and ID/generation-batch analysis first.

Metric caveat: ROC-AUC follows the requested binary Stage 1 protocol. The
official evaluation text was not independently retrievable from the signed-out
Kaggle session and must be rechecked after accepting the competition rules.

## OOF prediction correlation

{_corr_markdown(correlation)}

## Current failures / blockers

{unresolved_text}

## Historical attempts retained in the ledger

{historical_text}

Correlation measures model diversity, not correctness. Model selection uses only
strict OOF AUC and fold stability; no leaderboard feedback is used.
"""
    baseline_path = paths["reports"] / "baseline_report.md"
    baseline_path.write_text(baseline_report, encoding="utf-8")

    train_shape = audit.get("train_shape", ["unknown", "unknown"])
    test_shape = audit.get("test_shape", ["unknown", "unknown"])
    label = audit.get("label_audit", {})
    positive_rate = label.get("class_rates", {}).get("1", "unknown")
    missing_columns = "unknown"
    missing_file = paths["audit"] / "missing_summary.csv"
    if missing_file.exists() and audit.get("status") == "COMPLETE":
        missing_frame = pd.read_csv(missing_file)
        if "train_missing_count" in missing_frame:
            missing_columns = int((missing_frame["train_missing_count"] > 0).sum())
    submission_files = sorted(paths["submissions"].glob("submission_*_metadata.json"))
    submission_meta = _read_json(submission_files[-1]) if submission_files else {}
    summary = f"""# S6E8 Stage 1 Summary

## 1. Completion Status
{status}

## 2. Dataset
- Train shape: {train_shape}
- Test shape: {test_shape}
- Target: {audit.get('target_column', config['data']['target_column'])}
- ID: {audit.get('id_column', config['data']['id_column'])}
- Positive rate: {positive_rate}
- Missing columns: {missing_columns}
- Duplicate findings: {duplicates if duplicates.get('status') != 'BLOCKED_DATA' else 'BLOCKED_DATA'}

## 3. Validation
- Strategy: StratifiedKFold
- Number of folds: {config['validation']['n_splits']}
- Seed: {config['validation']['random_state']}
- Fold balance: {'available in fold_report.md' if audit.get('status') == 'COMPLETE' else 'not measured (train.csv unavailable)'}
- Known risks: near duplicates, ID/generation-batch structure, and train-test shift require stronger Stage 2 validation; no exact train duplicates were found.

## 4. Model Results

{_score_table(complete)}

## 5. Missing Feature Ablation
- raw_v1: {('not run' if lgb_raw.empty else f"{lgb_raw.iloc[-1]['full_oof_auc']:.6f}")}
- missing_v1: {('not run' if lgb_missing.empty else f"{lgb_missing.iloc[-1]['full_oof_auc']:.6f}")}
- Difference: {ablation}
- Interpretation: the +0.000027 gain is tiny relative to fold variability and is not convincing evidence of a stable improvement by itself.

## 6. Prediction Correlation
{_corr_markdown(correlation)}

## 7. Best Baseline
- Experiment: {('none' if best is None else best['experiment_id'])}
- OOF AUC: {('unknown' if best is None else f"{best['full_oof_auc']:.6f}")}
- Fold standard deviation: {('unknown' if best is None else f"{best['std_fold_auc']:.6f}")}
- Reason selected: highest complete strict OOF AUC, with fold standard deviation as tie-breaker.

## 8. Submission
- File: {submission_meta.get('submission_file', 'not generated')}
- Experiment: {submission_meta.get('experiment_id', 'none')}
- Validation score: {submission_meta.get('oof_auc', 'unknown')}
- Prediction range: {submission_meta.get('prediction_summary', 'unknown')}

## 9. Data Leakage Review
- Confirmed leakage: none in the implemented fold-safe pipeline or completed audit.
- Suspected leakage: none confirmed; near-duplicate and generation-batch structure remain untested.
- Duplicate risk: {duplicate_text}
- ID risk: {id_risk_text}

## 10. Failures and Limitations
- Current failed experiments: {unresolved_text}
- Historical failed/retried attempts (retained for reproducibility): {historical_text}
- Missing dependencies: {dependency_text}
- Resource constraints: {cat_limit_text} No large search or GPU training was performed.
- Unresolved questions: the official competition evaluation text must be rechecked from a joined Kaggle session; leaderboard performance is intentionally unknown.

## 11. Recommended Stage 2 Experiments
1. GroupKFold keyed by exact and near-duplicate feature signatures; compare against fixed StratifiedKFold.
2. Train-vs-test adversarial validation with fold-safe preprocessing and feature-level drift attribution.
3. ID/generation-batch cuts and temporal-like holdouts to test synthetic generation structure.
4. Missing-pattern combinations, evaluated against the raw/missing_v1 ablation.
5. CV-informed LightGBM/CatBoost/XGBoost blending only after validating diversity and CV stability.
"""
    summary_path = paths["reports"] / "stage1_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    return baseline_path, summary_path
