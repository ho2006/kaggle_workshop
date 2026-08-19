from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import load_config, project_path
from src.data import (
    DataBundle,
    DataUnavailableError,
    classify_feature_columns,
    load_data,
)
from src.features import align_feature_frames, build_features
from src.folds import create_and_save_folds, validate_fold_frame
from src.metrics import auc, validate_probabilities
from src.utils import (
    append_experiment_row,
    ensure_output_dirs,
    git_info,
    unique_experiment_id,
    utc_now,
)


EXPERIMENT_PLAN = [
    ("E000_dummy_raw_v1_seed42", "dummy", "raw_v1", "configs/baseline.yaml"),
    ("E001_logistic_raw_v1_seed42", "logistic", "raw_v1", "configs/baseline.yaml"),
    ("E002_lgb_raw_v1_seed42", "lightgbm", "raw_v1", "configs/lightgbm.yaml"),
    ("E003_lgb_missing_v1_seed42", "lightgbm", "missing_v1", "configs/lightgbm.yaml"),
    ("E004_cat_raw_v1_seed42", "catboost", "raw_v1", "configs/catboost.yaml"),
    ("E005_xgb_raw_v1_seed42", "xgboost", "raw_v1", "configs/xgboost.yaml"),
]


@dataclass
class FoldResult:
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    model: Any
    best_iteration: int | None
    feature_importance: pd.DataFrame | None
    inference_seconds: float


def _column_types(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    return classify_feature_columns(
        frame, config["data"].get("low_cardinality_numeric_threshold", 20)
    )


def _categorical_string_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed numeric/string categoricals without learning from validation rows."""
    return frame.astype("string").fillna("__MISSING__").astype(str)


def _logistic_pipeline(
    numeric_columns: list[str], categorical_columns: list[str], seed: int, threads: int
) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        (
            "to_string",
            FunctionTransformer(
                _categorical_string_frame, validate=False, feature_names_out="one-to-one"
            ),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    transformer = ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)],
        remainder="drop", sparse_threshold=1.0,
    )
    return Pipeline([
        ("preprocess", transformer),
        ("model", LogisticRegression(
            max_iter=1000, random_state=seed, solver="lbfgs"
        )),
    ])


def _lightgbm_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    validation = validation.copy()
    test = test.copy()
    for column in categorical_columns:
        categories = pd.Index(train[column].dropna().unique())
        train[column] = pd.Categorical(train[column], categories=categories)
        validation_values = validation[column].where(validation[column].isin(categories))
        test_values = test[column].where(test[column].isin(categories))
        validation[column] = pd.Categorical(validation_values, categories=categories)
        test[column] = pd.Categorical(test_values, categories=categories)
    return train, validation, test


def _catboost_frame(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in categorical_columns:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def _ordinal_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OrdinalEncoder | None]:
    train_encoded = train.copy()
    validation_encoded = validation.copy()
    test_encoded = test.copy()
    encoder: OrdinalEncoder | None = None
    if categorical_columns:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-2, dtype=np.float32,
        )
        train_cat = train[categorical_columns].astype("string").fillna("__MISSING__")
        valid_cat = validation[categorical_columns].astype("string").fillna("__MISSING__")
        test_cat = test[categorical_columns].astype("string").fillna("__MISSING__")
        train_encoded[categorical_columns] = encoder.fit_transform(train_cat)
        validation_encoded[categorical_columns] = encoder.transform(valid_cat)
        test_encoded[categorical_columns] = encoder.transform(test_cat)
    return (
        train_encoded.to_numpy(dtype=np.float32),
        validation_encoded.to_numpy(dtype=np.float32),
        test_encoded.to_numpy(dtype=np.float32),
        encoder,
    )


def _importance(model: Any, columns: list[str], fold: int) -> pd.DataFrame | None:
    values = getattr(model, "feature_importances_", None)
    if values is None or len(values) != len(columns):
        return None
    return pd.DataFrame({"feature": columns, "importance": values, "fold": fold})


def _fit_fold(
    model_name: str,
    config: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    fold: int,
) -> FoldResult:
    seed = config["project"]["seed"]
    threads = config["training"]["num_threads"]
    early_stopping = config["training"]["early_stopping_rounds"]
    numeric, categorical = _column_types(x_train, config)
    if model_name == "dummy":
        model = DummyClassifier(strategy="prior", random_state=seed)
        model.fit(np.zeros((len(x_train), 1)), y_train)
        infer_start = time.perf_counter()
        valid_pred = model.predict_proba(np.zeros((len(x_valid), 1)))[:, 1]
        test_pred = model.predict_proba(np.zeros((len(x_test), 1)))[:, 1]
        return FoldResult(valid_pred, test_pred, model, None, None, time.perf_counter() - infer_start)

    if model_name == "logistic":
        model = _logistic_pipeline(numeric, categorical, seed, threads)
        model.fit(x_train, y_train)
        infer_start = time.perf_counter()
        valid_pred = model.predict_proba(x_valid)[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]
        return FoldResult(valid_pred, test_pred, model, None, None, time.perf_counter() - infer_start)

    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("LightGBM is not installed; install requirements.txt") from exc
        tr, va, te = _lightgbm_frames(x_train, x_valid, x_test, categorical)
        model = lgb.LGBMClassifier(**config["model"]["params"])
        model.fit(
            tr, y_train,
            eval_set=[(va, y_valid)],
            eval_metric="auc",
            categorical_feature=categorical,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        infer_start = time.perf_counter()
        valid_pred = model.predict_proba(va, num_iteration=model.best_iteration_)[:, 1]
        test_pred = model.predict_proba(te, num_iteration=model.best_iteration_)[:, 1]
        return FoldResult(
            valid_pred, test_pred, model, int(model.best_iteration_),
            _importance(model, list(x_train.columns), fold),
            time.perf_counter() - infer_start,
        )

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost is not installed; install requirements.txt") from exc
        tr = _catboost_frame(x_train, categorical)
        va = _catboost_frame(x_valid, categorical)
        te = _catboost_frame(x_test, categorical)
        params = dict(config["model"]["params"])
        model = CatBoostClassifier(**params)
        model.fit(
            tr, y_train, eval_set=(va, y_valid), cat_features=categorical,
            early_stopping_rounds=early_stopping, use_best_model=True,
        )
        infer_start = time.perf_counter()
        valid_pred = model.predict_proba(va)[:, 1]
        test_pred = model.predict_proba(te)[:, 1]
        best = int(model.get_best_iteration() + 1)
        return FoldResult(
            valid_pred, test_pred, model, best,
            _importance(model, list(x_train.columns), fold),
            time.perf_counter() - infer_start,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("XGBoost is not installed; install requirements.txt") from exc
        tr, va, te, encoder = _ordinal_frames(x_train, x_valid, x_test, categorical)
        params = dict(config["model"]["params"])
        params["early_stopping_rounds"] = early_stopping
        model = XGBClassifier(**params)
        model.fit(tr, y_train, eval_set=[(va, y_valid)], verbose=False)
        infer_start = time.perf_counter()
        valid_pred = model.predict_proba(va)[:, 1]
        test_pred = model.predict_proba(te)[:, 1]
        model._stage1_ordinal_encoder = encoder
        best = int(getattr(model, "best_iteration", params["n_estimators"] - 1) + 1)
        return FoldResult(
            valid_pred, test_pred, model, best,
            _importance(model, list(x_train.columns), fold),
            time.perf_counter() - infer_start,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _save_model(model: Any, path: Path, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        model.save_model(str(path.with_suffix(".cbm")))
    elif model_name == "xgboost":
        encoder = getattr(model, "_stage1_ordinal_encoder", None)
        if hasattr(model, "_stage1_ordinal_encoder"):
            delattr(model, "_stage1_ordinal_encoder")
        model.save_model(str(path.with_suffix(".json")))
        if encoder is not None:
            joblib.dump(encoder, path.with_name(path.name + "_encoder.joblib"))
    else:
        joblib.dump(model, path.with_suffix(".joblib"))


def _fit_full_model(
    model_name: str,
    config: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    best_iterations: list[int],
) -> tuple[Any, Any | None]:
    numeric, categorical = _column_types(features, config)
    seed = config["project"]["seed"]
    threads = config["training"]["num_threads"]
    rounds = max(1, int(np.median(best_iterations))) if best_iterations else None
    if model_name == "dummy":
        model = DummyClassifier(strategy="prior", random_state=seed)
        model.fit(np.zeros((len(features), 1)), target)
        return model, None
    if model_name == "logistic":
        model = _logistic_pipeline(numeric, categorical, seed, threads)
        model.fit(features, target)
        return model, None
    if model_name == "lightgbm":
        import lightgbm as lgb

        tr, _, _ = _lightgbm_frames(features, features.iloc[:0], features.iloc[:0], categorical)
        params = dict(config["model"]["params"])
        if rounds:
            params["n_estimators"] = rounds
        model = lgb.LGBMClassifier(**params)
        model.fit(tr, target, categorical_feature=categorical)
        return model, None
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        params = dict(config["model"]["params"])
        if rounds:
            params["iterations"] = rounds
        params["verbose"] = False
        model = CatBoostClassifier(**params)
        model.fit(_catboost_frame(features, categorical), target, cat_features=categorical)
        return model, None
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        encoded = features.copy()
        encoder = None
        if categorical:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=-2, dtype=np.float32,
            )
            cat_values = features[categorical].astype("string").fillna("__MISSING__")
            encoded[categorical] = encoder.fit_transform(cat_values)
        tr = encoded.to_numpy(dtype=np.float32)
        params = dict(config["model"]["params"])
        params.pop("early_stopping_rounds", None)
        if rounds:
            params["n_estimators"] = rounds
        model = XGBClassifier(**params)
        model.fit(tr, target, verbose=False)
        model._stage1_ordinal_encoder = encoder
        return model, encoder
    raise ValueError(model_name)


def _load_folds(config: dict[str, Any], bundle: DataBundle) -> pd.DataFrame:
    fold_path = project_path(config["validation"]["fold_file"])
    if not fold_path.exists():
        create_and_save_folds(config)
    folds = pd.read_csv(fold_path)
    validate_fold_frame(folds, len(bundle.train), config["validation"]["n_splits"])
    if not np.array_equal(folds["target"].to_numpy(), bundle.train[bundle.schema.target_column].to_numpy()):
        raise ValueError("Fold targets do not match the current train.csv")
    if not np.array_equal(folds[bundle.schema.id_column].to_numpy(), bundle.train[bundle.schema.id_column].to_numpy()):
        raise ValueError("Fold IDs/order do not match the current train.csv")
    return folds


def train_experiment(
    config: dict[str, Any], model_name: str, feature_version: str, base_experiment_id: str
) -> dict[str, Any]:
    paths = ensure_output_dirs(config)
    bundle = load_data(config)
    folds = _load_folds(config, bundle)
    experiment_id = unique_experiment_id(base_experiment_id, paths)
    model_dir = paths["models"] / experiment_id
    model_dir.mkdir(parents=True, exist_ok=False)
    target = bundle.train[bundle.schema.target_column]
    train_features = build_features(bundle.train, bundle.schema.feature_columns, feature_version)
    test_features = build_features(bundle.test, bundle.schema.feature_columns, feature_version)
    align_feature_frames(train_features, test_features)
    oof = np.full(len(bundle.train), np.nan, dtype=np.float64)
    test_fold_predictions: list[np.ndarray] = []
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    importance_frames: list[pd.DataFrame] = []
    training_seconds = 0.0
    inference_seconds = 0.0

    for fold in range(config["validation"]["n_splits"]):
        print(
            f"[{experiment_id}] fold {fold}/{config['validation']['n_splits'] - 1} start",
            flush=True,
        )
        valid_mask = folds["fold"].to_numpy() == fold
        train_mask = ~valid_mask
        fit_start = time.perf_counter()
        result = _fit_fold(
            model_name, config,
            train_features.loc[train_mask], target.loc[train_mask],
            train_features.loc[valid_mask], target.loc[valid_mask],
            test_features, fold,
        )
        fold_elapsed = time.perf_counter() - fit_start
        # The fold function includes prediction; split inference is measured separately below
        # with a conservative test-prediction timing field retained as part of fold elapsed.
        training_seconds += max(0.0, fold_elapsed - result.inference_seconds)
        valid_prediction = np.asarray(result.validation_prediction, dtype=float)
        test_prediction = np.asarray(result.test_prediction, dtype=float)
        inference_seconds += result.inference_seconds
        validate_probabilities(valid_prediction, int(valid_mask.sum()), f"fold {fold} validation")
        validate_probabilities(test_prediction, len(bundle.test), f"fold {fold} test")
        oof[valid_mask] = valid_prediction
        test_fold_predictions.append(test_prediction)
        fold_scores.append(auc(target.loc[valid_mask].to_numpy(), valid_prediction))
        print(
            f"[{experiment_id}] fold {fold} AUC={fold_scores[-1]:.6f} "
            f"best_iteration={result.best_iteration} elapsed={fold_elapsed:.1f}s",
            flush=True,
        )
        if result.best_iteration is not None:
            best_iterations.append(result.best_iteration)
        if result.feature_importance is not None:
            importance_frames.append(result.feature_importance)
        _save_model(result.model, model_dir / f"fold_{fold}", model_name)

    validate_probabilities(oof, len(bundle.train), "OOF")
    test_prediction = np.mean(np.vstack(test_fold_predictions), axis=0)
    validate_probabilities(test_prediction, len(bundle.test), "test mean")
    full_oof_auc = auc(target.to_numpy(), oof)
    oof_frame = pd.DataFrame({
        "row_index": np.arange(len(bundle.train)),
        bundle.schema.id_column: bundle.train[bundle.schema.id_column].to_numpy(),
        "fold": folds["fold"].to_numpy(),
        "target": target.to_numpy(),
        "prediction": oof,
    })
    test_frame = pd.DataFrame({
        "row_index": np.arange(len(bundle.test)),
        bundle.schema.id_column: bundle.test[bundle.schema.id_column].to_numpy(),
        "prediction": test_prediction,
    })
    oof_frame.to_csv(paths["oof"] / f"{experiment_id}.csv", index=False)
    test_frame.to_csv(paths["predictions"] / f"{experiment_id}_test.csv", index=False)
    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(
            model_dir / "feature_importance.csv", index=False
        )
    if config["training"].get("train_full_model", True):
        print(f"[{experiment_id}] full-data model start", flush=True)
        full_start = time.perf_counter()
        full_model, _ = _fit_full_model(
            model_name, config, train_features, target, best_iterations
        )
        training_seconds += time.perf_counter() - full_start
        _save_model(full_model, model_dir / "full_model", model_name)
    metadata = {
        "experiment_id": experiment_id,
        "base_experiment_id": base_experiment_id,
        "model": model_name,
        "feature_version": feature_version,
        "fold_scores": fold_scores,
        "best_iterations": best_iterations,
        "feature_columns": list(train_features.columns),
        "categorical_policy": "feature type and mappings inferred independently in each training fold",
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    git = git_info()
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "planned_experiment_id": base_experiment_id,
        "timestamp": utc_now(),
        "status": "COMPLETE",
        "model": model_name,
        "feature_version": feature_version,
        "seed": config["project"]["seed"],
        "fold_version": Path(config["validation"]["fold_file"]).stem,
        "mean_fold_auc": float(np.mean(fold_scores)),
        "std_fold_auc": float(np.std(fold_scores)),
        "full_oof_auc": full_oof_auc,
        **{f"fold_{fold}_auc": fold_scores[fold] for fold in range(len(fold_scores))},
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "best_iterations": json.dumps(best_iterations),
        "prediction_min": float(oof.min()),
        "prediction_max": float(oof.max()),
        "prediction_mean": float(oof.mean()),
        "test_prediction_min": float(test_prediction.min()),
        "test_prediction_max": float(test_prediction.max()),
        "test_prediction_mean": float(test_prediction.mean()),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "config_path": config["_config_path"],
        "notes": "Strict OOF; fold-fitted preprocessing; ID excluded.",
        "error": None,
    }
    append_experiment_row(paths["experiments"] / "experiments.csv", row)
    print(
        f"[{experiment_id}] COMPLETE OOF={full_oof_auc:.6f} "
        f"fold_std={np.std(fold_scores):.6f}",
        flush=True,
    )
    return row


def record_failed_experiment(
    config: dict[str, Any], base_id: str, model: str, feature_version: str,
    status: str, error: str,
) -> dict[str, Any]:
    paths = ensure_output_dirs(config)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_id = f"{base_id}__{status}_{stamp}"
    git = git_info()
    row = {
        "experiment_id": experiment_id,
        "planned_experiment_id": base_id,
        "timestamp": utc_now(),
        "status": status,
        "model": model,
        "feature_version": feature_version,
        "seed": config["project"]["seed"],
        "fold_version": Path(config["validation"]["fold_file"]).stem,
        "mean_fold_auc": None, "std_fold_auc": None, "full_oof_auc": None,
        **{f"fold_{fold}_auc": None for fold in range(config["validation"]["n_splits"])},
        "training_seconds": 0.0, "inference_seconds": 0.0,
        "best_iterations": "[]", "prediction_min": None,
        "prediction_max": None, "prediction_mean": None,
        "test_prediction_min": None, "test_prediction_max": None,
        "test_prediction_mean": None, "git_commit": git["commit"],
        "git_dirty": git["dirty"], "config_path": config["_config_path"],
        "notes": status, "error": error[:2000],
    }
    append_experiment_row(paths["experiments"] / "experiments.csv", row)
    return row


def run_all_baselines() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    base_config = load_config("configs/baseline.yaml")
    try:
        load_data(base_config)
    except DataUnavailableError as exc:
        for base_id, model, feature_version, config_path in EXPERIMENT_PLAN:
            config = load_config(config_path)
            results.append(record_failed_experiment(
                config, base_id, model, feature_version, "BLOCKED_DATA", str(exc)
            ))
        return results

    for base_id, model, feature_version, config_path in EXPERIMENT_PLAN:
        config = load_config(config_path)
        print(f"[{base_id}] experiment start", flush=True)
        try:
            results.append(train_experiment(config, model, feature_version, base_id))
        except ImportError as exc:
            results.append(record_failed_experiment(
                config, base_id, model, feature_version, "BLOCKED_DEPENDENCY", str(exc)
            ))
        except Exception as exc:  # keep independent experiments running, but retain evidence
            details = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            results.append(record_failed_experiment(
                config, base_id, model, feature_version, "FAILED", details
            ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one Stage 1 experiment")
    parser.add_argument("--config", default="configs/lightgbm.yaml")
    parser.add_argument("--feature-version", choices=["raw_v1", "missing_v1"], default="raw_v1")
    parser.add_argument("--model-name", choices=["dummy", "logistic", "lightgbm", "catboost", "xgboost"])
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    config = load_config(args.config)
    model = args.model_name or config.get("model", {}).get("name")
    if not model:
        raise SystemExit("--model-name is required for baseline.yaml")
    shorthand = {"lightgbm": "lgb", "catboost": "cat", "xgboost": "xgb"}.get(model, model)
    experiment = args.experiment_id or (
        f"manual_{shorthand}_{args.feature_version}_seed{config['project']['seed']}"
    )
    row = train_experiment(config, model, args.feature_version, experiment)
    print(json.dumps(row, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
