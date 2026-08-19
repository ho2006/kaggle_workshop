from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import project_path
from src.data import DataUnavailableError, load_data
from src.utils import ensure_output_dirs, utc_now


def make_fold_frame(
    train: pd.DataFrame,
    target_column: str,
    id_column: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    target = train[target_column]
    if target.nunique() != 2:
        raise ValueError(f"Stratified binary folds require two classes, found {target.nunique()}")
    folds = pd.Series(-1, index=train.index, dtype="int8")
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None
    )
    for fold, (_, validation_index) in enumerate(splitter.split(train, target)):
        folds.iloc[validation_index] = fold
    result = pd.DataFrame({
        "row_index": range(len(train)),
        id_column: train[id_column].to_numpy(),
        "fold": folds.to_numpy(),
        "target": target.to_numpy(),
    })
    validate_fold_frame(result, len(train), n_splits)
    return result


def validate_fold_frame(frame: pd.DataFrame, train_length: int, n_splits: int) -> None:
    required = {"row_index", "fold", "target"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Fold file missing columns: {required - set(frame.columns)}")
    if len(frame) != train_length or frame["row_index"].nunique() != train_length:
        raise ValueError("Every training row must appear exactly once in the fold file")
    if frame["fold"].isna().any() or set(frame["fold"].unique()) != set(range(n_splits)):
        raise ValueError("Fold numbers are missing or outside the configured range")
    if (frame.groupby("fold")["target"].nunique() != 2).any():
        raise ValueError("Every validation fold must contain both target classes")


def create_and_save_folds(config: dict[str, Any]) -> pd.DataFrame:
    paths = ensure_output_dirs(config)
    report_path = paths["reports"] / "fold_report.md"
    try:
        bundle = load_data(config)
    except DataUnavailableError as exc:
        report_path.write_text(
            f"# Fold Report\n\n## Status: BLOCKED_DATA\n\n{exc}\n\n"
            "No fold assignments were generated without the real training labels.\n",
            encoding="utf-8",
        )
        raise
    validation = config["validation"]
    folds = make_fold_frame(
        bundle.train,
        bundle.schema.target_column,
        bundle.schema.id_column,
        validation["n_splits"],
        validation["shuffle"],
        validation["random_state"],
    )
    fold_path = project_path(validation["fold_file"])
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    if fold_path.exists():
        existing = pd.read_csv(fold_path)
        pd.testing.assert_frame_equal(existing, folds, check_dtype=False)
    else:
        folds.to_csv(fold_path, index=False)
    stats = folds.groupby("fold")["target"].agg(["count", "sum", "mean"])
    stats["negative"] = stats["count"] - stats["sum"]
    try:
        fold_display = str(fold_path.relative_to(project_path(".")))
    except ValueError:
        fold_display = str(fold_path)
    lines = [
        "# Fold Report", "", "## Status: COMPLETE", "", f"Generated: {utc_now()}", "",
        f"Strategy: StratifiedKFold(n_splits={validation['n_splits']}, "
        f"shuffle={validation['shuffle']}, random_state={validation['random_state']})",
        "", f"Fold file: `{fold_display}`", "",
        "| Fold | Rows | Positive | Negative | Positive rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold, row in stats.iterrows():
        lines.append(
            f"| {fold} | {int(row['count']):,} | {int(row['sum']):,} | "
            f"{int(row['negative']):,} | {row['mean']:.6f} |"
        )
    lines.extend(["", "All rows occur once, all fold IDs are valid, and every fold contains both classes."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folds
