# Kaggle Playground S6E8 — Stage 1 baseline

Leakage-safe, reproducible binary-classification baselines for **Predicting
Smartphone Addiction** (Playground Series, Season 6 Episode 8).

## Current data status

All three competition files are available under `data/raw/`. The validated
schema uses `addicted_label` as the binary target and `id` as the submission ID.
See `data/README.md`.

## Setup

```bash
python -m pip install -r requirements.txt
```

Put competition files in `data/raw/` (or at repository root), then run:

```bash
python scripts/run_stage1.py
```

That command runs the audit, creates deterministic folds, trains the planned
baselines, selects the best stable completed experiment, builds a submission,
runs tests, and refreshes Markdown reports. Existing model/OOF/prediction files
are never silently overwritten; a timestamp suffix is used when needed.

Individual commands:

```bash
python scripts/run_audit.py
python scripts/make_folds.py
python scripts/train_baselines.py
python scripts/make_submission.py --best
python scripts/make_submission.py --prediction artifacts/predictions/<experiment_id>_test.csv
python -m src.train --config configs/lightgbm.yaml --feature-version raw_v1
python -m pytest -q
```

On systems with Make:

```bash
make stage1
```

## Planned experiments

| Base ID | Model | Features |
|---|---|---|
| E000 | DummyClassifier(prior) | raw_v1 |
| E001 | LogisticRegression | raw_v1 |
| E002 | LightGBM | raw_v1 |
| E003 | LightGBM | missing_v1 |
| E004 | CatBoost | raw_v1 |
| E005 | XGBoost (CPU hist) | raw_v1 |

All completed models load the same persisted five-fold assignment. Any learned
preprocessing is fitted inside each training fold. Test predictions are the mean
of the five fold models; a full-data model is also saved for traceability.

The Stage 1 metric is ROC-AUC, following the requested binary-classification
protocol after the single probability target was confirmed. The official
evaluation text could not be independently retrieved in the unauthenticated
session, so it should be rechecked on Kaggle after the competition rules are
accepted.

## Main outputs

- `artifacts/audit/`: machine-readable audit evidence
- `artifacts/folds/folds_seed42.csv`: fixed fold assignments
- `artifacts/oof/`: strict out-of-fold predictions
- `artifacts/predictions/`: test probabilities
- `artifacts/models/`: per-fold and full-data models
- `artifacts/experiments/experiments.csv`: append-only experiment ledger
- `artifacts/reports/`: environment, audit, fold, baseline, and stage summaries
- `submissions/`: validated Kaggle submissions and metadata
