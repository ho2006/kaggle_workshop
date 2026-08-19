.PHONY: audit folds baseline submission test stage1

audit:
	python scripts/run_audit.py

folds:
	python scripts/make_folds.py

baseline:
	python scripts/train_baselines.py

submission:
	python scripts/make_submission.py --best

test:
	python -m pytest -q

stage1:
	python scripts/run_stage1.py

