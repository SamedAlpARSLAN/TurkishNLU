# Convenience targets (Linux/macOS/Kaggle). On Windows, use run.bat instead.
.PHONY: install test data validate smoke matrix aggregate clean

install:
	python -m pip install -r requirements.txt

test:
	python tests/test_core.py
	python -m pytest tests/ -q

data:
	python scripts/download_data.py

validate:
	python scripts/validate_alignment.py --data data/raw/tr-TR.jsonl --models berturk mbert xlmr

smoke:
	python -m src.train --config configs/smoke.yaml

matrix:
	python scripts/run_matrix.py

aggregate:
	python scripts/aggregate.py

clean:
	rm -rf outputs/* __pycache__ src/__pycache__ scripts/__pycache__ tests/__pycache__
