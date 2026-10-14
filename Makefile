.PHONY: help setup data dataset baseline test lint all

help:
	@echo "setup     install Python dependencies"
	@echo "data      download + verify the C-MAPSS data (data/raw/)"
	@echo "dataset   build the clean Parquet data set (data/processed/)"
	@echo "baseline  train + evaluate the RUL baselines (docs/)"
	@echo "test      run the test suite"
	@echo "lint      run ruff"
	@echo "all       data -> dataset -> baseline -> test"

setup:
	pip install -r requirements-dev.txt

data:
	python scripts/fetch_data.py

dataset:
	python scripts/build_dataset.py

baseline:
	python scripts/train_baseline.py

test:
	python -m pytest -q

lint:
	ruff check .

all: data dataset baseline test
