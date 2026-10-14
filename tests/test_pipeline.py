"""Data-dependent tests: the pipeline and the learned baselines on real data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.baseline import evaluate_subset
from vsml.columns import ALL_SUBSETS, RAW_COLUMNS
from vsml.data import last_cycle_rows, read_raw_table, read_test_rul
from vsml.pipeline import build_subset

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "docs" / "evidence" / "baseline_results.json"


def test_test_units_match_rul_truth(raw_dir: Path):
    for s in ALL_SUBSETS:
        te = read_raw_table(raw_dir / f"test_{s}.txt")
        rul = read_test_rul(raw_dir / f"RUL_{s}.txt")
        assert te["unit"].nunique() == len(rul), s
        assert len(last_cycle_rows(te)) == len(rul), s


def test_build_subset_writes_parquet(raw_dir: Path, tmp_path: Path):
    rec = build_subset("FD001", raw_dir, tmp_path)
    assert rec["quality"]["train"]["passed"]
    tr = __import__("pandas").read_parquet(tmp_path / "parquet" / "train_FD001.parquet")
    assert list(tr.columns) == RAW_COLUMNS + ["rul"]
    assert tr["rul"].max() <= 125 and tr["rul"].min() == 0


def test_learned_models_beat_trivial_baselines(raw_dir: Path):
    res = evaluate_subset("FD001", raw_dir)
    mean_rmse = res["models"]["predict_mean"]["rmse"]
    for learned in ("ridge", "hist_gbr"):
        assert res["models"][learned]["rmse"] < mean_rmse
        assert res["models"][learned]["phm_score"] < res["models"]["predict_mean"]["phm_score"]


@pytest.mark.skipif(not COMMITTED.exists(), reason="committed results not present")
@pytest.mark.parametrize("subset", ALL_SUBSETS)
def test_reproduces_committed_numbers(raw_dir: Path, subset: str):
    committed = {r["subset"]: r for r in json.loads(COMMITTED.read_text())}
    res = evaluate_subset(subset, raw_dir)
    for model in ("ridge", "hist_gbr"):
        got = res["models"][model]["rmse"]
        want = committed[subset]["models"][model]["rmse"]
        assert got == pytest.approx(want, rel=1e-3), (subset, model, got, want)
