"""Tests for windowing (synthetic) and the sequence model (data-dependent)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.columns import CYCLE, UNIT
from vsml.windows import _engine_windows, last_window_per_engine, make_training_windows


def _toy_feature_df(units_cycles: dict[int, int]):
    rows, feats = [], []
    for u, n in units_cycles.items():
        for c in range(1, n + 1):
            rows.append({UNIT: u, CYCLE: c, "rul": n - c})
            feats.append([float(u * 100 + c)])  # 1 feature, unique per row
    return pd.DataFrame(rows), np.array(feats, dtype=np.float32)


def test_engine_windows_left_pad():
    feat = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    w = _engine_windows(feat, window=3)
    assert w.shape == (3, 3, 1)
    # first row's window is the first value padded three times
    assert w[0].ravel().tolist() == [1.0, 1.0, 1.0]
    assert w[1].ravel().tolist() == [1.0, 1.0, 2.0]
    assert w[2].ravel().tolist() == [1.0, 2.0, 3.0]


def test_make_training_windows_counts_and_labels():
    df, X = _toy_feature_df({1: 5, 2: 3})
    wins, labels, groups = make_training_windows(df, X, window=4)
    assert wins.shape == (8, 4, 1)  # one window per row
    assert labels.shape == (8,) and groups.shape == (8,)
    # labels are the per-engine RUL in order
    assert labels[groups == 1].tolist() == [4, 3, 2, 1, 0]
    assert labels[groups == 2].tolist() == [2, 1, 0]


def test_last_window_per_engine_one_each_in_order():
    df, X = _toy_feature_df({2: 3, 1: 6})
    last = last_window_per_engine(df, X, window=4)
    assert last.shape == (2, 4, 1)  # engines 1 and 2, ascending
    # engine 1's last window ends at its last feature value (1*100+6 = 106)
    assert last[0, -1, 0] == 106.0
    assert last[1, -1, 0] == 203.0


@pytest.mark.slow
def test_cnn_trains_beats_trivial_and_is_deterministic(raw_dir: Path):
    from vsml.baseline import evaluate_subset
    from vsml.seqeval import evaluate_subset_cnn
    from vsml.seqmodel import TrainConfig

    cfg = TrainConfig(seed=0, max_epochs=20)
    r1 = evaluate_subset_cnn("FD001", raw_dir, cfg=cfg)["metrics"]["rmse"]
    r2 = evaluate_subset_cnn("FD001", raw_dir, cfg=cfg)["metrics"]["rmse"]
    assert r1 == r2  # deterministic with a fixed seed

    trivial = evaluate_subset("FD001", raw_dir)["models"]["predict_mean"]["rmse"]
    assert r1 < trivial  # must beat predicting the mean
    assert r1 < 25.0  # and land in the useful range
