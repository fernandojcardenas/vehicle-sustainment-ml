"""Tests for the data-quality gates, using small synthetic tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.columns import RAW_COLUMNS
from vsml.quality import DataQualityError, check_table


def _good(n_units: int, cycles: int, subset: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, cycles + 1):
            rows.append([u, c, *rng.normal(size=len(RAW_COLUMNS) - 2)])
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    # sensors non-negative to avoid the warning path
    for col in RAW_COLUMNS[5:]:
        df[col] = df[col].abs()
    return df


def test_good_table_passes():
    rep = check_table(_good(100, 30, "FD001"), "FD001", "train")
    assert rep.passed and not rep.errors


def test_null_fails():
    df = _good(100, 30, "FD001")
    df.loc[0, "sensor_2"] = np.nan
    with pytest.raises(DataQualityError):
        check_table(df, "FD001", "train")


def test_non_consecutive_cycles_fail():
    df = _good(100, 30, "FD001")
    df.loc[df.index[5], "cycle"] = 99  # break the +1 run for unit 1
    with pytest.raises(DataQualityError):
        check_table(df, "FD001", "train")


def test_wrong_engine_count_fails():
    with pytest.raises(DataQualityError):
        check_table(_good(99, 30, "FD001"), "FD001", "train")


def test_non_strict_returns_report_instead_of_raising():
    rep = check_table(_good(99, 30, "FD001"), "FD001", "train", strict=False)
    assert not rep.passed and rep.errors
