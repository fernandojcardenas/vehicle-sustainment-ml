"""Tests for raw loading and RUL labelling, using small synthetic tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.columns import RAW_COLUMNS
from vsml.data import add_training_rul, last_cycle_rows, read_test_rul


def _toy(units_cycles: dict[int, int]) -> pd.DataFrame:
    rows = []
    for unit, n in units_cycles.items():
        for c in range(1, n + 1):
            rows.append([unit, c] + [0.0] * (len(RAW_COLUMNS) - 2))
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_rul_uncapped_counts_down_to_zero():
    df = add_training_rul(_toy({1: 5}), cap=None)
    assert list(df["rul"]) == [4, 3, 2, 1, 0]


def test_rul_cap_applied():
    df = add_training_rul(_toy({1: 10}), cap=3)
    assert list(df["rul"]) == [3, 3, 3, 3, 3, 3, 3, 2, 1, 0]


def test_rul_per_unit_independent():
    df = add_training_rul(_toy({1: 3, 2: 2}), cap=None)
    assert list(df[df["unit"] == 1]["rul"]) == [2, 1, 0]
    assert list(df[df["unit"] == 2]["rul"]) == [1, 0]


def test_last_cycle_rows_picks_max_cycle():
    last = last_cycle_rows(_toy({1: 4, 2: 7}))
    assert dict(zip(last["unit"], last["cycle"], strict=True)) == {1: 4, 2: 7}


def test_read_test_rul_is_one_based(tmp_path: Path):
    p = tmp_path / "RUL_FD001.txt"
    p.write_text("112\n98\n69\n")
    s = read_test_rul(p)
    assert list(s.index) == [1, 2, 3]
    assert list(s) == [112, 98, 69]
