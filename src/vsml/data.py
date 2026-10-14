"""Load the raw C-MAPSS text files and build RUL labels.

The data are fetched by ``scripts/fetch_data.py`` into ``data/raw/`` and are
never committed. Each subset (FD001..FD004) has ``train_*.txt``, ``test_*.txt``
and ``RUL_*.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .columns import CYCLE, RAW_COLUMNS, UNIT

# Piecewise-linear RUL cap. Early in life the engine shows no degradation, so a
# raw linear RUL (max_cycle - cycle) asks the model to predict large values it
# cannot possibly see evidence for. Capping the label is standard practice
# (Heimes, 2008) and is recorded as a modelling choice in docs/adr/0002.
DEFAULT_RUL_CAP = 125


def read_raw_table(path: str | Path) -> pd.DataFrame:
    """Read one raw C-MAPSS text file into a typed, named DataFrame."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    )
    # Files have 26 real columns; some rows carry trailing whitespace that can
    # produce a 27th all-NaN column. Keep exactly the 26 documented columns.
    if df.shape[1] < len(RAW_COLUMNS):
        raise ValueError(f"{path}: expected >= {len(RAW_COLUMNS)} columns, got {df.shape[1]}")
    df = df.iloc[:, : len(RAW_COLUMNS)]
    df.columns = RAW_COLUMNS
    df[UNIT] = df[UNIT].astype("int32")
    df[CYCLE] = df[CYCLE].astype("int32")
    for c in RAW_COLUMNS[2:]:
        df[c] = df[c].astype("float64")
    return df


def add_training_rul(df: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Add a ``rul`` column to a *training* table.

    For a run-to-failure training trajectory the RUL at a cycle is
    ``max_cycle_for_unit - cycle``; optionally capped.
    """
    out = df.copy()
    max_cycle = out.groupby(UNIT)[CYCLE].transform("max")
    rul = (max_cycle - out[CYCLE]).astype("int32")
    if cap is not None:
        rul = rul.clip(upper=cap)
    out["rul"] = rul.astype("int32")
    return out


def read_test_rul(path: str | Path) -> pd.Series:
    """Read a ``RUL_*.txt`` file: one true-RUL value per test engine, in order."""
    vals = pd.read_csv(path, header=None).iloc[:, 0].astype("int32")
    vals.index = range(1, len(vals) + 1)  # 1-based unit numbers
    vals.name = "rul_true"
    return vals


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The final observed cycle of each engine -- what the test truth aligns to."""
    idx = df.groupby(UNIT)[CYCLE].idxmax()
    return df.loc[idx].sort_values(UNIT).reset_index(drop=True)
