"""Data-quality checks and lineage metadata for the C-MAPSS pipeline.

The checks turn the readme's implicit assumptions into explicit, testable
gates: the right columns and types, no missing values, cycles that start at 1
and increase by 1 per engine, the documented engine counts, and physically
sane sensor ranges. ``near_constant_sensors`` reports sensors that carry no
information in a given subset (several are flat under a single operating
condition) so features can drop them with a record of why.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import CYCLE, OP_SETTINGS, RAW_COLUMNS, SENSORS, SUBSETS, UNIT


class DataQualityError(ValueError):
    """Raised when a hard data-quality gate fails."""


@dataclass
class QualityReport:
    subset: str
    split: str
    rows: int
    units: int
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    near_constant_sensors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "split": self.split,
            "rows": self.rows,
            "units": self.units,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "near_constant_sensors": self.near_constant_sensors,
        }


def check_table(df: pd.DataFrame, subset: str, split: str, *, strict: bool = True) -> QualityReport:
    """Run the quality gates on one train/test table."""
    rep = QualityReport(subset=subset, split=split, rows=len(df), units=int(df[UNIT].nunique()))

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")

    if df[RAW_COLUMNS].isna().any().any():
        bad = df[RAW_COLUMNS].isna().sum()
        rep.errors.append(f"null values present: {bad[bad > 0].to_dict()}")

    # Cycles: per engine, start at 1 and increase by exactly 1.
    g = df.sort_values([UNIT, CYCLE]).groupby(UNIT)[CYCLE]
    starts_ok = bool((g.min() == 1).all())
    if not starts_ok:
        rep.errors.append("some engines do not start at cycle 1")
    steps = df.sort_values([UNIT, CYCLE]).groupby(UNIT)[CYCLE].diff().dropna()
    if not bool((steps == 1).all()):
        rep.errors.append("cycles are not strictly consecutive (+1) within every engine")

    # Documented engine counts.
    expected = SUBSETS[subset][f"{split}_units"]
    if rep.units != expected:
        rep.errors.append(f"expected {expected} {split} engines, found {rep.units}")

    # Physical sanity: op settings and sensors finite; sensors non-negative
    # except the op settings (settings 1/2 can be negative deltas).
    for c in OP_SETTINGS + SENSORS:
        col = df[c].to_numpy()
        if not np.isfinite(col).all():
            rep.errors.append(f"{c} has non-finite values")
    for c in SENSORS:
        if (df[c].to_numpy() < 0).any():
            rep.warnings.append(f"{c} has negative values (unexpected for a sensor)")

    # Near-constant sensors (no information in this subset).
    for c in SENSORS:
        if float(df[c].std(ddof=0)) < 1e-6:
            rep.near_constant_sensors.append(c)

    rep.passed = not rep.errors
    if strict and not rep.passed:
        raise DataQualityError(f"{subset}/{split} failed quality checks: {rep.errors}")
    return rep


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lineage(
    *,
    subset: str,
    source_files: dict[str, str | Path],
    rul_cap: int | None,
    extra: dict | None = None,
) -> dict:
    """Build a lineage record: where the rows came from and how they were made."""
    rec = {
        "subset": subset,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rul_cap": rul_cap,
        "source": {
            name: {"path": str(p), "sha256": file_sha256(p)}
            for name, p in source_files.items()
        },
    }
    if extra:
        rec.update(extra)
    return rec


def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
