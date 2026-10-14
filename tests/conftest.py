"""Shared fixtures: locate the raw data, or skip the data-dependent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REQUIRED = [f"{p}_FD001.txt" for p in ("train", "test", "RUL")]


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    if not all((RAW / f).exists() for f in REQUIRED):
        pytest.skip("raw C-MAPSS data not present (run scripts/fetch_data.py)")
    return RAW
