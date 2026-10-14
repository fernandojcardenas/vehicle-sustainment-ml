"""Tests for the RUL metrics, including the asymmetric PHM score."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.scoring import phm_score, rmse, score_report


def test_rmse_zero_when_exact():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))


def test_phm_zero_when_exact():
    assert phm_score([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)


def test_phm_known_value():
    # one late (+3) and one early (-3) prediction.
    got = phm_score([10, 10], [13, 7])
    want = math.expm1(3 / 10) + math.expm1(3 / 13)
    assert got == pytest.approx(want)


def test_phm_penalises_late_more_than_early():
    # predicting more life than exists (late) is the dangerous error.
    late = phm_score([0], [5])
    early = phm_score([0], [-5])
    assert late > early


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        phm_score([1, 2], [1, 2, 3])


def test_score_report_fields():
    rep = score_report(np.array([10, 20]), np.array([12, 18]))
    assert set(rep) == {"rmse", "phm_score", "phm_score_per_engine", "n_engines"}
    assert rep["n_engines"] == 2
    assert rep["phm_score_per_engine"] == pytest.approx(rep["phm_score"] / 2)
