"""Evaluation metrics for remaining-useful-life (RUL) prediction.

Two metrics are reported:

* RMSE -- symmetric, easy to compare across studies.
* The C-MAPSS / PHM08 asymmetric score -- late predictions (predicting more
  life than the engine actually has, d = pred - true > 0) are penalised more
  heavily than early ones, because overestimating remaining life is the
  dangerous error for maintenance. Lower is better; it is unbounded and
  dominated by the worst engines, so it is always read alongside RMSE.

The score is defined per engine on the final prediction:

    d = pred - true
    s = exp(-d / 13) - 1   if d <  0   (early, conservative)
        exp( d / 10) - 1   if d >= 0   (late, dangerous)

and summed over engines (Saxena et al., 2008).
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-MAPSS/PHM08 asymmetric score (sum over engines). Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    d = y_pred - y_true
    penalty = np.where(d < 0.0, np.expm1(-d / 13.0), np.expm1(d / 10.0))
    return float(np.sum(penalty))


def score_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Both metrics plus the mean per-engine PHM score, as a dict."""
    n = int(np.asarray(y_true).shape[0])
    s = phm_score(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "phm_score": s,
        "phm_score_per_engine": s / n if n else float("nan"),
        "n_engines": n,
    }
