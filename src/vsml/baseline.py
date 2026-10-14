"""Classical remaining-useful-life baselines and honest evaluation.

For each subset we fit condition-based normalisation on the training engines,
train the models on capped-RUL labels, cross-validate *grouped by engine* (so
no engine's cycles leak between train and validation folds), and then score a
single prediction per test engine -- the model's output at that engine's last
observed cycle -- against the official RUL truth.

Two trivial baselines bound the useful range: predicting the training-mean RUL
and predicting the RUL cap. A learned model has to beat both to be worth
anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .columns import SUBSETS
from .data import DEFAULT_RUL_CAP, add_training_rul, last_cycle_rows, read_raw_table, read_test_rul
from .features import FeaturePipeline
from .scoring import score_report


def make_models(random_state: int = 0) -> dict:
    """The learned baselines. Kept small, deterministic and fast."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "hist_gbr": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.1,
            max_iter=300,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def cross_val_rmse(model, X, y, groups, n_splits: int = 5) -> tuple[float, float]:
    """Grouped CV RMSE (mean, std) -- groups are engine units."""
    from sklearn.base import clone

    gkf = GroupKFold(n_splits=n_splits)
    scores: list[float] = []
    for tr, va in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[va])
        scores.append(float(np.sqrt(np.mean((pred - y[va]) ** 2))))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_subset(
    subset: str,
    data_dir: str | Path,
    *,
    rul_cap: int | None = DEFAULT_RUL_CAP,
    random_state: int = 0,
    cv_splits: int = 5,
) -> dict:
    data_dir = Path(data_dir)
    train = add_training_rul(read_raw_table(data_dir / f"train_{subset}.txt"), cap=rul_cap)
    test = read_raw_table(data_dir / f"test_{subset}.txt")
    rul_true = read_test_rul(data_dir / f"RUL_{subset}.txt").to_numpy()

    fp = FeaturePipeline(n_conditions=SUBSETS[subset]["conditions"], random_state=random_state).fit(train)
    X_train = fp.transform(train)
    y_train = train["rul"].to_numpy()
    groups = train["unit"].to_numpy()

    test_last = last_cycle_rows(test)
    X_test = fp.transform(test_last)

    cap_hi = rul_cap if rul_cap is not None else float("inf")

    results: dict[str, dict] = {}

    # Trivial baselines.
    mean_rul = float(np.mean(y_train))
    for name, const in {"predict_mean": mean_rul, "predict_cap": float(cap_hi)}.items():
        pred = np.full(len(rul_true), min(const, cap_hi))
        results[name] = score_report(rul_true, pred)

    # Learned models.
    for name, model in make_models(random_state).items():
        cv_mean, cv_std = cross_val_rmse(model, X_train, y_train, groups, n_splits=cv_splits)
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0.0, cap_hi)
        rep = score_report(rul_true, pred)
        rep["cv_rmse_mean"] = cv_mean
        rep["cv_rmse_std"] = cv_std
        results[name] = rep

    return {
        "subset": subset,
        "rul_cap": rul_cap,
        "n_train_rows": int(len(train)),
        "n_features": int(X_train.shape[1]),
        "kept_sensors": fp.feature_names,
        "models": results,
    }
