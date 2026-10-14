"""Evaluate the sequence model on a C-MAPSS subset, comparable to the baselines.

Same data, same per-regime normalisation and same capped-RUL labels as the
classical baselines (M1); the only change is the model. One prediction per test
engine (its last window) is scored against the official RUL truth.
"""

from __future__ import annotations

from pathlib import Path

from .columns import SUBSETS
from .data import add_training_rul, read_raw_table, read_test_rul
from .features import FeaturePipeline
from .scoring import score_report
from .seqmodel import TrainConfig, predict, train_cnn
from .windows import last_window_per_engine, make_training_windows


def evaluate_subset_cnn(
    subset: str,
    data_dir: str | Path,
    *,
    cfg: TrainConfig | None = None,
) -> dict:
    cfg = cfg or TrainConfig()
    data_dir = Path(data_dir)

    train = add_training_rul(read_raw_table(data_dir / f"train_{subset}.txt"), cap=cfg.rul_cap)
    test = read_raw_table(data_dir / f"test_{subset}.txt")
    rul_true = read_test_rul(data_dir / f"RUL_{subset}.txt").to_numpy()

    fp = FeaturePipeline(n_conditions=SUBSETS[subset]["conditions"], random_state=cfg.seed).fit(train)
    Xtr = fp.transform(train)
    Xte = fp.transform(test)

    windows, labels, groups = make_training_windows(train, Xtr, cfg.window)
    res = train_cnn(windows, labels, groups, cfg)

    test_windows = last_window_per_engine(test, Xte, cfg.window)
    pred = predict(res.model, test_windows, rul_cap=cfg.rul_cap)

    rep = score_report(rul_true, pred)
    rep.update(
        {
            "val_rmse": res.best_val_rmse,
            "epochs_run": res.epochs_run,
            "n_features": int(windows.shape[2]),
            "window": cfg.window,
            "n_train_windows": int(windows.shape[0]),
        }
    )
    return {"subset": subset, "model": "temporal_cnn", "metrics": rep}
