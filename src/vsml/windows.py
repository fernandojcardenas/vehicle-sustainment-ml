"""Turn per-engine feature rows into fixed-length time windows for a sequence model.

For each engine the rows are already in cycle order. A window of length ``W``
ending at cycle *t* is the last ``W`` cycles up to and including *t*; when fewer
than ``W`` cycles precede *t* the window is left-padded with the engine's first
row (so every cycle yields a window and short test engines are handled the same
way as training ones).

Training uses every cycle's window (one label per row). Test uses the single
last window of each engine, aligned to the official RUL truth.
"""

from __future__ import annotations

import numpy as np

from .columns import CYCLE, UNIT


def _engine_windows(feat: np.ndarray, window: int) -> np.ndarray:
    """All left-padded windows for one engine's (cycles x features) array.

    Returns an array of shape (n_cycles, window, n_features).
    """
    n, f = feat.shape
    out = np.empty((n, window, f), dtype=np.float32)
    for i in range(n):
        start = i - window + 1
        if start >= 0:
            out[i] = feat[start : i + 1]
        else:
            pad = np.repeat(feat[:1], -start, axis=0)
            out[i] = np.concatenate([pad, feat[: i + 1]], axis=0)
    return out


def make_training_windows(
    df, X: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows, labels and engine ids for every training row.

    ``df`` carries ``unit``, ``cycle`` and ``rul``; ``X`` is the row-aligned
    feature matrix. Returns (windows [N, W, F], labels [N], groups [N]).
    """
    order = df.sort_values([UNIT, CYCLE]).index.to_numpy()
    units = df.loc[order, UNIT].to_numpy()
    rul = df.loc[order, "rul"].to_numpy().astype(np.float32)
    Xo = X[df.index.get_indexer(order)]

    wins, labels, groups = [], [], []
    for u in np.unique(units):
        mask = units == u
        w = _engine_windows(Xo[mask], window)
        wins.append(w)
        labels.append(rul[mask])
        groups.append(np.full(mask.sum(), u))
    return (
        np.concatenate(wins, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(groups, axis=0),
    )


def last_window_per_engine(df, X: np.ndarray, window: int) -> np.ndarray:
    """One window per engine: the last ``window`` cycles (left-padded). [n_engines, W, F].

    Engines are returned in ascending unit order, matching ``read_test_rul``.
    """
    order = df.sort_values([UNIT, CYCLE]).index.to_numpy()
    units = df.loc[order, UNIT].to_numpy()
    Xo = X[df.index.get_indexer(order)]
    out = []
    for u in np.unique(units):
        feat = Xo[units == u]
        out.append(_engine_windows(feat, window)[-1])
    return np.stack(out, axis=0).astype(np.float32)
