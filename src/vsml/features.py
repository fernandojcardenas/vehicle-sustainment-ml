"""Feature engineering: operating-regime detection and per-regime normalisation.

Under multiple operating conditions (FD002/FD004) the raw sensor values jump
between six regimes, and that regime variation swamps the degradation signal.
The standard fix is condition-based normalisation: cluster the three
operational settings into regimes, then z-normalise each sensor *within* its
regime using statistics learned on the training engines only. Under a single
condition (FD001/FD003) there is one regime and this is ordinary z-scoring.

Near-constant sensors (flat within a subset) carry no information and are
dropped, with the drop recorded so the choice is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .columns import OP_SETTINGS, SENSORS


@dataclass
class FeaturePipeline:
    """Fit regimes + per-regime sensor normalisation on train, apply to any split."""

    n_conditions: int
    random_state: int = 0
    _op_scaler: StandardScaler | None = None
    _kmeans: KMeans | None = None
    kept_sensors: list[str] | None = None
    _stats: dict[int, dict[str, tuple[float, float]]] | None = None

    def fit(self, train: pd.DataFrame) -> FeaturePipeline:
        # Which sensors carry information in this subset (learned on train).
        self.kept_sensors = [c for c in SENSORS if float(train[c].std(ddof=0)) >= 1e-6]

        # Regime assignment from the three operational settings.
        if self.n_conditions > 1:
            self._op_scaler = StandardScaler().fit(train[OP_SETTINGS].to_numpy())
            x = self._op_scaler.transform(train[OP_SETTINGS].to_numpy())
            self._kmeans = KMeans(
                n_clusters=self.n_conditions, n_init=10, random_state=self.random_state
            ).fit(x)
            regimes = self._kmeans.labels_
        else:
            regimes = np.zeros(len(train), dtype=int)

        # Per-regime mean/std for each kept sensor.
        self._stats = {}
        tmp = train.copy()
        tmp["_regime"] = regimes
        for r, grp in tmp.groupby("_regime"):
            self._stats[int(r)] = {}
            for c in self.kept_sensors:
                mu = float(grp[c].mean())
                sd = float(grp[c].std(ddof=0))
                self._stats[int(r)][c] = (mu, sd if sd >= 1e-9 else 1.0)
        return self

    def _regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions > 1:
            assert self._op_scaler is not None and self._kmeans is not None
            x = self._op_scaler.transform(df[OP_SETTINGS].to_numpy())
            return self._kmeans.predict(x)
        return np.zeros(len(df), dtype=int)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return the normalised feature matrix (rows x kept_sensors)."""
        assert self.kept_sensors is not None and self._stats is not None
        regimes = self._regimes(df)
        out = np.empty((len(df), len(self.kept_sensors)), dtype=float)
        vals = {c: df[c].to_numpy() for c in self.kept_sensors}
        for j, c in enumerate(self.kept_sensors):
            col = vals[c].astype(float)
            mu = np.array([self._stats[int(r)][c][0] for r in regimes])
            sd = np.array([self._stats[int(r)][c][1] for r in regimes])
            out[:, j] = (col - mu) / sd
        return out

    @property
    def feature_names(self) -> list[str]:
        assert self.kept_sensors is not None
        return list(self.kept_sensors)
