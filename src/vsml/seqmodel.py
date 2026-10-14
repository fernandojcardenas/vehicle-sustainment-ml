"""A small, deterministic 1D-CNN for remaining-useful-life, in PyTorch.

Windows arrive as (batch, time, features); a temporal CNN over the time axis
with global average pooling maps each window to a single RUL value. The model is
intentionally small so it trains on CPU in CI. Training is seeded and uses a
grouped validation split (by engine) with early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .scoring import rmse


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class TemporalCNN(nn.Module):
    """Three 1D-conv blocks over time, global-average-pooled, then an MLP head."""

    def __init__(self, n_features: int, channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) -> (batch, features, time)
        x = x.permute(0, 2, 1)
        return self.head(self.net(x)).squeeze(-1)


@dataclass
class TrainConfig:
    window: int = 30
    channels: int = 32
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 8
    val_frac: float = 0.2
    rul_cap: float = 125.0
    seed: int = 0


@dataclass
class TrainResult:
    model: TemporalCNN
    best_val_rmse: float
    epochs_run: int
    history: list[float] = field(default_factory=list)


def _grouped_val_split(groups: np.ndarray, val_frac: float, seed: int):
    units = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(units)
    n_val = max(1, int(round(len(units) * val_frac)))
    val_units = set(perm[:n_val].tolist())
    val_mask = np.array([g in val_units for g in groups])
    return ~val_mask, val_mask


def train_cnn(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    cfg: TrainConfig,
) -> TrainResult:
    set_seed(cfg.seed)
    n_features = windows.shape[2]
    tr_mask, va_mask = _grouped_val_split(groups, cfg.val_frac, cfg.seed)

    Xtr = torch.from_numpy(windows[tr_mask])
    ytr = torch.from_numpy(labels[tr_mask].astype(np.float32))
    Xva = torch.from_numpy(windows[va_mask])
    yva = labels[va_mask].astype(np.float32)

    model = TemporalCNN(n_features, cfg.channels, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    gen = torch.Generator().manual_seed(cfg.seed)
    n = Xtr.shape[0]

    best_rmse = float("inf")
    best_state = None
    bad = 0
    history: list[float] = []
    epochs_run = 0

    for epoch in range(cfg.max_epochs):
        epochs_run = epoch + 1
        model.train()
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s : s + cfg.batch_size]
            opt.zero_grad()
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xva).numpy()
        vp = np.clip(vp, 0.0, cfg.rul_cap)
        val_rmse = rmse(yva, vp)
        history.append(val_rmse)

        if val_rmse < best_rmse - 1e-4:
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_val_rmse=best_rmse, epochs_run=epochs_run, history=history)


def predict(model: TemporalCNN, windows: np.ndarray, rul_cap: float = 125.0) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(windows.astype(np.float32))).numpy()
    return np.clip(out, 0.0, rul_cap)
