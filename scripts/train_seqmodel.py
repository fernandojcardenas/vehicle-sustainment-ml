#!/usr/bin/env python3
"""Train + evaluate the 1D-CNN sequence model on all four C-MAPSS subsets.

Writes docs/evidence/seqmodel_results.json and docs/model-card.md, and prints a
table comparing the CNN to the M1 hist-gradient-boosting baseline.

Usage:
  python scripts/train_seqmodel.py [--raw data/raw] [--seed 0] [--subsets FD001 ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.columns import ALL_SUBSETS  # noqa: E402
from vsml.seqeval import evaluate_subset_cnn  # noqa: E402
from vsml.seqmodel import TrainConfig  # noqa: E402

BASELINE_JSON = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "baseline_results.json"


def _baseline_rmse() -> dict[str, float]:
    if not BASELINE_JSON.exists():
        return {}
    data = {r["subset"]: r for r in json.loads(BASELINE_JSON.read_text())}
    return {s: data[s]["models"]["hist_gbr"]["rmse"] for s in data}


def model_card(results: list[dict], base: dict[str, float], cfg: TrainConfig) -> str:
    lines = ["# Model card -- Temporal CNN for RUL\n"]
    lines.append(
        "A small 1D convolutional network over a sliding window of the last "
        f"{cfg.window} cycles of per-regime-normalised sensors, predicting "
        "remaining useful life. Same data, normalisation and capped-RUL labels "
        "(cap 125) as the M1 baselines; one prediction per test engine (its last "
        "window) scored against the official RUL truth.\n"
    )
    lines.append("## Intended use\n")
    lines.append(
        "A portfolio demonstration of sequence modelling for predictive "
        "maintenance on an open benchmark; not a certified safety system.\n"
    )
    lines.append("## Architecture\n")
    lines.append(
        f"- Input: window {cfg.window} x features (per-subset kept sensors)\n"
        f"- 3 x Conv1d({cfg.channels}) over time + global average pool + MLP head "
        f"(dropout {cfg.dropout})\n"
        f"- Adam (lr {cfg.lr}), MSE loss, batch {cfg.batch_size}, up to "
        f"{cfg.max_epochs} epochs, early stopping (patience {cfg.patience}) on a "
        f"{int(cfg.val_frac * 100)}% engine-grouped validation split; seed {cfg.seed}\n"
    )
    lines.append("## Results (test RMSE, cycles)\n")
    lines.append("| Subset | CNN RMSE | M1 hist-GBR RMSE | CNN PHM score | Val RMSE | Epochs |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        s = r["subset"]
        m = r["metrics"]
        b = f"{base[s]:.2f}" if s in base else "—"
        lines.append(
            f"| {s} | {m['rmse']:.2f} | {b} | {m['phm_score']:.0f} | "
            f"{m['val_rmse']:.2f} | {m['epochs_run']} |"
        )
    lines.append("")
    lines.append(
        "The CNN is compared like-for-like against the strongest M1 baseline. "
        "The validation RMSE (on held-out training engines) is lower than the "
        "test RMSE because the test trajectories are truncated at varied points; "
        "this gap is expected on C-MAPSS and is reported rather than hidden.\n"
    )
    lines.append("## Limitations\n")
    lines.append(
        "- CPU-sized model, few epochs: competitive with the classical baseline "
        "but below heavily tuned deep models in the literature.\n"
        "- Trained and evaluated only on the C-MAPSS simulation; real vehicle "
        "telemetry (E4's J1939 simulator) comes in later milestones.\n"
        "- RUL is capped at 125; healthy engines far from failure are "
        "under-predicted by construction (see ADR 0002).\n"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsets", nargs="*", default=ALL_SUBSETS)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--out-md", default="docs/model-card.md")
    ap.add_argument("--out-json", default="docs/evidence/seqmodel_results.json")
    args = ap.parse_args()

    cfg = TrainConfig(seed=args.seed, max_epochs=args.max_epochs)
    results = [evaluate_subset_cnn(s, args.raw, cfg=cfg) for s in args.subsets]
    base = _baseline_rmse()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    Path(args.out_md).write_text(model_card(results, base, cfg), encoding="utf-8")

    print(f"{'subset':7}{'CNN RMSE':>10}{'hist-GBR':>10}{'CNN PHM':>10}{'epochs':>8}")
    for r in results:
        s, m = r["subset"], r["metrics"]
        b = f"{base.get(s, float('nan')):.2f}"
        print(f"{s:7}{m['rmse']:>10.2f}{b:>10}{m['phm_score']:>10.0f}{m['epochs_run']:>8}")
    print(f"\nwrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
