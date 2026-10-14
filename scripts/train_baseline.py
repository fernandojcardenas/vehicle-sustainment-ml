#!/usr/bin/env python3
"""Train and evaluate the classical RUL baselines on all four C-MAPSS subsets.

Writes a machine-readable results file (docs/evidence/baseline_results.json) and
a human-readable report (docs/evaluation.md), and prints a summary table.

Usage:
  python scripts/train_baseline.py [--raw data/raw] [--rul-cap 125] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.baseline import evaluate_subset  # noqa: E402
from vsml.columns import ALL_SUBSETS  # noqa: E402

MODEL_ORDER = ["predict_mean", "predict_cap", "ridge", "hist_gbr"]
MODEL_LABELS = {
    "predict_mean": "Predict mean RUL (trivial)",
    "predict_cap": "Predict RUL cap (trivial)",
    "ridge": "Ridge regression",
    "hist_gbr": "Hist gradient boosting",
}


def summarise(results: list[dict]) -> str:
    lines = []
    lines.append("# RUL baseline evaluation\n")
    lines.append(
        "Condition-based normalisation (per operating regime), capped-RUL labels "
        "(cap 125), grouped 5-fold CV by engine, one prediction per test engine "
        "scored against the official RUL truth. Test RMSE and the C-MAPSS/PHM08 "
        "asymmetric score (lower is better) are shown; the two trivial baselines "
        "bound the useful range.\n"
    )
    for r in results:
        s = r["subset"]
        lines.append(f"## {s}\n")
        lines.append(
            f"- Train rows: {r['n_train_rows']:,}; features (kept sensors): {r['n_features']}"
        )
        lines.append(f"- Kept sensors: `{', '.join(r['kept_sensors'])}`\n")
        lines.append("| Model | CV RMSE | Test RMSE | PHM score | PHM / engine |")
        lines.append("|---|---|---|---|---|")
        for m in MODEL_ORDER:
            rep = r["models"][m]
            cv = (
                f"{rep['cv_rmse_mean']:.2f} ± {rep['cv_rmse_std']:.2f}"
                if "cv_rmse_mean" in rep
                else "—"
            )
            lines.append(
                f"| {MODEL_LABELS[m]} | {cv} | {rep['rmse']:.2f} | "
                f"{rep['phm_score']:.0f} | {rep['phm_score_per_engine']:.2f} |"
            )
        lines.append("")
    lines.append(
        "RMSE is in cycles. The PHM score is unbounded and dominated by the "
        "worst engines, so it is read alongside RMSE, never alone.\n"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--rul-cap", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", default="docs/evaluation.md")
    ap.add_argument("--out-json", default="docs/evidence/baseline_results.json")
    args = ap.parse_args()

    results = [
        evaluate_subset(s, args.raw, rul_cap=args.rul_cap, random_state=args.seed)
        for s in ALL_SUBSETS
    ]

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(summarise(results), encoding="utf-8")

    # Console summary.
    print(f"{'subset':7} {'model':22} {'CV RMSE':>12} {'test RMSE':>10} {'PHM score':>10}")
    for r in results:
        for m in MODEL_ORDER:
            rep = r["models"][m]
            cv = f"{rep['cv_rmse_mean']:.2f}" if "cv_rmse_mean" in rep else "-"
            print(f"{r['subset']:7} {MODEL_LABELS[m]:22} {cv:>12} {rep['rmse']:>10.2f} {rep['phm_score']:>10.0f}")
    print(f"\nwrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
