#!/usr/bin/env python3
"""Build the clean Parquet data set + quality/lineage records from raw C-MAPSS.

Usage:
  python scripts/build_dataset.py [--raw data/raw] [--out data/processed] [--rul-cap 125]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsml.pipeline import build_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--rul-cap", type=int, default=125)
    args = ap.parse_args()

    recs = build_all(args.raw, args.out, rul_cap=args.rul_cap)
    print(f"{'subset':8} {'train rows':>10} {'test rows':>10} {'near-constant sensors dropped':>32}")
    for r in recs:
        nc = r["quality"]["train"]["near_constant_sensors"]
        print(f"{r['subset']:8} {r['rows']['train']:>10} {r['rows']['test']:>10} {len(nc):>10}   {','.join(nc) if nc else '-'}")
    print(f"\nwrote parquet + quality/lineage to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
