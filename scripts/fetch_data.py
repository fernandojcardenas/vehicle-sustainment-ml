#!/usr/bin/env python3
"""Fetch the NASA C-MAPSS turbofan degradation data set, pinned by SHA-256.

The data are distributed by the NASA Prognostics Center of Excellence and
mirrored by the PHM Society. They are not committed to this repository; this
script downloads and verifies them into ``data/raw/``.

Source (mirror):
  https://data.phmsociety.org/nasa/  -> item 6, "Turbofan Engine Degradation
  Simulation Data Set". Terms: use at your own risk; acknowledge the repository
  and the data donors; cite Saxena et al. (2008). See THIRD-PARTY.md.

Usage:
  python scripts/fetch_data.py [--out data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

OUTER_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
OUTER_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_NAME = "CMAPSSData.zip"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"

EXPECTED_FILES = [
    *[f"{p}_FD00{i}.txt" for p in ("train", "test", "RUL") for i in range(1, 5)],
    "readme.txt",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify(name: str, got: str, want: str) -> None:
    if got != want:
        sys.exit(f"ERROR: {name} SHA-256 mismatch\n  got  {got}\n  want {want}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--url", default=OUTER_URL, help="override mirror URL")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if all((out / f).exists() for f in EXPECTED_FILES):
        print(f"data already present in {out}/ -- nothing to do")
        return 0

    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=120) as resp:  # noqa: S310 (fixed https URL)
        outer = resp.read()
    verify("outer zip", sha256_bytes(outer), OUTER_SHA256)
    print(f"  {len(outer):,} bytes, SHA-256 OK")

    with zipfile.ZipFile(io.BytesIO(outer)) as zouter:
        inner_path = next((n for n in zouter.namelist() if n.endswith(INNER_NAME)), None)
        if inner_path is None:
            sys.exit(f"ERROR: {INNER_NAME} not found inside the outer archive")
        inner = zouter.read(inner_path)
    verify("inner zip", sha256_bytes(inner), INNER_SHA256)
    print(f"  {INNER_NAME}: {len(inner):,} bytes, SHA-256 OK")

    with zipfile.ZipFile(io.BytesIO(inner)) as zinner:
        for n in zinner.namelist():
            base = Path(n).name
            if not base:
                continue
            (out / base).write_bytes(zinner.read(n))

    missing = [f for f in EXPECTED_FILES if not (out / f).exists()]
    if missing:
        sys.exit(f"ERROR: expected files missing after extract: {missing}")
    print(f"extracted {len(EXPECTED_FILES)} files into {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
