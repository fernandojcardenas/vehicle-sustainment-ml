"""Turn the raw C-MAPSS text files into a clean, checked Parquet data set.

For each subset this writes ``train_<subset>.parquet`` (with the ``rul`` label)
and ``test_<subset>.parquet`` (the full test trajectories), plus a per-subset
quality report and a lineage record capturing the source file SHA-256s, the RUL
cap and the row/engine counts. Parquet is columnar and typed, so downstream
code loads it without re-parsing whitespace or re-guessing dtypes.
"""

from __future__ import annotations

from pathlib import Path

from .columns import ALL_SUBSETS
from .data import DEFAULT_RUL_CAP, add_training_rul, read_raw_table
from .quality import check_table, lineage, write_json


def build_subset(
    subset: str,
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    rul_cap: int | None = DEFAULT_RUL_CAP,
) -> dict:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    (out_dir / "parquet").mkdir(parents=True, exist_ok=True)
    (out_dir / "quality").mkdir(parents=True, exist_ok=True)

    train_src = raw_dir / f"train_{subset}.txt"
    test_src = raw_dir / f"test_{subset}.txt"
    rul_src = raw_dir / f"RUL_{subset}.txt"

    train = read_raw_table(train_src)
    test = read_raw_table(test_src)

    train_rep = check_table(train, subset, "train")
    test_rep = check_table(test, subset, "test")

    train = add_training_rul(train, cap=rul_cap)

    train.to_parquet(out_dir / "parquet" / f"train_{subset}.parquet", index=False)
    test.to_parquet(out_dir / "parquet" / f"test_{subset}.parquet", index=False)

    rec = lineage(
        subset=subset,
        source_files={"train": train_src, "test": test_src, "rul": rul_src},
        rul_cap=rul_cap,
        extra={
            "quality": {"train": train_rep.to_dict(), "test": test_rep.to_dict()},
            "rows": {"train": int(len(train)), "test": int(len(test))},
        },
    )
    write_json(rec, out_dir / "quality" / f"lineage_{subset}.json")
    return rec


def build_all(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    rul_cap: int | None = DEFAULT_RUL_CAP,
) -> list[dict]:
    return [build_subset(s, raw_dir, out_dir, rul_cap=rul_cap) for s in ALL_SUBSETS]
