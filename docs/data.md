# Data: NASA C-MAPSS turbofan degradation

The C-MAPSS data set simulates a fleet of turbofan engines run to failure. Each
of the four sub-sets is split into training trajectories (run all the way to
failure) and test trajectories (truncated some time before failure), with a
separate file of the true remaining useful life (RUL) at each test engine's
last recorded cycle.

| Subset | Operating conditions | Fault modes | Train engines | Test engines |
|---|---|---|---|---|
| FD001 | 1 (sea level) | 1 (HPC degradation) | 100 | 100 |
| FD002 | 6 | 1 (HPC degradation) | 260 | 259 |
| FD003 | 1 (sea level) | 2 (HPC + fan) | 100 | 100 |
| FD004 | 6 | 2 (HPC + fan) | 249 | 248 |

Engine counts are the **verified** counts from the files. The archive's
`readme.txt` lists FD004 as 248 train / 249 test; the files actually contain
249 train / 248 test (test always matches the RUL-truth length). The pipeline
checks `test engines == RUL rows` as a hard gate.

## Columns

Each row is one operational cycle, 26 space-separated columns:

1. `unit` - engine id
2. `cycle` - time in cycles (starts at 1, increments by 1)
3-5. `op_setting_1..3` - operational settings (define the regime)
6-26. `sensor_1..21` - sensor measurements (with noise)

## Pipeline

`scripts/fetch_data.py` downloads and SHA-256-verifies the archive into
`data/raw/` (never committed). `scripts/build_dataset.py` writes typed Parquet
plus a per-subset quality report and lineage record (source file hashes, RUL
cap, row/engine counts) into `data/processed/`.

Quality gates: correct columns and dtypes, no missing values, per-engine cycles
that start at 1 and increase by 1, the verified engine counts, finite sensors,
and a report of near-constant (uninformative) sensors. Several sensors are flat
under a single operating condition (FD001 drops sensor 1, 5, 10, 16, 18, 19);
under six conditions all sensors vary.
