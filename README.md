# vehicle-sustainment-ml

Predictive maintenance for a fleet of engines: predict the **remaining useful
life (RUL)** of each unit from its sensor history, end to end — a checked data
pipeline, honest baselines, and an evaluation that scores against held-out
truth with the same asymmetric metric the field uses.

Milestones 1–2 are in: a checked data pipeline with classical baselines, and a
sequence model (1D-CNN) that beats them. Later milestones add synthetic J1939
vehicle telemetry, an ONNX edge-serving path on SocketCAN with a drift monitor,
and an infrastructure-as-code AWS slice — see [Roadmap](#roadmap).

## What it does

- Fetches the NASA **C-MAPSS** turbofan degradation data set (FD001–FD004),
  pinned by SHA-256, and never commits it.
- Builds a typed **Parquet** data set with explicit **quality gates** (schema,
  dtypes, no missing values, per-engine cycle continuity, verified engine
  counts, near-constant-sensor detection) and a **lineage** record (source
  hashes, RUL cap, row/engine counts) for every subset.
- Trains classical RUL baselines with **condition-based normalisation** and
  **grouped cross-validation by engine** (no engine leaks between folds).
- Evaluates one prediction per test engine against the official RUL truth with
  **RMSE** and the **C-MAPSS/PHM08 asymmetric score**, and bounds the useful
  range with two trivial baselines.

## Results

Test RMSE (cycles) at the last observed cycle of each test engine, capped-RUL
labels (cap 125), grouped 5-fold CV by engine. Full tables, the PHM score and
the kept-sensor lists are in [docs/evaluation.md](docs/evaluation.md).

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Predict mean RUL (trivial) | 43.07 | 54.08 | 45.07 | 54.90 |
| Ridge regression | 21.67 | 32.67 | 22.29 | 34.53 |
| Hist gradient boosting (M1) | 17.93 | 29.76 | 21.54 | 30.32 |
| Temporal CNN (M2) | **17.84** | **28.32** | **19.15** | 30.48 |

The classical baselines beat both trivial predictors everywhere. The **M2
sequence model** (a windowed 1D-CNN, [docs/model-card.md](docs/model-card.md))
improves on the strongest baseline's RMSE on three of four subsets and improves
the asymmetric PHM score on all four — FD004 (six conditions, two faults) is
roughly tied. The six-condition subsets are harder than the single-condition
ones, as expected. Numbers come from fixed-seed runs; CI retrains and checks
them.

## Quickstart

```bash
pip install -r requirements-dev.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu  # M2 only
python scripts/fetch_data.py        # download + verify C-MAPSS into data/raw/
python scripts/build_dataset.py     # -> data/processed/ (Parquet + quality + lineage)
python scripts/train_baseline.py    # M1 -> docs/evaluation.md + docs/evidence/
python scripts/train_seqmodel.py    # M2 -> docs/model-card.md + docs/evidence/
python -m pytest -q                 # tests (data-dependent ones need the fetch step)
```

Or `make all`.

## Layout

```
src/vsml/        columns, data loading + RUL labels, quality gates,
                 feature pipeline (regimes + per-regime normalisation),
                 scoring (RMSE + PHM), baselines,
                 windows + 1D-CNN sequence model (seqmodel, seqeval)
scripts/         fetch_data, build_dataset, train_baseline, train_seqmodel
tests/           scoring/data/quality/windowing units + data-dependent tests
docs/            data.md, evaluation.md, model-card.md, adr/, evidence/
```

## Data and modelling choices

- Data set, columns, verified engine counts and the pipeline:
  [docs/data.md](docs/data.md).
- Why the RUL label is capped:
  [ADR 0002](docs/adr/0002-piecewise-linear-rul-cap.md).
- Why sensors are normalised per operating regime:
  [ADR 0003](docs/adr/0003-condition-based-normalisation.md).
- The sequence model (window + 1D-CNN):
  [ADR 0004](docs/adr/0004-sequence-model.md) and [model card](docs/model-card.md).

The data is NASA C-MAPSS, used under the PCoE/PHM repository terms (acknowledge
and cite Saxena et al., 2008); see [THIRD-PARTY.md](THIRD-PARTY.md). It is
fetched, verified and git-ignored, never redistributed here.

## Roadmap

- **M1 — Data pipeline + classical baselines.** ✅ Done.
- **M2 — Sequence model (1D-CNN) for RUL**, compared on the same splits;
  [model card](docs/model-card.md). ✅ Done.
- **M3 — Synthetic J1939 vehicle telemetry** with injected faults, labelled.
- **M4 — Edge serving:** ONNX in a C++ service on SocketCAN, drift monitor, CI
  gate on metric regressions.
- **M5 — CDAO-style T&E report** (data integrity, bias, resilience, drift).
- **M6 — AWS slice (optional):** SageMaker script-mode training, model
  registry, endpoint and CloudWatch as Terraform, validated with
  `terraform plan` only (no paid run unless explicitly chosen).

## Why

I'm building toward an M.S. in Computer Science and work in ground-vehicle
sustainment. This project shows an end-to-end predictive-maintenance system —
data engineering with real quality and lineage discipline, honest evaluation,
and a path to an edge-deployed model — on an open benchmark before it moves to
vehicle-network telemetry.

## License

MIT (own code); see [LICENSE](LICENSE). Data is third-party; see
[THIRD-PARTY.md](THIRD-PARTY.md).
