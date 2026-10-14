# 0004 - Sequence model: windowed 1D-CNN with capped-RUL labels

Status: accepted

## Context

The classical baselines (M1) see one cycle at a time. Degradation is a trend
over many cycles, so a model that sees a window of recent history should do
better, especially on the asymmetric PHM score where confidently-late
predictions are expensive.

## Decision

Use a fixed-length **window of the last 30 cycles** of the same per-regime
-normalised sensors (ADR 0003) and the same capped-RUL labels (ADR 0002), and a
small **1D-CNN over time** (three Conv1d blocks, global average pool, MLP head).
Windows are left-padded with the engine's first cycle so every cycle yields a
window and short test engines are handled like training ones. Training is seeded
and deterministic, with an engine-grouped validation split and early stopping;
test uses one prediction per engine (its last window), scored against the RUL
truth exactly as the baselines are.

## Consequences

- Like-for-like comparison with M1: only the model changes.
- The CNN beats the strongest classical baseline (hist-GBR) on RMSE on three of
  four subsets and improves the PHM score on all four; FD004 (six conditions,
  two faults) is roughly tied.
- The model is deliberately CPU-sized (reproducible in CI); heavier tuning is a
  later option, not this milestone.
- Full training numbers come from a fixed seed; CI trains a short FD001 model to
  check it runs and beats the trivial baseline (NN results are reproducible with
  a fixed seed on CPU but not asserted bit-for-bit across machines).
