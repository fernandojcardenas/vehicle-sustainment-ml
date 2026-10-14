# 0003 - Condition-based (per-regime) sensor normalisation

Status: accepted

## Context

FD002 and FD004 are recorded under six operating conditions. The raw sensor
values jump between those regimes, and that between-regime variation is far
larger than the slow drift caused by degradation. A model trained on raw values
spends its capacity separating regimes instead of detecting wear.

## Decision

Cluster the three operational-setting columns into `n_conditions` regimes with
k-means (fit on the training engines, `random_state` fixed), then z-normalise
each sensor **within its regime** using per-regime mean/standard-deviation
learned on training data only. Under a single condition (FD001/FD003) this is
ordinary z-scoring. Sensors that are near-constant within a subset carry no
information and are dropped, with the drop recorded in the quality report.

## Consequences

- Degradation becomes the dominant remaining signal on FD002/FD004.
- No test statistics leak into normalisation (fit on train only).
- The regime model and the per-regime statistics are part of the fitted
  pipeline and are applied identically to train and test.
