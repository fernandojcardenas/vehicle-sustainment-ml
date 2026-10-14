# Model card -- Temporal CNN for RUL

A small 1D convolutional network over a sliding window of the last 30 cycles of per-regime-normalised sensors, predicting remaining useful life. Same data, normalisation and capped-RUL labels (cap 125) as the M1 baselines; one prediction per test engine (its last window) scored against the official RUL truth.

## Intended use

A portfolio demonstration of sequence modelling for predictive maintenance on an open benchmark; not a certified safety system.

## Architecture

- Input: window 30 x features (per-subset kept sensors)
- 3 x Conv1d(32) over time + global average pool + MLP head (dropout 0.2)
- Adam (lr 0.001), MSE loss, batch 256, up to 60 epochs, early stopping (patience 8) on a 20% engine-grouped validation split; seed 0

## Results (test RMSE, cycles)

| Subset | CNN RMSE | M1 hist-GBR RMSE | CNN PHM score | Val RMSE | Epochs |
|---|---|---|---|---|---|
| FD001 | 17.84 | 17.93 | 667 | 13.57 | 30 |
| FD002 | 28.32 | 29.76 | 11132 | 16.53 | 17 |
| FD003 | 19.15 | 21.54 | 959 | 19.12 | 56 |
| FD004 | 30.48 | 30.32 | 8378 | 20.31 | 25 |

The CNN is compared like-for-like against the strongest M1 baseline. The validation RMSE (on held-out training engines) is lower than the test RMSE because the test trajectories are truncated at varied points; this gap is expected on C-MAPSS and is reported rather than hidden.

## Limitations

- CPU-sized model, few epochs: competitive with the classical baseline but below heavily tuned deep models in the literature.
- Trained and evaluated only on the C-MAPSS simulation; real vehicle telemetry (E4's J1939 simulator) comes in later milestones.
- RUL is capped at 125; healthy engines far from failure are under-predicted by construction (see ADR 0002).
