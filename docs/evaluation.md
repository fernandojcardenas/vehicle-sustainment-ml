# RUL baseline evaluation

Condition-based normalisation (per operating regime), capped-RUL labels (cap 125), grouped 5-fold CV by engine, one prediction per test engine scored against the official RUL truth. Test RMSE and the C-MAPSS/PHM08 asymmetric score (lower is better) are shown; the two trivial baselines bound the useful range.

## FD001

- Train rows: 20,631; features (kept sensors): 15
- Kept sensors: `sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21`

| Model | CV RMSE | Test RMSE | PHM score | PHM / engine |
|---|---|---|---|---|
| Predict mean RUL (trivial) | — | 43.07 | 33629 | 336.29 |
| Predict RUL cap (trivial) | — | 64.62 | 1502475 | 15024.75 |
| Ridge regression | 21.63 ± 1.38 | 21.67 | 1288 | 12.88 |
| Hist gradient boosting | 18.68 ± 0.57 | 17.93 | 916 | 9.16 |

## FD002

- Train rows: 53,759; features (kept sensors): 21
- Kept sensors: `sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_16, sensor_17, sensor_18, sensor_19, sensor_20, sensor_21`

| Model | CV RMSE | Test RMSE | PHM score | PHM / engine |
|---|---|---|---|---|
| Predict mean RUL (trivial) | — | 54.08 | 153229 | 591.62 |
| Predict RUL cap (trivial) | — | 69.37 | 5163063 | 19934.61 |
| Ridge regression | 21.84 ± 0.54 | 32.67 | 19171 | 74.02 |
| Hist gradient boosting | 19.66 ± 0.64 | 29.76 | 13903 | 53.68 |

## FD003

- Train rows: 24,720; features (kept sensors): 16
- Kept sensors: `sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21`

| Model | CV RMSE | Test RMSE | PHM score | PHM / engine |
|---|---|---|---|---|
| Predict mean RUL (trivial) | — | 45.07 | 57950 | 579.50 |
| Predict RUL cap (trivial) | — | 64.67 | 1390905 | 13909.05 |
| Ridge regression | 19.65 ± 0.81 | 22.29 | 1561 | 15.61 |
| Hist gradient boosting | 16.76 ± 0.94 | 21.54 | 2165 | 21.65 |

## FD004

- Train rows: 61,249; features (kept sensors): 21
- Kept sensors: `sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_16, sensor_17, sensor_18, sensor_19, sensor_20, sensor_21`

| Model | CV RMSE | Test RMSE | PHM score | PHM / engine |
|---|---|---|---|---|
| Predict mean RUL (trivial) | — | 54.90 | 189683 | 764.85 |
| Predict RUL cap (trivial) | — | 66.72 | 4030493 | 16251.99 |
| Ridge regression | 21.20 ± 0.69 | 34.53 | 12093 | 48.76 |
| Hist gradient boosting | 17.60 ± 0.97 | 30.32 | 8579 | 34.59 |

RMSE is in cycles. The PHM score is unbounded and dominated by the worst engines, so it is read alongside RMSE, never alone.
