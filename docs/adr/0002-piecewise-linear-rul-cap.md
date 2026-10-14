# 0002 - Cap the training RUL label (piecewise-linear RUL)

Status: accepted

## Context

A run-to-failure training trajectory gives a natural RUL label at each cycle:
`max_cycle_for_engine - cycle`, counting linearly down to 0 at failure. But
early in an engine's life there is no observable degradation -- the sensors
look like a healthy engine's -- so asking a model to predict a large RUL there
forces it to fit a target the inputs cannot support, and it is penalised for
being "wrong" on cycles where no method could be right.

## Decision

Cap the training RUL label at **125 cycles** (a piecewise-linear RUL: constant
at 125 while true RUL exceeds 125, then linear to 0). 125 is the value used in
the standard C-MAPSS literature (Heimes, 2008 and after). Test predictions are
clipped to `[0, 125]` for the same reason. The cap is a single parameter
(`DEFAULT_RUL_CAP`) and can be changed or disabled.

## Consequences

- The model is trained only where the inputs carry a degradation signal.
- Engines whose true test RUL exceeds 125 are under-predicted by construction;
  these are healthy engines far from failure, where exact RUL matters least.
- Reported numbers are comparable to the piecewise-linear-RUL literature; they
  are not comparable to studies that use an uncapped linear RUL.
