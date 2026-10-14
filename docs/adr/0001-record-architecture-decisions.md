# 0001 - Record architecture decisions

Status: accepted

## Context

This project makes modelling choices (how RUL is labelled, how sensors are
normalised, which metrics are trusted) that affect every number it reports. A
reader should be able to see what was chosen and why, not reverse-engineer it
from code.

## Decision

Record each significant, non-obvious decision as a short ADR in `docs/adr/`,
numbered in order. An ADR states the context, the decision, and the
consequences. Routine or self-explanatory choices do not get an ADR.

## Consequences

Decisions are reviewable in isolation and travel with the code.
