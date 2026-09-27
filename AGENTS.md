# AGENTS.md

Guidance for AI coding agents (Claude Code primary, Codex CLI review).

## What this is
A portfolio project demonstrating adversarial ML against an anomaly detector,
framed in the CAN-bus / satellite-telemetry domain. Correctness and honesty of
the result matter more than feature count. This is synthetic-data research only.

## Non-negotiables
- Reproducible: fix all seeds; a clean clone must regenerate every result.
- No leakage: the autoencoder trains on NORMAL windows only. The detection
  threshold is calibrated on a held-out NORMAL split — never on test or attack
  data.
- The evasion in `attack.py` must actually reduce reconstruction error relative
  to the naive attack, and must respect bus constraints (bytes 0-255, counts >= 0,
  feasible inter-arrival timing). An attack that produces infeasible frames is a
  bug, not a result.
- The headline metric is the detection-rate delta: naive vs. evasion-optimized
  attacks at a fixed FPR. Report it plainly, including if the gap is small.

## Conventions
- Python 3.11, torch (CPU is fine — the model is tiny), numpy, scikit-learn,
  matplotlib, pytest.
- Small, reviewable commits, one working piece at a time.
- Save artifacts to `results/`; save models to the repo root or `results/`
  (git-ignored).

## Build order
generate -> features -> model -> train (+threshold) -> evaluate -> attack -> report
