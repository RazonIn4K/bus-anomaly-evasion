# Bus Anomaly Evasion

A small autoencoder that detects anomalies in synthetic CAN-bus-style telemetry,
and adversarial attacks that evade it.

**Thesis:** anomaly-detector evasion = minimize reconstruction error subject to an
attack-objective constraint. A detector trained only on "normal" traffic flags
what it can't reconstruct; a shaped attack that sits on the normal manifold slips
under the threshold.

## Result to reproduce

Detection rate on naive attacks vs. evasion-optimized attacks at a fixed false-
positive rate. The gap between the two is the point.

## Layout

- `src/generate.py`  — synthetic normal + attack telemetry
- `src/features.py`  — windowing + feature extraction
- `src/model.py`     — tiny MLP autoencoder
- `src/train.py`     — train on normal only; calibrate threshold
- `src/evaluate.py`  — ROC-AUC, detection @ fixed FPR
- `src/attack.py`    — mimicry (black-box) and PGD (white-box) evasion
- `tests/`           — determinism, no-leakage, threshold calibration
- `DESIGN.md`        — threat model and design decisions (Phase 1 output)

## Setup

    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Disclaimer

Synthetic data only. Built as a research/portfolio exercise in adversarial ML
against anomaly detectors. Not affiliated with any employer or product.
