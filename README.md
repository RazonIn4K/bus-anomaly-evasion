# Bus Anomaly Evasion

A small autoencoder that detects anomalies in synthetic CAN-bus-style telemetry,
and adversarial attacks that evade it.

**Thesis:** anomaly-detector evasion = minimize reconstruction error subject to an
attack-objective constraint. A detector trained only on "normal" traffic flags
what it can't reconstruct; a shaped attack that sits on the normal manifold slips
under the threshold.

> **Synthetic only.** Portfolio research artifact. Not a real vehicle bus, not a
> production IDS, no HIL claims.

## Results (seeds 0–2; see `results/report.md`)

| Story | Result |
| --- | --- |
| **(a) Injection mimicry** | Naive TPR ~70% → mimicry ~35% at fixed τ; objective retained ~84%; 0 constraint violations |
| **(b) Blind spot** | Replay / spoof naive TPR ~2–25% — stateless window features miss in-range on-schedule attacks |
| **(c) Flood / drop** | Not evadable under objective (recon ≫ τ); frame-space PGD does not change that |

Full per-family TPR @ τ and @ matched 1% FPR, PGD `n_true_pgd` counts, and FPR±std:
**`results/report.md`**.

## Reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m src.run_pipeline
pytest -q
```

## Layout

- `src/generate.py`  — synthetic normal + attack telemetry
- `src/features.py`  — windowing + feature extraction
- `src/model.py`     — tiny MLP autoencoder
- `src/train.py`     — train on normal only; calibrate threshold (p99 val)
- `src/evaluate.py`  — ROC-AUC, detection @ fixed FPR, matched 1% FPR τ
- `src/attack.py`    — mimicry (black-box) + **frame-space** PGD (white-box)
- `src/run_pipeline.py` — end-to-end seeds → `results/`
- `tests/`           — determinism, no-leakage, threshold calibration, evasion honesty
- `DESIGN.md`        — threat model and design decisions
- `AGENTS.md`        — agent conventions / non-negotiables

## Honesty rules

- τ from held-out **NORMAL** only (never attack/test).
- No never-worse-than-naive clamp on recon MSE.
- Thinning ≠ PGD (`true_pgd` is frame-space byte PGD with in-loop Π_C only).
- Report small / null / family-local gaps plainly.
