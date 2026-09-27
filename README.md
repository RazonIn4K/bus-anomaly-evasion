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

Primary operating point: **τ = p99 reconstruction MSE on held-out NORMAL
calibration only** (never attack/test; never retuned on attacks).

| Story | What to look for |
| --- | --- |
| **(a) Injection mimicry @ τ** | Gated naive→mimicry ΔTPR at fixed NORMAL-derived τ, with objective retention and 0 constraint violations. Headline only if predeclared gates pass (naive≥20%/seed, mean Δ≥10pp, Δ>0 all seeds); otherwise report actual rates honestly. |
| **(b) Blind spot** | Replay / spoof low naive TPR — stateless window features miss in-range on-schedule attacks (detector limitation, not an evasion win). |
| **(c) Flood / drop** | Not evadable under objective (recon ≫ τ). Frame-space PGD does not change that — say **not evadable**, not “PGD failed.” |

**PGD** is reported per-family with `n_true_pgd` window counts; zero / null Δ is an
honest outcome (thinning ≠ PGD; no never-worse clamp). Matched 1% FPR ROC point is
**secondary/diagnostic only**.

Numbers: **`results/report.md`** and **`results/aggregate.json`** (regenerate with the
pipeline — do not trust stale copy-paste).

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
- `src/train.py`     — NORMAL-only train; τ = p99 held-out NORMAL calib
- `src/evaluate.py`  — ROC-AUC, detection @ fixed FPR, matched 1% FPR (diagnostic)
- `src/attack.py`    — mimicry (black-box) + **frame-space** PGD (white-box, in-loop Π_C)
- `src/run_pipeline.py` — end-to-end seeds → `results/`
- `tests/`           — determinism, no-leakage, free-byte freezes, byte-stat identity
- `DESIGN.md`        — threat model and design decisions
- `AGENTS.md`        — agent conventions / non-negotiables

## Honesty rules

- τ from held-out **NORMAL** only (never attack/test).
- Fresh-NORMAL FPR uses a **different** seed/stream than train/calib.
- No never-worse-than-naive clamp on recon MSE.
- Thinning ≠ PGD (`true_pgd` = frame-space byte PGD with in-loop Π_C only).
- Report small / null / family-local gaps plainly. Macro Δ hides family structure.
