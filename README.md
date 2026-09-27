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
calibration only** (never attack/test; never retuned on attacks). Calibration:
ordered independent NORMAL blocks (12×200s: 6 train / 1 earlystop / 3 calib→τ=p99
/ 2 test). Fresh FPR stream seed = `seed+777`.

### Gate status (honest)

| Gate | Status |
| --- | --- |
| Fresh-NORMAL FPR @ τ | **PASS** — per seed `[0.0068, 0.0118, 0.0182]`, mean±std **0.0123 ± 0.0047** |
| Injection mimicry all-seed headline | **FAIL** — seed2 Δ=0 (honest null); mean Δ still positive |
| **GATES_PASS (win claim)** | **False** — FPR PASS + inj FAIL; no all-seed evasion win claimed |

Matched 1% FPR is **secondary/diagnostic only**.

### Headlines

**(a) Injection mimicry @ τ** — per-seed (naive / mim / Δ / obj retention):

| Seed | Naive TPR | Mimicry TPR | Δ | Obj retention |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.0000 | 0.8286 | 0.1714 | 0.875 |
| 1 | 1.0000 | 0.6786 | 0.3214 | 0.700 |
| 2 | 1.0000 | 1.0000 | **0.0000** | 0.700 |

Mean naive → mimicry: **1.0000 → 0.8357** (Δ **0.1643**); mean obj retention **0.758**; constraint violations **0**. Seed2 Δ=0 is reported as an honest null — no further morph search / no τ retune.

**(b) Replay / spoof blind spot** — replay naive TPR @ τ **0.0167**, spoof **0.0000** (mimicry identical). Stateless window features miss in-range on-schedule payload attacks (detector limitation, not an evasion win).

**(c) Flood / drop not evadable under objective** — naive recon ≫ τ (~618 flood / ~714 drop vs τ ~0.53). Frame-space PGD cannot close that gap while retaining the objective (`true_pgd` TPR still **1.0**, Δ **0**, `n_true_pgd`=40/seed). Say **not evadable**, not “PGD failed.”

**PGD** is reported per-family with `n_true_pgd`; zero / null Δ is honest (thinning ≠ PGD; no never-worse clamp). Full tables: `results/report.md` + `results/aggregate.json`.

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

- τ from held-out **NORMAL** only (never attack/test; never attack-tuned).
- Fresh-NORMAL FPR uses a **different** seed/stream than train/calib.
- No never-worse-than-naive clamp on recon MSE.
- Thinning ≠ PGD (`true_pgd` = frame-space byte PGD with in-loop Π_C only).
- Report small / null / family-local gaps plainly (including seed2 Δ=0). Macro Δ hides family structure.
