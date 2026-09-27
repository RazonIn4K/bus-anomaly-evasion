# Results Report — Bus Anomaly Evasion

> **Disclaimer (synthetic only):** All numbers below are from a synthetic
> CAN-style generator. They do **not** claim transfer to any real vehicle,
> OEM bus matrix, or production IDS. Portfolio research artifact only.

- Seeds: `[0, 1, 2]`
- Threshold rule: τ = p99 MSE on held-out NORMAL validation (never attack/test).
- τ calibration split: **`ordered_independent_NORMAL_calib_blocks`** (ordered independent NORMAL calibration blocks; p99 primary).
- Mean val windows / seed: **13699.0** (effective indep ≈ **6849.7**)
- Same-stream split FPR (mean±std): **0.0078 ± 0.0044**
- Fresh-normal FPR @ τ per seed: `[0.006787825706152836, 0.011823954455879132, 0.018173855922925333]`
- Fresh-normal FPR @ τ (mean±std): **0.0123 ± 0.0047** (gates: std<1pp=True, mean≤1.5%=True, max≤2.5%=True)
- **gate_fpr_all_ok=True** · **gate_injection_headline_ok=False** · **GATES_PASS (win claim)=False**
- Mean τ (p99 held-out NORMAL val) — **primary**: `0.527456`
- Secondary τ @ matched 1% FPR (fresh-NORMAL ROC; diagnostic): `0.533900`
- Fresh-NORMAL windows / seed (mean): **4567.0** (effective indep ≈ **2284.0**)

## Headlines (read these first)

### (0) Key finding — structure over payload

> **Key finding:** The autoencoder's decisions are driven by message structure (rates, timing, ID mix), not payload content. Consequences: payload-only attacks (spoof, replay) are invisible to it; white-box PGD on payload bytes lowers reconstruction error in 99% of windows but can't cross τ; only rate-shaping mimicry evades, cutting injection detection from 100% to 84% on average (seed 2: no reduction). Flood/drop dominate the structural features and can't be evaded while keeping their objective.

Numbers from `results/aggregate.json` (unchanged metrics): injection mean recon naive **1.52** / mimicry **0.79** / PGD **1.38** vs τ≈**0.53**; injection PGD `frac_improved` **0.99**; spoof/replay naive recon ≈**0.23** (< τ). Gate status below remains **GATES_PASS=False** (all-seed inj FAIL; FPR PASS).

### (a) Injection mimicry @ τ (primary = held-out NORMAL p99)

- Gate headline_ok: **False** (naive≥20%/seed, mean Δ≥10pp, Δ>0 all seeds)
- Per-seed naive TPR: `[1.0, 1.0, 1.0]`
- Per-seed ΔTPR (naive−mim): `[0.17142857142857137, 0.3214285714285714, 0.0]`
- Mean naive → mimicry @ τ: **1.0000** → **0.8357** (Δ **0.1643**)
- Objective retention (mimicry): **0.758**; constraint violations: **0.000**
- **Honest null / no all-seed win claim:** all-seed injection mimicry headline gate FAILED (see per-seed Δ; zeros allowed). FPR gate status above. Reporting actual rates — not a forced win.
- Secondary/diagnostic matched 1% FPR (not primary): naive **1.0000** → mimicry **0.8619** (Δ **0.1381**).

### (b) Detector blind spot: replay / spoof

Stateless window features miss in-range, on-schedule payload attacks.
- Replay naive TPR @ τ: **0.0167** (mimicry **0.0167**)
- Spoof naive TPR @ τ: **0.0000** (mimicry **0.0000**)
Low TPR here is a detector limitation, not an evasion win.

### (c) Flood / drop not evadable under the objective

Rate attacks dominate reconstruction via count/IAT features. Typical naive
recon MSE is ~618.5 (flood) / ~713.8 (drop) vs τ ~ 0.53. Frame-space byte PGD cannot close that gap while retaining the objective — say **not evadable under objective**, not “PGD failed.”
- Flood true_pgd TPR @ τ: **1.0000** (Δ **0.0000**, n_true_pgd=40.0)
- Drop true_pgd TPR @ τ: **1.0000** (Δ **0.0000**, n_true_pgd=40.0)

## Per-family table (mean over seeds)

| Family | TPR naive @τ | TPR mim @τ | Δ mim | TPR true_pgd @τ | Δ PGD | n_true_pgd | TPR mim @1%FPR | Obj ret mim | Obj ret PGD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| injection | 1.0000 | 0.8357 | 0.1643 | 1.0000 | 0.0000 | 40.0 | 0.8619 | 0.758 | 0.975 |
| spoof | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 40.0 | 0.0000 | 0.958 | 0.975 |
| flood | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 40.0 | 1.0000 | 1.000 | 1.000 |
| drop | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 40.0 | 1.0000 | 0.950 | 0.950 |
| replay | 0.0167 | 0.0167 | 0.0000 | 0.0167 | 0.0000 | 40.0 | 0.0167 | 1.000 | 1.000 |

PGD ΔTPR is `nan` when a family has zero `true_pgd` windows (not imputed).
n_true_pgd is mean windows / seed labeled frame-space PGD.

## Frame-space PGD notes (Π_C matches code)

- PGD runs on **free payload bytes** of frames in each window (objective-critical
  bytes frozen: spoof phys b0/b1, full replay payload).
- Differentiable torch path: byte mean/std per TOP_IDS ID; counts / IAT /
  entropy stay fixed from the window structure.
- **Π_C in-loop:** after every gradient step, free bytes are projected onto
  `[0, 255] ∩ L∞(x₀, ε)` relative to the original window bytes.
- After optimization: quantize to uint8, **re-extract with the original numpy
  extractor**, score AE — two-pass verify. No feature-space-then-morph path.
- Thinning / blending is **mimicry only**; never labeled `true_pgd`.
- No never-worse-than-naive clamp; `frac_fallback` is honest.
- Macro frac_improved (PGD): **0.9800**; frac_fallback: **0.0200**

## Honest gaps

- Macro ΔTPR across families hides the partial, seed-dependent injection
  mimicry reduction and the flood/drop non-evasion — read the headlines and
  per-family table.
- Fresh-normal FPR std target is < 1% after enlarging val / fresh streams;
  residual deviation is reported above, not clamped.
- Synthetic only: no real captures, no HIL, no production claims.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.run_pipeline
pytest -q
```
