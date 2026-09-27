# Results Report — Bus Anomaly Evasion

> **Disclaimer (synthetic only):** All numbers below are from a synthetic
> CAN-style generator. They do **not** claim transfer to any real vehicle,
> OEM bus matrix, or production IDS. Portfolio research artifact only.

- Seeds: `[0, 1, 2]`
- Threshold rule: τ = p99 MSE on held-out NORMAL validation (never attack/test).
- τ calibration split: **`temporal_held_out_NORMAL_val`** (temporal early/mid/late contiguous windows).
- Same-stream split FPR (mean±std): **0.0366 ± 0.0238**
- Fresh-normal FPR (mean±std): **0.0430 ± 0.0489**
- Mean τ: `0.750857`

## Headline metrics (objective-gated; PGD = true_pgd path only)

Primary TPR / ΔTPR require `objective_retained`. Constrained PGD headline
further requires path=`true_pgd` (rate-thinning is **not** counted as PGD).
Ungated numbers are secondary only. Families with zero true_pgd windows
contribute `nan` to the PGD headline (not imputed as success).

| Method | TPR (gated) | ΔTPR vs naive | Mean recon MSE |
| --- | ---: | ---: | ---: |
| Naive (obj-gated) | 0.5908 | — | 246.051329 |
| Mimicry (obj-gated) | 0.5209 | 0.0699 | 221.993811 |
| Naive (true_pgd-matched) | 1.0000 | — | (matched windows only) |
| Constrained PGD (true_pgd ∩ obj) | 1.0000 | 0.0000 | 578.013986 |

**Headline ΔTPR (true_pgd-matched naive − true_pgd, objective-gated):** `0.0000`

### PGD path counts (mean windows / family / seed)

- thin: **25.07**
- true_pgd: **14.93**
- naive_fallback: **0.00**

- frac_improved (PGD all paths, recon < naive): **0.4900**
- frac_fallback (PGD all paths, recon ≥ naive; no clamp): **0.5100**

### Secondary (ungated)

- Mimicry ungated ΔTPR: `0.0733` (TPR `0.5133`)
- PGD ungated ΔTPR (all paths, may include thin): `-0.2400` (TPR `0.8267`)

## Per-family table (mean over seeds; primary gated)

| Family | TPR naive† | TPR mim | TPR true_pgd | ΔTPR PGD | thin/true_pgd/fb | frac_imp | frac_fb | Obj ret PGD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| injection | nan | 0.3546 | nan | nan | 40.0/0.0/0.0 | 0.8917 | 0.1083 | 0.708 |
| spoof | nan | 0.2417 | nan | nan | 40.0/0.0/0.0 | 0.0000 | 1.0000 | 0.958 |
| flood | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3.3/36.7/0.0 | 0.8917 | 0.1083 | 1.000 |
| drop | 1.0000 | 0.9833 | 1.0000 | 0.0000 | 2.0/38.0/0.0 | 0.6667 | 0.3333 | 0.950 |
| replay | nan | 0.0250 | nan | nan | 40.0/0.0/0.0 | 0.0000 | 1.0000 | 1.000 |

† Naive TPR for PGD Δ is matched to true_pgd ∩ objective_retained windows.

## Secondary notes

- Attack-objective retention and constraint-violation rates are reported per family.
- Two-pass verification: feature-space step → frame morph → re-extract → score.
- Never-worse-than-naive clamp removed: recon MSE and frac_fallback are actual.
- In-loop Π_C is a soft feature-space clip before frame morph; hard feasibility is
  enforced by byte quantization (0–255) and two-pass re-extract — residual gap vs a
  true projective Π_C on the discrete frame set remains (P2 note).
- Evasion is expected to lower mean reconstruction error vs naive when the morph
  works; ΔTPR may be small or family-local.

## Honest gaps

- Gated true_pgd ΔTPR is small or undefined on some families. Acceptable portfolio outcome: we do not inflate PGD by counting rate-thinning as white-box success.
- Mean reconstruction error under true_pgd did not clearly beat naive after two-pass; reported honestly (no clamp).
- Mimicry is the black-box baseline; true PGD is the white-box primary demo.
- No real captures; no HIL; no production claims.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.run_pipeline
pytest -q
```
