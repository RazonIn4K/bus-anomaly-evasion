# Results Report — Bus Anomaly Evasion

> **Disclaimer (synthetic only):** All numbers below are from a synthetic
> CAN-style generator. They do **not** claim transfer to any real vehicle,
> OEM bus matrix, or production IDS. Portfolio research artifact only.

- Seeds: `[0, 1, 2]`
- Threshold rule: τ = p99 MSE on held-out NORMAL validation (never attack/test).
- Empirical FPR on NORMAL test split (mean±std): **0.0129 ± 0.0053**
- Mean τ: `0.592506`

## Headline metrics (macro-average over attack families)

| Method | TPR | ΔTPR vs naive | Mean recon MSE |
| --- | ---: | ---: | ---: |
| Naive | 0.6517 | — | 257.080952 |
| Mimicry | 0.5683 | 0.0833 | 232.164986 |
| Constrained PGD | 0.5317 | 0.1200 | 234.401373 |

**Headline ΔTPR (naive − PGD):** `0.1200`

## Per-family table (mean over seeds)

| Family | TPR naive | TPR mimicry | TPR PGD | ΔTPR PGD | Recon naive | Recon PGD | Obj ret PGD | Viol PGD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| injection | 0.9333 | 0.5083 | 0.3333 | 0.6000 | 2.401892 | 0.551261 | 0.708 | 0.000 |
| spoof | 0.3083 | 0.3083 | 0.3083 | 0.0000 | 0.516791 | 0.516791 | 0.975 | 0.000 |
| flood | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 595.747816 | 487.482061 | 1.000 | 0.000 |
| drop | 0.9833 | 0.9917 | 0.9833 | 0.0000 | 686.353439 | 683.071927 | 0.950 | 0.000 |
| replay | 0.0333 | 0.0333 | 0.0333 | 0.0000 | 0.384824 | 0.384824 | 1.000 | 0.000 |

## Secondary notes

- Attack-objective retention and constraint-violation rates are reported per family.
- Two-pass verification: feature-space step → frame morph → re-extract → score.
- Evasion is expected to lower mean reconstruction error vs naive; ΔTPR may be small.

## Honest gaps

- Constrained PGD reduces detection relative to naive at fixed τ; magnitude should be read alongside objective-retention (evasion that undoes the attack is not counted as success).
- Mean reconstruction error under PGD is lower than naive — evasion directionally correct.
- Mimicry is the black-box baseline; PGD is the white-box primary demo.
- No real captures; no HIL; no production claims.

## Reproduce

```bash
source .venv/bin/activate
PYTHONPATH=. python -m src.run_pipeline
pytest -q
```
