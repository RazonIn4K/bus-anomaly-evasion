# DESIGN.md — Bus Anomaly Evasion (Phase 1)

**Project:** Synthetic CAN-bus-style telemetry · reconstruction-error IDS · constrained evasion  
**Scope:** Portfolio research only. No real vehicle bus, no HIL, no production claims.  
**Status:** Phase 1 design locked for Phase 2 implementation.

---

## Thesis (one line)

**Evasion = minimize reconstruction error subject to an attack-objective constraint**, under bus-feasible projections.

---

## 1. Threat model

### Assets (what we protect, in this sandbox)

| Asset | Notes |
| --- | --- |
| Synthetic CAN-style frame stream | Timestamp, 11-bit ID, ≤8-byte payload |
| Window feature vectors | Aggregates over fixed windows (see §7) |
| MLP autoencoder + threshold | Trained on NORMAL only; threshold from held-out NORMAL |

### Adversary position

- **Already on-bus** (post-compromise node). How the node was compromised is out of scope.
- Can inject, suppress, or replace frames within synthetic generator rules.
- Goal: achieve an attack objective (spoof / flood / drop / replay / injection) while staying under the IDS threshold.

### Black-box mimicry (secondary method)

| Capability | Detail |
| --- | --- |
| Knowledge | Feature schema + approximate normal feature statistics (empirical moments / histograms from NORMAL train). No gradients. Optional: binary above/below-threshold oracle only. |
| Action | Morph attack windows so feature stats approach the NORMAL profile while preserving attack objective (e.g. injected rate ≥ target, spoofed signal moves toward target, flood intensity, drop duration, replay payload identity). |
| Inspiration | Classical mimicry / blending against anomaly detectors ([4], [5]). |

### White-box PGD (primary method)

| Capability | Detail |
| --- | --- |
| Knowledge | Full AE weights, loss (MSE reconstruction), and fixed threshold τ. |
| Action | Iterative projected gradient descent on features: minimize reconstruction error (or push score below τ) **subject to** attack-objective constraint + feasibility set \(\mathcal{C}\). |
| Projection \(\Pi_{\mathcal{C}}\) | Bytes ∈ {0…255}; counts ∈ ℕ₀; inter-arrival / rates physically feasible (IAT > 0, rate ≤ bus capacity proxy); after each step re-quantize and re-extract features through the **same** pipeline as inference. |

### In scope

- Synthetic generator: injection, spoof (incl. gradual change), flood, drop/suppress, replay.
- Feature-space mimicry and constrained PGD.
- Honest gap reporting: naive attack TPR vs evasion-optimized TPR at fixed FPR.

### Out of scope

- Real vehicle capture, ECU reverse-engineering, HIL, or claims of transfer to production IDS.
- Cryptographic CAN (SecOC), physical-layer fingerprinting, bus arbitration exploits as the research object.
- Training-time poisoning.
- Claiming passenger-safety impact from synthetic experiments.

---

## 2. Synthetic-data realism risks

Synthetic CAN can support **method** research (AE + threshold discipline + constrained evasion). It **cannot** support claims about a specific vehicle, OEM matrix, or deployed IDS.

### What synthetic CAN can claim

- Relative behavior of reconstruction-error detectors under controlled attack families.
- Whether constrained optimization reduces detection rate at a fixed NORMAL-derived FPR.
- Engineering of projection / feasibility constraints in feature space.

### What it cannot claim

- “Bypasses automotive IDS in the wild.”
- Coverage of real ECU correlations, sleep/wake modes, diagnostic sessions, or OEM-specific encodings.
- Absolute security of any real bus.

### Failure modes that would fake a win (must guard against)

| Failure mode | Why it fakes a win | Mitigation |
| --- | --- | --- |
| Over-clean NORMAL | AE learns a tiny manifold; naive attacks look “hard,” evasion looks “easy” | Inject realistic jitter, multi-mode driving regimes, mild missingness; report NORMAL reconstruction distribution |
| Circular evaluation | Attack generator shares hidden state with detector features | Generator emits discrete frames → **same** feature extractor as IDS; no gradient shortcuts past frame assembly |
| Unconstrained PGD | Floating features not invertible to frames | Mandatory \(\Pi_{\mathcal{C}}\) + two-pass verify: optimize → frames → re-extract → score |
| Threshold leakage | τ fitted on attack/test | τ = p99 of reconstruction error on **held-out NORMAL only**; never retuned on attacks |
| Objective collapse | Evasion succeeds by undoing the attack (signal returns to baseline) | Report **attack-objective retention** rate alongside TPR drop |
| Scope creep in writeups | Portfolio text implies vehicle results | Every results claim tagged “synthetic only” |

**Uncertainty:** How “hard” evasion will be is unknown until Phase 2 numbers exist. Small gaps are acceptable and must be reported honestly.

---

## 3. Minimal feature set

Target: a **small** vector suitable for a tiny MLP (not an LSTM/CANet-scale model). Prefer aggregates that catch flow + coarse payload anomalies.

### Keep (locked)

| Feature group | Definition (per window) | Why keep |
| --- | --- | --- |
| Per-ID counts | Message count for each of top-K IDs (+ “other” bucket) | Detects flood / drop / injection rate shifts |
| Inter-arrival mean/std | Mean and std of Δt for top-K IDs | Timing anomalies without full sequence models |
| Byte / payload stats | Per selected ID (or global): mean & std of payload bytes (or decoded phys signals if generator exposes them) | Catches spoof / replay payload shifts cheaply |
| ID entropy | Shannon entropy of ID distribution in window | Compact mix-shift signal |

Approximate dimension: with K≈8–16 IDs → roughly **40–80** features. Exact K locked in Phase 2 after generator ID set is fixed.

### Drop (and why)

| Drop | Why |
| --- | --- |
| Full raw 64-bit bitstrings as AE input | Bloats MLP; CRC/counter bits add noise (see CANnolo/CANdito practice of excluding them) |
| CRC / counters as reconstruction targets | Deterministic / high-entropy; inflate error without semantic value |
| Per-bit features | Too sparse for tiny MLP; poor portfolio ROI |
| Cross-ID joint LSTM latent | Out of scope for Phase 2 “tiny MLP” skill demo (CANet-style is literature context only) |
| Exact DBC physical decoding of a real car | No real DBC; synthetic phys fields only if generator defines them |

### Uncertainty

If Phase 2 shows payload stats add little vs counts+IAT alone, drop payload stats and re-run with ablations. Do not expand features to chase AUC.

---

## 4. Primary evasion method (portfolio recommendation)

### Decision

| Role | Method | Rationale |
| --- | --- | --- |
| **Primary demo** | **Constrained white-box PGD on features** | Directly encodes the thesis (min reconstruction error s.t. attack objective + \(\mathcal{C}\)). Shows ML skill: gradients, projection, constraint design, two-pass verification. Anchored in Madry-style PGD ([1]). |
| **Required baseline** | **Black-box mimicry** | Shows evasion without model access; links to classical IDS literature ([4], [5]). Prevents “only white-box matters” story. |

**Both are implemented and reported.** PGD is the headline technical artifact; mimicry is the honesty / realism baseline.

### Why not mimicry-only

Mimicry alone under-sells the optimization thesis and looks like feature engineering, not adversarial ML.

### Why not PGD-only

White-box-only results overstate attacker power for a portfolio narrative and skip the classical IDS evasion line of work.

### Formulation sketch (PGD)

\[
\min_{\delta}\; \|\,x+\delta - \mathrm{AE}(x+\delta)\,\|_2^2
\quad\text{s.t.}\quad
g_{\text{attack}}(x+\delta)\ge \tau_{\text{obj}},\quad
x+\delta\in\mathcal{C}
\]

Alternate practical form: minimize reconstruction score until score < τ_IDS while projecting onto \(\{x : g_{\text{attack}}(x)\ge\tau_{\text{obj}}\}\cap\mathcal{C}\).

---

## 5. Honest headline metric + secondary metrics

### Headline (operational)

1. Fit threshold \(\tau\) = **99th percentile** of reconstruction MSE on **held-out NORMAL validation** (never attack, never test-normal used for fitting if that split is reserved for FPR check).
2. Confirm empirical FPR on a disjoint NORMAL test slice is ≈1% (report exact).
3. At that **fixed** \(\tau\), measure **detection rate (TPR)** on:
   - naive attacks (per family)
   - mimicry-optimized attacks
   - PGD-optimized attacks
4. Headline number: **ΔTPR = TPR_naive − TPR_evasion** (per family and macro-average). Report even if small or negative.

### Secondary (diagnostic)

| Metric | Role |
| --- | --- |
| ROC-AUC (NORMAL vs attack) | Ranking quality; **not** the headline; threshold-free |
| Mean reconstruction error (NORMAL / naive / evasive) | Sanity for AE fit and evasion progress |
| Attack-objective retention rate | Fraction of evasive samples that still meet \(g_{\text{attack}}\) |
| Constraint-violation rate | Fraction failing \(\mathcal{C}\) after projection (must be ~0 after two-pass) |
| Per-family breakdown | Injection / spoof / flood / drop / replay |

### What NOT to claim

- Production readiness, real-vehicle security, or “defeats automotive IDS.”
- That ROC-AUC alone proves operational detection.
- That a large ΔTPR is guaranteed; small honest gaps are a valid result.
- Transfer of PGD examples to unseen architectures without measuring it.

---

## 6. Five key sources (verified)

Only verified entries. Each checked via arXiv / ACM / IEEE / DOI.

### [1] Madry et al., 2018 — PGD / first-order adversary

- **Title:** Towards Deep Learning Models Resistant to Adversarial Attacks  
- **Authors:** Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu  
- **Venue:** ICLR 2018; arXiv:1706.06083  
- **Verify:** https://arxiv.org/abs/1706.06083  
- **Relevance:** Canonical projected gradient descent adversary; basis for our white-box feature-space PGD.

### [2] Hanselmann et al., 2020 — CANet (CAN AE IDS)

- **Title:** CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data  
- **Authors:** Markus Hanselmann, Thilo Strauss, Katharina Dormann, Holger Ulmer  
- **Venue:** IEEE Access 8:58194–58205, 2020; DOI 10.1109/ACCESS.2020.2982544; arXiv:1906.02492  
- **Verify:** https://arxiv.org/abs/1906.02492 · https://doi.org/10.1109/ACCESS.2020.2982544  
- **Relevance:** Unsupervised reconstruction IDS on CAN; percentile thresholds on normal; synthetic+real evaluation culture we echo (synthetic-only for us).

### [3] Longari et al., 2021 — CANnolo (LSTM-AE CAN IDS)

- **Title:** CANnolo: An Anomaly Detection System Based on LSTM Autoencoders for Controller Area Network  
- **Authors:** Stefano Longari, Daniel Humberto Nova Valcarcel, Mattia Zago, Michele Carminati, Stefano Zanero  
- **Venue:** IEEE Transactions on Network and Service Management, 18(2):1913–1924, 2021; DOI 10.1109/TNSM.2020.3039121  
- **Verify:** https://doi.org/10.1109/TNSM.2020.3039121 · https://ieeexplore.ieee.org/document/9262960/  
- **Relevance:** LSTM autoencoder reconstruction-error IDS for CAN; justifies NORMAL-only training and reconstruction scoring (we substitute a tiny MLP for portfolio scope).

### [4] Wagner & Soto, 2002 — Mimicry attacks

- **Title:** Mimicry Attacks on Host-Based Intrusion Detection Systems  
- **Authors:** David Wagner, Paolo Soto  
- **Venue:** ACM CCS 2002; DOI 10.1145/586110.586145  
- **Verify:** https://dl.acm.org/doi/10.1145/586110.586145  
- **Relevance:** Defines mimicry as preserving malicious effect while matching the detector’s normal model — conceptual root of our black-box baseline.

### [5] Fogla et al., 2006 — Blending / anomaly IDS evasion

- **Title:** Evading Network Anomaly Detection Systems: Formal Reasoning and Practical Techniques  
- **Authors:** Prahlad Fogla, Monirul Sharif, Roberto Perdisci, Oleg Kolesnikov, Wenke Lee  
- **Venue:** ACM CCS 2006; DOI 10.1145/1180405.1180414  
- **Verify:** https://dl.acm.org/doi/10.1145/1180405.1180414  
- **Relevance:** Formalizes matching normal traffic statistics to evade payload anomaly detectors (polymorphic blending); informs mimicry feature-stat matching.

**Note:** Automotive-specific adversarial-IDS papers exist (e.g. later Longari/Cerracchio line on gradient evasion vs CAN IDS). They are useful Phase 2 reading but are **not** padded into this five-source set; the five above are sufficient and fully verified.

---

## 7. Design decisions locked for Phase 2

| Decision | Lock | Notes |
| --- | --- | --- |
| Data | Synthetic CAN-style only | Discrete frames out of generator; no real captures required |
| Windowing | **N = 50 consecutive frames** per window, stride 25 (50% overlap) for training; non-overlap optional at eval for speed | Fixed count handles low load better than pure time windows; document if generator prefers W=1.0s instead — pick **one** before training |
| Features | Counts + IAT mean/std + byte mean/std + ID entropy (§3) | K IDs fixed once generator ID set known |
| Model | Tiny MLP AE: `d → 64 → 16 → 64 → d`, ReLU, tied or untied weights OK | MSE reconstruction loss |
| Train set | NORMAL windows only | Early-stop on NORMAL val reconstruction |
| Threshold | **p99** of MSE on held-out NORMAL val | Never fit on attack/test |
| Attacks | injection, spoof (incl. gradual), flood, drop, replay | Naive + mimicry + PGD variants |
| Primary evasion | Constrained PGD + \(\Pi_{\mathcal{C}}\) + two-pass frame verify | Mimicry required baseline |
| Eval protocol | Fixed τ; report TPR naive vs evasive; ΔTPR; secondary metrics (§5); ≥3 seeds; per-family tables | No threshold retune; synthetic disclaimer on all plots |
| Non-goals | Real car, HIL, SecOC, production IDS | Explicit in README/results |

### Phase 2 implementation order (suggested)

1. Synthetic generator + frame schema + attack injectors.  
2. Feature extractor (single code path for train/eval/attack).  
3. Train MLP-AE on NORMAL; freeze; set τ = p99(NORMAL val).  
4. Naive attack eval → baseline TPR table.  
5. Mimicry morpher → TPR + objective retention.  
6. Constrained PGD + projection + two-pass → TPR + objective retention.  
7. Write results with synthetic caveats; no citation invention.

### Open uncertainties (do not block Phase 2)

- Exact K and whether payload stats survive ablation.  
- Whether W=1.0s time windows outperform N=50 for the chosen generator (if conflict, prefer N=50 and note).  
- Magnitude of ΔTPR — unknown a priori; honesty > hype.  
- How tight attack-objective constraints must be to avoid “evasion by undoing the attack.”

---

## Document control

| Field | Value |
| --- | --- |
| Path | `/workspace/bus-anomaly-evasion/DESIGN.md` |
| Phase | 1 — design only |
| Next | Phase 2 builder implements from §7 locks |
| Citation rule | Verified sources only; if unverifiable, omit |
