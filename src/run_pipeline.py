"""End-to-end pipeline: train → evaluate naive → evade → write report artifacts.

Synthetic-data research only. Fixed seeds. No threshold retune on attacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.attack import run_evasion_suite
from src.evaluate import (
    evaluate_naive_attacks,
    evaluate_stream,
    matched_fpr_threshold,
    scores_for_frames,
)
from src.features import effective_independent_windows
from src.generate import generate_normal
from src.train import run_training_pipeline


def _json_safe(obj):
    """Convert NaN/Inf to None for strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj



def run_seed(seed: int, out_dir: Path, duration_normal: float = 200.0, epochs: int = 60) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = run_training_pipeline(
        seed=seed,
        duration_s=duration_normal,  # API compat; blocks use block_duration_s=200
        out_dir=out_dir,
        epochs=epochs,
        n_blocks=12,
        block_duration_s=200.0,
        n_train_blocks=6,
        n_earlystop_blocks=1,
        n_calib_blocks=3,
        n_test_blocks=2,
    )
    model, scaler, tau = train["model"], train["scaler"], train["tau"]

    naive = evaluate_naive_attacks(
        model, scaler, tau, seed=seed, duration_s=10.0, normal_duration_s=10.0,
    )
    # Large fresh NORMAL stream for FPR stability + matched 1% FPR threshold (ROC)
    normal_check = generate_normal(duration_s=200.0, seed=seed + 777)
    fpr_stream = evaluate_stream(model, scaler, tau, normal_check)
    fresh_err, fresh_y, _ = scores_for_frames(model, scaler, normal_check)
    # Use NORMAL windows only (y==0); pure normal stream is all zeros
    normal_errs = fresh_err[fresh_y == 0] if (fresh_y == 0).any() else fresh_err
    tau_fpr1 = matched_fpr_threshold(normal_errs, target_fpr=0.01)
    fpr_at_tau_fpr1 = float((normal_errs > tau_fpr1).mean()) if len(normal_errs) else float("nan")

    evasion = run_evasion_suite(
        model, scaler, tau, seed=seed, duration_s=10.0, tau_fpr1=tau_fpr1,
    )

    summary = {
        "seed": seed,
        "tau": tau,
        "tau_calibration_split": train["meta"].get(
            "tau_calibration_split", "temporal_held_out_NORMAL_val"
        ),
        "fpr_same_stream_split": train["meta"]["fpr_test_normal"],
        "fpr_test_normal_split": train["meta"]["fpr_test_normal"],  # alias
        "fpr_fresh_normal_stream": (
            fpr_stream["fpr"] if fpr_stream["fpr"] == fpr_stream["fpr"]
            else fpr_stream["alert_rate"]
        ),
        "n_fresh_normal_windows": int(len(normal_errs)),
        "n_fresh_normal_effective_indep": int(effective_independent_windows(len(normal_errs))),
        "n_val_effective_indep": int(train["meta"].get("n_val_effective_indep", 0)),
        "tau_fpr1_matched": tau_fpr1,
        "fpr_at_tau_fpr1": fpr_at_tau_fpr1,
        "n_val_windows": int(train["meta"].get("n_val", 0)),
        "naive_overall_tpr": naive["tpr"],
        "naive_roc_auc": naive["roc_auc"],
        "naive_family_tables": {
            fam: {
                "tpr": v["tpr"],
                "mean_recon_error": v["mean_recon_error_attack"],
                "n_attack": v["n_attack"],
            }
            for fam, v in naive["family_tables"].items()
        },
        "evasion": evasion,
        "disclaimer": "synthetic only — not a real vehicle bus",
    }
    (out_dir / f"summary_seed{seed}.json").write_text(json.dumps(_json_safe(summary), indent=2))
    return summary


def aggregate(summaries: list[dict]) -> dict:
    families = list(summaries[0]["evasion"].keys())
    agg = {"seeds": [s["seed"] for s in summaries], "families": {}, "macro": {}}

    fprs_split = [s.get("fpr_same_stream_split", s["fpr_test_normal_split"]) for s in summaries]
    fprs_fresh = [s["fpr_fresh_normal_stream"] for s in summaries]
    agg["fpr_same_stream_split_mean"] = float(np.mean(fprs_split))
    agg["fpr_same_stream_split_std"] = float(np.std(fprs_split))
    agg["fpr_fresh_normal_mean"] = float(np.mean(fprs_fresh))
    agg["fpr_fresh_normal_std"] = float(np.std(fprs_fresh))
    # Population SD across seeds (ddof=0) for the predeclared gate; also store per-seed
    agg["fpr_fresh_normal_per_seed"] = [float(x) for x in fprs_fresh]
    agg["n_val_effective_indep_mean"] = float(np.mean([
        s.get("n_val_effective_indep", 0) for s in summaries
    ]))
    agg["n_fresh_normal_effective_indep_mean"] = float(np.mean([
        s.get("n_fresh_normal_effective_indep", 0) for s in summaries
    ]))
    # Predeclared FPR gates (Zen): SD < 1pp, mean ≤ 1.5%, no seed > 2.5%
    agg["gate_fpr_std_ok"] = bool(agg["fpr_fresh_normal_std"] < 0.01)
    agg["gate_fpr_mean_ok"] = bool(agg["fpr_fresh_normal_mean"] <= 0.015)
    agg["gate_fpr_max_ok"] = bool(max(fprs_fresh) <= 0.025 if fprs_fresh else False)
    agg["gate_fpr_all_ok"] = bool(
        agg["gate_fpr_std_ok"] and agg["gate_fpr_mean_ok"] and agg["gate_fpr_max_ok"]
    )
    # Back-compat aliases (same-stream split)
    agg["fpr_at_tau_mean"] = agg["fpr_same_stream_split_mean"]
    agg["fpr_at_tau_std"] = agg["fpr_same_stream_split_std"]
    agg["tau_mean"] = float(np.mean([s["tau"] for s in summaries]))
    agg["tau_fpr1_matched_mean"] = float(np.mean([s["tau_fpr1_matched"] for s in summaries]))
    agg["n_val_windows_mean"] = float(np.mean([s.get("n_val_windows", 0) for s in summaries]))
    agg["n_fresh_normal_windows_mean"] = float(
        np.mean([s.get("n_fresh_normal_windows", 0) for s in summaries])
    )
    agg["tau_calibration_split"] = summaries[0].get(
        "tau_calibration_split", "temporal_held_out_NORMAL_val"
    )

    keys = [
        "tpr_naive", "tpr_naive_pgd_matched", "tpr_mimicry", "tpr_pgd",
        "delta_tpr_mimicry", "delta_tpr_pgd",
        "tpr_naive_fpr1", "tpr_mimicry_fpr1", "tpr_naive_pgd_matched_fpr1", "tpr_pgd_fpr1",
        "delta_tpr_mimicry_fpr1", "delta_tpr_pgd_fpr1",
        "tpr_naive_ungated", "tpr_mimicry_ungated", "tpr_pgd_ungated",
        "delta_tpr_mimicry_ungated", "delta_tpr_pgd_ungated",
        "mean_recon_naive", "mean_recon_mimicry", "mean_recon_pgd",
        "mean_recon_pgd_all_paths", "mean_recon_naive_pgd_matched",
        "obj_retention_mimicry", "obj_retention_pgd",
        "constraint_viol_mimicry", "constraint_viol_pgd",
        "frac_improved_mimicry", "frac_fallback_mimicry",
        "frac_improved_pgd", "frac_fallback_pgd",
        "n_true_pgd", "n_thin", "n_naive_fallback",
    ]

    macro_acc = {k: [] for k in (
        "tpr_naive", "tpr_mimicry", "tpr_pgd",
        "tpr_naive_pgd_matched",
        "delta_tpr_mimicry", "delta_tpr_pgd",
        "tpr_naive_fpr1", "tpr_mimicry_fpr1", "tpr_pgd_fpr1",
        "delta_tpr_mimicry_fpr1", "delta_tpr_pgd_fpr1",
        "tpr_mimicry_ungated", "tpr_pgd_ungated",
        "delta_tpr_mimicry_ungated", "delta_tpr_pgd_ungated",
        "mean_recon_naive", "mean_recon_mimicry", "mean_recon_pgd",
        "frac_improved_pgd", "frac_fallback_pgd",
        "n_true_pgd", "n_thin", "n_naive_fallback",
    )}

    for fam in families:
        rows = [s["evasion"][fam] for s in summaries]

        def _m(key):
            vals = []
            for r in rows:
                if key not in r:
                    continue
                v = r[key]
                if isinstance(v, (int, float)) and v == v:
                    vals.append(float(v))
            return float(np.mean(vals)) if vals else float("nan")

        entry = {k: _m(k) for k in keys}
        # Sum path counts across seeds then we also keep means; store mean counts
        entry["path_counts_pgd"] = {
            "thin": entry["n_thin"],
            "true_pgd": entry["n_true_pgd"],
            "naive_fallback": entry["n_naive_fallback"],
        }
        agg["families"][fam] = entry
        for k in macro_acc:
            macro_acc[k].append(entry[k])

    agg["macro"] = {k: float(np.nanmean(v)) for k, v in macro_acc.items()}
    agg["macro"]["path_counts_pgd"] = {
        "thin": agg["macro"]["n_thin"],
        "true_pgd": agg["macro"]["n_true_pgd"],
        "naive_fallback": agg["macro"]["n_naive_fallback"],
    }

    # Predeclared injection-mimicry@τ headline gate (Zen):
    # naive≥20% every seed, mean paired Δ≥10pp, Δ>0 all seeds.
    inj_naive_seeds = []
    inj_delta_seeds = []
    for s in summaries:
        row = s["evasion"].get("injection", {})
        tn = row.get("tpr_naive")
        tm = row.get("tpr_mimicry")
        if isinstance(tn, (int, float)) and tn == tn:
            inj_naive_seeds.append(float(tn))
        if isinstance(tn, (int, float)) and isinstance(tm, (int, float)) and tn == tn and tm == tm:
            inj_delta_seeds.append(float(tn) - float(tm))
    agg["injection_naive_tpr_per_seed"] = inj_naive_seeds
    agg["injection_delta_tpr_per_seed"] = inj_delta_seeds
    agg["gate_injection_naive_min_ok"] = bool(
        len(inj_naive_seeds) == len(summaries) and all(v >= 0.20 for v in inj_naive_seeds)
    )
    agg["gate_injection_delta_mean_ok"] = bool(
        len(inj_delta_seeds) == len(summaries) and float(np.mean(inj_delta_seeds)) >= 0.10
    )
    agg["gate_injection_delta_all_pos_ok"] = bool(
        len(inj_delta_seeds) == len(summaries) and all(d > 0.0 for d in inj_delta_seeds)
    )
    agg["gate_injection_headline_ok"] = bool(
        agg["gate_injection_naive_min_ok"]
        and agg["gate_injection_delta_mean_ok"]
        and agg["gate_injection_delta_all_pos_ok"]
    )
    return agg


def write_report(agg: dict, summaries: list[dict], path: Path) -> None:
    m = agg["macro"]
    fams = agg["families"]
    cal = agg.get("tau_calibration_split", "temporal_held_out_NORMAL_val")

    def _fmt(x, nd=4):
        if x is None or (isinstance(x, float) and x != x):
            return "nan"
        return f"{x:.{nd}f}"

    inj = fams.get("injection", {})
    spoof = fams.get("spoof", {})
    replay = fams.get("replay", {})
    flood = fams.get("flood", {})
    drop = fams.get("drop", {})

    lines = [
        "# Results Report — Bus Anomaly Evasion",
        "",
        "> **Disclaimer (synthetic only):** All numbers below are from a synthetic",
        "> CAN-style generator. They do **not** claim transfer to any real vehicle,",
        "> OEM bus matrix, or production IDS. Portfolio research artifact only.",
        "",
        f"- Seeds: `{agg['seeds']}`",
        f"- Threshold rule: τ = p99 MSE on held-out NORMAL validation (never attack/test).",
        f"- τ calibration split: **`{cal}`** (ordered independent NORMAL calibration blocks; p99 primary).",
        f"- Mean val windows / seed: **{_fmt(agg.get('n_val_windows_mean'), 1)}** "
        f"(effective indep ≈ **{_fmt(agg.get('n_val_effective_indep_mean'), 1)}**)",
        f"- Same-stream split FPR (mean±std): "
        f"**{_fmt(agg['fpr_same_stream_split_mean'])} ± {_fmt(agg['fpr_same_stream_split_std'])}**",
        f"- Fresh-normal FPR @ τ per seed: `{agg.get('fpr_fresh_normal_per_seed')}`",
        f"- Fresh-normal FPR @ τ (mean±std): "
        f"**{_fmt(agg['fpr_fresh_normal_mean'])} ± {_fmt(agg['fpr_fresh_normal_std'])}** "
        f"(gates: std<1pp={agg.get('gate_fpr_std_ok')}, mean≤1.5%={agg.get('gate_fpr_mean_ok')}, "
        f"max≤2.5%={agg.get('gate_fpr_max_ok')})",
        f"- **gate_fpr_all_ok={agg.get('gate_fpr_all_ok')}** · "
        f"**gate_injection_headline_ok={agg.get('gate_injection_headline_ok')}** · "
        f"**GATES_PASS (win claim)={bool(agg.get('gate_fpr_all_ok') and agg.get('gate_injection_headline_ok'))}**",
        f"- Mean τ (p99 held-out NORMAL val) — **primary**: `{agg['tau_mean']:.6f}`",
        f"- Secondary τ @ matched 1% FPR (fresh-NORMAL ROC; diagnostic): "
        f"`{_fmt(agg.get('tau_fpr1_matched_mean'), 6)}`",
        f"- Fresh-NORMAL windows / seed (mean): "
        f"**{_fmt(agg.get('n_fresh_normal_windows_mean'), 1)}** "
        f"(effective indep ≈ **{_fmt(agg.get('n_fresh_normal_effective_indep_mean'), 1)}**)",
        "",
        "## Headlines (read these first)",
        "",
        "### (a) Injection mimicry @ τ (primary = held-out NORMAL p99)",
        "",
        f"- Gate headline_ok: **{agg.get('gate_injection_headline_ok')}** "
        f"(naive≥20%/seed, mean Δ≥10pp, Δ>0 all seeds)",
        f"- Per-seed naive TPR: `{agg.get('injection_naive_tpr_per_seed')}`",
        f"- Per-seed ΔTPR (naive−mim): `{agg.get('injection_delta_tpr_per_seed')}`",
        f"- Mean naive → mimicry @ τ: **{_fmt(inj.get('tpr_naive'))}** → "
        f"**{_fmt(inj.get('tpr_mimicry'))}** (Δ **{_fmt(inj.get('delta_tpr_mimicry'))}**)",
        f"- Objective retention (mimicry): **{_fmt(inj.get('obj_retention_mimicry'), 3)}**; "
        f"constraint violations: **{_fmt(inj.get('constraint_viol_mimicry'), 3)}**",
        (
            "- **Headline:** injection mimicry reduces detection at fixed NORMAL-derived τ."
            if agg.get("gate_injection_headline_ok")
            else "- **Honest null / no all-seed win claim:** all-seed injection mimicry headline gate FAILED (see per-seed Δ; zeros allowed). FPR gate status above. Reporting actual rates — not a forced win."
        ),
        f"- Secondary/diagnostic matched 1% FPR (not primary): naive "
        f"**{_fmt(inj.get('tpr_naive_fpr1'))}** → mimicry **{_fmt(inj.get('tpr_mimicry_fpr1'))}** "
        f"(Δ **{_fmt(inj.get('delta_tpr_mimicry_fpr1'))}**).",
        "",
        "### (b) Detector blind spot: replay / spoof",
        "",
        "Stateless window features miss in-range, on-schedule payload attacks.",
        f"- Replay naive TPR @ τ: **{_fmt(replay.get('tpr_naive_ungated', replay.get('tpr_naive')))}** "
        f"(mimicry **{_fmt(replay.get('tpr_mimicry'))}**)",
        f"- Spoof naive TPR @ τ: **{_fmt(spoof.get('tpr_naive_ungated', spoof.get('tpr_naive')))}** "
        f"(mimicry **{_fmt(spoof.get('tpr_mimicry'))}**)",
        "Low TPR here is a detector limitation, not an evasion win.",
        "",
        "### (c) Flood / drop not evadable under the objective",
        "",
        "Rate attacks dominate reconstruction via count/IAT features. Typical naive",
        f"recon MSE is ~{_fmt(flood.get('mean_recon_naive'), 1)} (flood) / "
        f"~{_fmt(drop.get('mean_recon_naive'), 1)} (drop) vs τ ~ {_fmt(agg['tau_mean'], 2)}. "
        "Frame-space byte PGD cannot close that gap while retaining the objective — "
        "say **not evadable under objective**, not “PGD failed.”",
        f"- Flood true_pgd TPR @ τ: **{_fmt(flood.get('tpr_pgd'))}** "
        f"(Δ **{_fmt(flood.get('delta_tpr_pgd'))}**, "
        f"n_true_pgd={_fmt(flood.get('n_true_pgd'), 1)})",
        f"- Drop true_pgd TPR @ τ: **{_fmt(drop.get('tpr_pgd'))}** "
        f"(Δ **{_fmt(drop.get('delta_tpr_pgd'))}**, "
        f"n_true_pgd={_fmt(drop.get('n_true_pgd'), 1)})",
        "",
        "## Per-family table (mean over seeds)",
        "",
        "| Family | TPR naive @τ | TPR mim @τ | Δ mim | TPR true_pgd @τ | Δ PGD | "
        "n_true_pgd | TPR mim @1%FPR | Obj ret mim | Obj ret PGD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam, e in fams.items():
        naive_t = e.get("tpr_naive", float("nan"))
        lines.append(
            f"| {fam} | {_fmt(naive_t)} | {_fmt(e.get('tpr_mimicry'))} | "
            f"{_fmt(e.get('delta_tpr_mimicry'))} | {_fmt(e.get('tpr_pgd'))} | "
            f"{_fmt(e.get('delta_tpr_pgd'))} | {_fmt(e.get('n_true_pgd'), 1)} | "
            f"{_fmt(e.get('tpr_mimicry_fpr1'))} | {_fmt(e.get('obj_retention_mimicry'), 3)} | "
            f"{_fmt(e.get('obj_retention_pgd'), 3)} |"
        )

    lines += [
        "",
        "PGD ΔTPR is `nan` when a family has zero `true_pgd` windows (not imputed).",
        "n_true_pgd is mean windows / seed labeled frame-space PGD.",
        "",
        "## Frame-space PGD notes (Π_C matches code)",
        "",
        "- PGD runs on **free payload bytes** of frames in each window (objective-critical",
        "  bytes frozen: spoof phys b0/b1, full replay payload).",
        "- Differentiable torch path: byte mean/std per TOP_IDS ID; counts / IAT /",
        "  entropy stay fixed from the window structure.",
        "- **Π_C in-loop:** after every gradient step, free bytes are projected onto",
        "  `[0, 255] ∩ L∞(x₀, ε)` relative to the original window bytes.",
        "- After optimization: quantize to uint8, **re-extract with the original numpy",
        "  extractor**, score AE — two-pass verify. No feature-space-then-morph path.",
        "- Thinning / blending is **mimicry only**; never labeled `true_pgd`.",
        "- No never-worse-than-naive clamp; `frac_fallback` is honest.",
        f"- Macro frac_improved (PGD): **{_fmt(m.get('frac_improved_pgd'))}**; "
        f"frac_fallback: **{_fmt(m.get('frac_fallback_pgd'))}**",
        "",
        "## Honest gaps",
        "",
        "- Macro ΔTPR across families hides the injection mimicry win and the",
        "  flood/drop non-evasion — read the headlines and per-family table.",
        "- Fresh-normal FPR std target is < 1% after enlarging val / fresh streams;",
        "  residual deviation is reported above, not clamped.",
        "- Synthetic only: no real captures, no HIL, no production claims.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "source .venv/bin/activate",
        "PYTHONPATH=. python -m src.run_pipeline",
        "pytest -q",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2]
    summaries = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        summaries.append(run_seed(seed, out))
    agg = aggregate(summaries)
    (out / "aggregate.json").write_text(json.dumps(_json_safe(agg), indent=2))
    write_report(agg, summaries, out / "report.md")
    print("Wrote", out / "report.md")
    print(json.dumps(agg["macro"], indent=2))


if __name__ == "__main__":
    main()
