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



def run_seed(seed: int, out_dir: Path, duration_normal: float = 150.0, epochs: int = 50) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = run_training_pipeline(
        seed=seed,
        duration_s=duration_normal,
        out_dir=out_dir,
        epochs=epochs,
    )
    model, scaler, tau = train["model"], train["scaler"], train["tau"]

    naive = evaluate_naive_attacks(
        model, scaler, tau, seed=seed, duration_s=10.0, normal_duration_s=10.0,
    )
    # Large fresh NORMAL stream for FPR stability + matched 1% FPR threshold (ROC)
    normal_check = generate_normal(duration_s=60.0, seed=seed + 777)
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
    return agg


def write_report(agg: dict, summaries: list[dict], path: Path) -> None:
    m = agg["macro"]
    pc = m.get("path_counts_pgd", {})
    cal = agg.get("tau_calibration_split", "temporal_held_out_NORMAL_val")

    def _fmt(x, nd=4):
        if x != x:
            return "nan"
        return f"{x:.{nd}f}"

    lines = [
        "# Results Report — Bus Anomaly Evasion",
        "",
        "> **Disclaimer (synthetic only):** All numbers below are from a synthetic",
        "> CAN-style generator. They do **not** claim transfer to any real vehicle,",
        "> OEM bus matrix, or production IDS. Portfolio research artifact only.",
        "",
        f"- Seeds: `{agg['seeds']}`",
        f"- Threshold rule: τ = p99 MSE on held-out NORMAL validation (never attack/test).",
        f"- τ calibration split: **`{cal}`** (temporal early/mid/late contiguous windows).",
        f"- Same-stream split FPR (mean±std): "
        f"**{_fmt(agg['fpr_same_stream_split_mean'])} ± {_fmt(agg['fpr_same_stream_split_std'])}**",
        f"- Fresh-normal FPR (mean±std): "
        f"**{_fmt(agg['fpr_fresh_normal_mean'])} ± {_fmt(agg['fpr_fresh_normal_std'])}**",
        f"- Mean τ: `{agg['tau_mean']:.6f}`",
        "",
        "## Headline metrics (objective-gated; PGD = true_pgd path only)",
        "",
        "Primary TPR / ΔTPR require `objective_retained`. Constrained PGD headline",
        "further requires path=`true_pgd` (rate-thinning is **not** counted as PGD).",
        "Ungated numbers are secondary only. Families with zero true_pgd windows",
        "contribute `nan` to the PGD headline (not imputed as success).",
        "",
        "| Method | TPR (gated) | ΔTPR vs naive | Mean recon MSE |",
        "| --- | ---: | ---: | ---: |",
        f"| Naive (obj-gated) | {_fmt(m['tpr_naive'])} | — | {_fmt(m['mean_recon_naive'], 6)} |",
        f"| Mimicry (obj-gated) | {_fmt(m['tpr_mimicry'])} | {_fmt(m['delta_tpr_mimicry'])} | {_fmt(m['mean_recon_mimicry'], 6)} |",
        f"| Naive (true_pgd-matched) | {_fmt(m.get('tpr_naive_pgd_matched'))} | — | "
        f"(matched windows only) |",
        f"| Constrained PGD (true_pgd ∩ obj) | {_fmt(m['tpr_pgd'])} | {_fmt(m['delta_tpr_pgd'])} | {_fmt(m['mean_recon_pgd'], 6)} |",
        "",
        f"**Headline ΔTPR (true_pgd-matched naive − true_pgd, objective-gated):** `{_fmt(m['delta_tpr_pgd'])}`",
        "",
        "### PGD path counts (mean windows / family / seed)",
        "",
        f"- thin: **{_fmt(pc.get('thin', m.get('n_thin', float('nan'))), 2)}**",
        f"- true_pgd: **{_fmt(pc.get('true_pgd', m.get('n_true_pgd', float('nan'))), 2)}**",
        f"- naive_fallback: **{_fmt(pc.get('naive_fallback', m.get('n_naive_fallback', float('nan'))), 2)}**",
        "",
        f"- frac_improved (PGD all paths, recon < naive): **{_fmt(m.get('frac_improved_pgd'))}**",
        f"- frac_fallback (PGD all paths, recon ≥ naive; no clamp): **{_fmt(m.get('frac_fallback_pgd'))}**",
        "",
        "### Secondary (ungated)",
        "",
        f"- Mimicry ungated ΔTPR: `{_fmt(m.get('delta_tpr_mimicry_ungated'))}` "
        f"(TPR `{_fmt(m.get('tpr_mimicry_ungated'))}`)",
        f"- PGD ungated ΔTPR (all paths, may include thin): `{_fmt(m.get('delta_tpr_pgd_ungated'))}` "
        f"(TPR `{_fmt(m.get('tpr_pgd_ungated'))}`)",
        "",
        "## Per-family table (mean over seeds; primary gated)",
        "",
        "| Family | TPR naive† | TPR mim | TPR true_pgd | ΔTPR PGD | "
        "thin/true_pgd/fb | frac_imp | frac_fb | Obj ret PGD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam, e in agg["families"].items():
        pc_f = e.get("path_counts_pgd", {})
        paths = (
            f"{_fmt(pc_f.get('thin', e.get('n_thin')), 1)}/"
            f"{_fmt(pc_f.get('true_pgd', e.get('n_true_pgd')), 1)}/"
            f"{_fmt(pc_f.get('naive_fallback', e.get('n_naive_fallback')), 1)}"
        )
        naive_t = e.get("tpr_naive_pgd_matched", e["tpr_naive"])
        lines.append(
            f"| {fam} | {_fmt(naive_t)} | {_fmt(e['tpr_mimicry'])} | {_fmt(e['tpr_pgd'])} | "
            f"{_fmt(e['delta_tpr_pgd'])} | {paths} | {_fmt(e.get('frac_improved_pgd'))} | "
            f"{_fmt(e.get('frac_fallback_pgd'))} | {_fmt(e['obj_retention_pgd'], 3)} |"
        )

    lines += [
        "",
        "† Naive TPR for PGD Δ is matched to true_pgd ∩ objective_retained windows.",
        "",
        "## Secondary notes",
        "",
        "- Attack-objective retention and constraint-violation rates are reported per family.",
        "- Two-pass verification: feature-space step → frame morph → re-extract → score.",
        "- Never-worse-than-naive clamp removed: recon MSE and frac_fallback are actual.",
        "- In-loop Π_C is a soft feature-space clip before frame morph; hard feasibility is",
        "  enforced by byte quantization (0–255) and two-pass re-extract — residual gap vs a",
        "  true projective Π_C on the discrete frame set remains (P2 note).",
        "- Evasion is expected to lower mean reconstruction error vs naive when the morph",
        "  works; ΔTPR may be small or family-local.",
        "",
        "## Honest gaps",
        "",
    ]
    dpgd = m["delta_tpr_pgd"]
    if dpgd != dpgd or dpgd < 0.05:
        lines.append(
            "- Gated true_pgd ΔTPR is small or undefined on some families. Acceptable portfolio"
            " outcome: we do not inflate PGD by counting rate-thinning as white-box success."
        )
    else:
        lines.append(
            "- Constrained true_pgd reduces detection relative to matched naive at fixed τ;"
            " read alongside objective-retention and path counts."
        )
    if m["mean_recon_pgd"] == m["mean_recon_pgd"] and m["mean_recon_pgd"] < m["mean_recon_naive"]:
        lines.append(
            "- Mean reconstruction error under true_pgd is lower than naive — directionally correct."
        )
    else:
        lines.append(
            "- Mean reconstruction error under true_pgd did not clearly beat naive after two-pass;"
            " reported honestly (no clamp)."
        )
    lines += [
        "- Mimicry is the black-box baseline; true PGD is the white-box primary demo.",
        "- No real captures; no HIL; no production claims.",
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
