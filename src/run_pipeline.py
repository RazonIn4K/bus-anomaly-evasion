"""End-to-end pipeline: train → evaluate naive → evade → write report artifacts.

Synthetic-data research only. Fixed seeds. No threshold retune on attacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.attack import run_evasion_suite
from src.evaluate import evaluate_naive_attacks, evaluate_stream
from src.generate import generate_normal
from src.train import run_training_pipeline


def run_seed(seed: int, out_dir: Path, duration_normal: float = 45.0, epochs: int = 50) -> dict:
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
    # FPR check on pure NORMAL test stream (already in meta); also score fresh normal
    normal_check = generate_normal(duration_s=15.0, seed=seed + 777)
    fpr_stream = evaluate_stream(model, scaler, tau, normal_check)

    evasion = run_evasion_suite(
        model, scaler, tau, seed=seed, duration_s=10.0,
    )

    summary = {
        "seed": seed,
        "tau": tau,
        "fpr_test_normal_split": train["meta"]["fpr_test_normal"],
        "fpr_fresh_normal_stream": fpr_stream["fpr"] if fpr_stream["fpr"] == fpr_stream["fpr"] else fpr_stream["alert_rate"],
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
    (out_dir / f"summary_seed{seed}.json").write_text(json.dumps(summary, indent=2))
    return summary


def aggregate(summaries: list[dict]) -> dict:
    families = list(summaries[0]["evasion"].keys())
    agg = {"seeds": [s["seed"] for s in summaries], "families": {}, "macro": {}}

    fprs = [s["fpr_test_normal_split"] for s in summaries]
    agg["fpr_at_tau_mean"] = float(np.mean(fprs))
    agg["fpr_at_tau_std"] = float(np.std(fprs))
    agg["tau_mean"] = float(np.mean([s["tau"] for s in summaries]))

    macro_naive, macro_mim, macro_pgd = [], [], []
    macro_d_mim, macro_d_pgd = [], []
    recon_naive, recon_mim, recon_pgd = [], [], []

    for fam in families:
        rows = [s["evasion"][fam] for s in summaries]
        def _m(key):
            vals = [r[key] for r in rows if r[key] == r[key]]
            return float(np.mean(vals)) if vals else float("nan")

        entry = {
            "tpr_naive": _m("tpr_naive"),
            "tpr_mimicry": _m("tpr_mimicry"),
            "tpr_pgd": _m("tpr_pgd"),
            "delta_tpr_mimicry": _m("delta_tpr_mimicry"),
            "delta_tpr_pgd": _m("delta_tpr_pgd"),
            "mean_recon_naive": _m("mean_recon_naive"),
            "mean_recon_mimicry": _m("mean_recon_mimicry"),
            "mean_recon_pgd": _m("mean_recon_pgd"),
            "obj_retention_mimicry": _m("obj_retention_mimicry"),
            "obj_retention_pgd": _m("obj_retention_pgd"),
            "constraint_viol_mimicry": _m("constraint_viol_mimicry"),
            "constraint_viol_pgd": _m("constraint_viol_pgd"),
        }
        agg["families"][fam] = entry
        macro_naive.append(entry["tpr_naive"])
        macro_mim.append(entry["tpr_mimicry"])
        macro_pgd.append(entry["tpr_pgd"])
        macro_d_mim.append(entry["delta_tpr_mimicry"])
        macro_d_pgd.append(entry["delta_tpr_pgd"])
        recon_naive.append(entry["mean_recon_naive"])
        recon_mim.append(entry["mean_recon_mimicry"])
        recon_pgd.append(entry["mean_recon_pgd"])

    agg["macro"] = {
        "tpr_naive": float(np.nanmean(macro_naive)),
        "tpr_mimicry": float(np.nanmean(macro_mim)),
        "tpr_pgd": float(np.nanmean(macro_pgd)),
        "delta_tpr_mimicry": float(np.nanmean(macro_d_mim)),
        "delta_tpr_pgd": float(np.nanmean(macro_d_pgd)),
        "mean_recon_naive": float(np.nanmean(recon_naive)),
        "mean_recon_mimicry": float(np.nanmean(recon_mim)),
        "mean_recon_pgd": float(np.nanmean(recon_pgd)),
    }
    return agg


def write_report(agg: dict, summaries: list[dict], path: Path) -> None:
    m = agg["macro"]
    lines = [
        "# Results Report — Bus Anomaly Evasion",
        "",
        "> **Disclaimer (synthetic only):** All numbers below are from a synthetic",
        "> CAN-style generator. They do **not** claim transfer to any real vehicle,",
        "> OEM bus matrix, or production IDS. Portfolio research artifact only.",
        "",
        f"- Seeds: `{agg['seeds']}`",
        f"- Threshold rule: τ = p99 MSE on held-out NORMAL validation (never attack/test).",
        f"- Empirical FPR on NORMAL test split (mean±std): "
        f"**{agg['fpr_at_tau_mean']:.4f} ± {agg['fpr_at_tau_std']:.4f}**",
        f"- Mean τ: `{agg['tau_mean']:.6f}`",
        "",
        "## Headline metrics (macro-average over attack families)",
        "",
        "| Method | TPR | ΔTPR vs naive | Mean recon MSE |",
        "| --- | ---: | ---: | ---: |",
        f"| Naive | {m['tpr_naive']:.4f} | — | {m['mean_recon_naive']:.6f} |",
        f"| Mimicry | {m['tpr_mimicry']:.4f} | {m['delta_tpr_mimicry']:.4f} | {m['mean_recon_mimicry']:.6f} |",
        f"| Constrained PGD | {m['tpr_pgd']:.4f} | {m['delta_tpr_pgd']:.4f} | {m['mean_recon_pgd']:.6f} |",
        "",
        f"**Headline ΔTPR (naive − PGD):** `{m['delta_tpr_pgd']:.4f}`",
        "",
        "## Per-family table (mean over seeds)",
        "",
        "| Family | TPR naive | TPR mimicry | TPR PGD | ΔTPR PGD | "
        "Recon naive | Recon PGD | Obj ret PGD | Viol PGD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fam, e in agg["families"].items():
        lines.append(
            f"| {fam} | {e['tpr_naive']:.4f} | {e['tpr_mimicry']:.4f} | {e['tpr_pgd']:.4f} | "
            f"{e['delta_tpr_pgd']:.4f} | {e['mean_recon_naive']:.6f} | {e['mean_recon_pgd']:.6f} | "
            f"{e['obj_retention_pgd']:.3f} | {e['constraint_viol_pgd']:.3f} |"
        )

    lines += [
        "",
        "## Secondary notes",
        "",
        "- Attack-objective retention and constraint-violation rates are reported per family.",
        "- Two-pass verification: feature-space step → frame morph → re-extract → score.",
        "- Evasion is expected to lower mean reconstruction error vs naive; ΔTPR may be small.",
        "",
        "## Honest gaps",
        "",
    ]
    # honest commentary based on numbers
    if m["delta_tpr_pgd"] < 0.05:
        lines.append(
            "- ΔTPR is small. That is an acceptable portfolio outcome: the detector still"
            " catches many evasive windows at fixed FPR≈1%, and we do not inflate the gap."
        )
    else:
        lines.append(
            "- Constrained PGD reduces detection relative to naive at fixed τ; magnitude"
            " should be read alongside objective-retention (evasion that undoes the attack"
            " is not counted as success)."
        )
    if m["mean_recon_pgd"] < m["mean_recon_naive"]:
        lines.append(
            "- Mean reconstruction error under PGD is lower than naive — evasion directionally correct."
        )
    else:
        lines.append(
            "- Mean reconstruction error under PGD did not beat naive after two-pass realization;"
            " feature-space gains did not fully survive frame projection. Reported honestly."
        )
    lines += [
        "- Mimicry is the black-box baseline; PGD is the white-box primary demo.",
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
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    write_report(agg, summaries, out / "report.md")
    print("Wrote", out / "report.md")
    print(json.dumps(agg["macro"], indent=2))


if __name__ == "__main__":
    main()
