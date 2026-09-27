"""Evaluation: ROC-AUC and detection rate @ fixed FPR on naive / evasive attacks."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from src.features import FeatureScaler, extract_features
from src.generate import ATTACK_FAMILIES, generate_attack, generate_normal
from src.model import Autoencoder
from src.train import reconstruction_errors


def scores_for_frames(
    model: Autoencoder,
    scaler: FeatureScaler,
    df,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (errors, y, family) for windows extracted from ``df``."""
    batch = extract_features(df)
    if len(batch.X) == 0:
        return (
            np.zeros(0),
            np.zeros(0, dtype=np.int64),
            np.array([], dtype=object),
        )
    Xs = scaler.transform(batch.X)
    err = reconstruction_errors(model, Xs)
    return err, batch.y, batch.family


def detection_rate(errors: np.ndarray, tau: float, y: Optional[np.ndarray] = None) -> float:
    """TPR if y given (fraction of attack windows with err > tau), else alert rate."""
    if len(errors) == 0:
        return float("nan")
    alerts = errors > tau
    if y is None:
        return float(alerts.mean())
    mask = y == 1
    if mask.sum() == 0:
        return float("nan")
    return float(alerts[mask].mean())


def false_positive_rate(errors: np.ndarray, tau: float, y: np.ndarray) -> float:
    mask = y == 0
    if mask.sum() == 0:
        return float("nan")
    return float((errors[mask] > tau).mean())


def evaluate_stream(
    model: Autoencoder,
    scaler: FeatureScaler,
    tau: float,
    df,
) -> dict:
    err, y, family = scores_for_frames(model, scaler, df)
    out: dict = {
        "n_windows": int(len(err)),
        "n_attack": int((y == 1).sum()),
        "n_normal": int((y == 0).sum()),
        "mean_recon_error": float(err.mean()) if len(err) else float("nan"),
        "mean_recon_error_attack": float(err[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_recon_error_normal": float(err[y == 0].mean()) if (y == 0).any() else float("nan"),
        "tpr": detection_rate(err, tau, y),
        "fpr": false_positive_rate(err, tau, y),
        "alert_rate": detection_rate(err, tau),
    }
    # per-family TPR
    per_family = {}
    for fam in sorted(set(family.tolist())):
        if not fam:
            continue
        m = (family == fam) & (y == 1)
        if m.sum() == 0:
            continue
        per_family[fam] = {
            "n": int(m.sum()),
            "tpr": float((err[m] > tau).mean()),
            "mean_recon_error": float(err[m].mean()),
        }
    out["per_family"] = per_family

    # ROC-AUC if both classes present
    if (y == 0).any() and (y == 1).any():
        out["roc_auc"] = float(roc_auc_score(y, err))
    else:
        out["roc_auc"] = float("nan")
    return out


def evaluate_naive_attacks(
    model: Autoencoder,
    scaler: FeatureScaler,
    tau: float,
    seed: int = 0,
    duration_s: float = 10.0,
    normal_duration_s: float = 10.0,
) -> dict:
    """Build a mixed NORMAL+naive-attack eval set and score at fixed τ."""
    parts = [generate_normal(duration_s=normal_duration_s, seed=seed + 99)]
    for i, fam in enumerate(ATTACK_FAMILIES):
        kwargs = {}
        if fam == "spoof":
            kwargs["target_phys"] = 115.0
        if fam == "spoof" and i % 2 == 1:
            kwargs["gradual"] = True
        parts.append(generate_attack(fam, duration_s=duration_s, seed=seed + 10 + i, **kwargs))

    import pandas as pd
    # concatenate with time offsets so timestamps stay sorted globally
    offset = 0.0
    shifted = []
    for p in parts:
        q = p.copy()
        q["timestamp"] = q["timestamp"] + offset
        shifted.append(q)
        offset = float(q["timestamp"].max()) + 1.0
    df = pd.concat(shifted, ignore_index=True)

    overall = evaluate_stream(model, scaler, tau, df)

    # also per-family isolated (cleaner TPR)
    family_tables = {}
    for i, fam in enumerate(ATTACK_FAMILIES):
        kwargs = {}
        if fam == "spoof":
            kwargs["target_phys"] = 115.0
        atk = generate_attack(fam, duration_s=duration_s, seed=seed + 10 + i, **kwargs)
        # prepend a bit of normal so windows have contrast; label from attack frames
        fam_df = atk
        family_tables[fam] = evaluate_stream(model, scaler, tau, fam_df)

    overall["family_tables"] = family_tables
    return overall


def delta_tpr(tpr_naive: float, tpr_evasive: float) -> float:
    return float(tpr_naive - tpr_evasive)
