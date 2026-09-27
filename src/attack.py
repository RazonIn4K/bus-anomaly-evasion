"""Adversarial evasion.

(a) Mimicry (black-box): morph attack windows toward NORMAL feature stats while
    preserving the malicious objective.
(b) Constrained PGD (white-box): gradient descent on window features to minimize
    reconstruction error, projected onto bus-feasible set C, then mapped back to
    frames and re-extracted (two-pass verify).

Evasion must beat the naive attack on mean recon error and stay feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.features import (
    FEATURE_DIM,
    K,
    TOP_IDS,
    WINDOW_N,
    FeatureScaler,
    extract_features,
    extract_window_features,
    frames_from_df,
    window_indices,
)
from src.generate import ALL_IDS, INJECT_ID, ID_TO_SPEC
from src.model import Autoencoder
from src.train import reconstruction_errors


# ---------------------------------------------------------------------------
# Feasibility projection Π_C (feature space)
# ---------------------------------------------------------------------------

def project_features(x: np.ndarray, k: int = K) -> np.ndarray:
    """Project a feature vector (or batch) onto the feasible set C.

    Layout: counts(K+1) | iat_mean(K) | iat_std(K) | byte_mean(K) | byte_std(K) | entropy(1)
    """
    single = x.ndim == 1
    X = np.atleast_2d(x).astype(np.float64).copy()
    # counts >= 0, roughly <= WINDOW_N
    X[:, : k + 1] = np.clip(X[:, : k + 1], 0.0, float(WINDOW_N))
    # IAT mean > 0 (use small epsilon), std >= 0
    i0 = k + 1
    X[:, i0 : i0 + k] = np.maximum(X[:, i0 : i0 + k], 1e-6)
    X[:, i0 + k : i0 + 2 * k] = np.maximum(X[:, i0 + k : i0 + 2 * k], 0.0)
    # byte mean/std in [0, 255]
    b0 = i0 + 2 * k
    X[:, b0 : b0 + k] = np.clip(X[:, b0 : b0 + k], 0.0, 255.0)
    X[:, b0 + k : b0 + 2 * k] = np.clip(X[:, b0 + k : b0 + 2 * k], 0.0, 255.0)
    # entropy in [0, log2(K+1)]
    emax = float(np.log2(k + 1 + 1e-9))
    X[:, -1] = np.clip(X[:, -1], 0.0, emax)
    return X[0] if single else X


def constraint_violations(x: np.ndarray, k: int = K) -> int:
    """Count hard constraint violations in a feature vector/batch (0 = feasible)."""
    single = x.ndim == 1
    X = np.atleast_2d(x)
    n = 0
    n += int((X[:, : k + 1] < -1e-6).sum())
    i0 = k + 1
    n += int((X[:, i0 : i0 + k] < -1e-6).sum())
    n += int((X[:, i0 + k : i0 + 2 * k] < -1e-6).sum())
    b0 = i0 + 2 * k
    n += int(((X[:, b0 : b0 + k] < -1e-6) | (X[:, b0 : b0 + k] > 255.0 + 1e-6)).sum())
    n += int((X[:, b0 + k : b0 + 2 * k] < -1e-6).sum())
    n += int((X[:, -1] < -1e-6).sum())
    return n if not single else n


# ---------------------------------------------------------------------------
# Attack-objective retention
# ---------------------------------------------------------------------------

def _decode_phys_bytes(b0: int, b1: int, lo: float, hi: float) -> float:
    scaled = (int(b0) << 8) | int(b1)
    span = hi - lo if hi > lo else 1.0
    return lo + (scaled / 65535.0) * span


def objective_retained(
    window_df: pd.DataFrame,
    family: str,
    *,
    inject_min: int = 2,
    flood_id: int = 0x100,
    flood_min: int = 20,
    drop_id: int = 0x100,
    spoof_id: int = 0x100,
    spoof_target: float = 115.0,
    spoof_tol: float = 25.0,
    replay_id: int = 0x100,
) -> bool:
    """Return True if the window still achieves the attack objective."""
    family = family.lower()
    ids = window_df["can_id"].to_numpy()
    if family == "injection":
        return int((ids == INJECT_ID).sum()) >= inject_min
    if family == "flood":
        return int((ids == flood_id).sum()) >= flood_min
    if family == "drop":
        return int((ids == drop_id).sum()) == 0
    if family == "spoof":
        spec = ID_TO_SPEC[spoof_id]
        rows = window_df[window_df["can_id"] == spoof_id]
        if len(rows) == 0:
            return False
        vals = [
            _decode_phys_bytes(int(r.b0), int(r.b1), spec.phys_lo, spec.phys_hi)
            for r in rows.itertuples()
        ]
        return abs(float(np.mean(vals)) - spoof_target) <= spoof_tol
    if family == "replay":
        rows = window_df[window_df["can_id"] == replay_id]
        return len(rows) > 0  # payload identity checked softly; presence required
    return True


# ---------------------------------------------------------------------------
# Two-pass: morph frames toward target features while keeping objective
# ---------------------------------------------------------------------------

def _morph_window_frames(
    window_df: pd.DataFrame,
    target_features: np.ndarray,
    family: str,
    normal_byte_means: dict[int, float],
    rng: np.random.Generator,
    strength: float = 0.6,
) -> pd.DataFrame:
    """Edit frames so re-extracted features move toward ``target_features``.

    Preserves attack-critical structure (injected IDs, drop absence, spoof
    magnitude band, flood minimum presence).
    """
    df = window_df.copy().reset_index(drop=True)
    family = family.lower()
    top_index = {cid: i for i, cid in enumerate(TOP_IDS)}

    # 1) Soften free payload bytes toward normal means (helps byte_mean/std)
    for i, row in df.iterrows():
        cid = int(row["can_id"])
        # never touch injection signature bytes 0-1
        if family == "injection" and cid == INJECT_ID:
            for b in range(2, 8):
                target = normal_byte_means.get(cid, 8.0)
                cur = float(row[f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + strength * (target - cur)), 0, 255))
            continue
        if family == "spoof" and cid == 0x100:
            # keep b0,b1 (phys); soften free bytes
            for b in range(2, 8):
                target = normal_byte_means.get(cid, 8.0)
                cur = float(row[f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + strength * (target - cur)), 0, 255))
            continue
        if family == "replay" and cid == 0x100:
            continue  # keep replay payloads intact
        for b in range(8):
            if cid in normal_byte_means:
                target = normal_byte_means[cid]
            else:
                target = 0.0
            cur = float(df.at[i, f"b{b}"])
            # only gently pull; counters/phys still move a bit for non-critical
            if family == "flood" and cid == 0x100:
                # flood payloads are disposable noise — pull hard toward normal
                df.at[i, f"b{b}"] = int(np.clip(round((1 - strength) * cur + strength * target), 0, 255))
            elif b >= 2:
                df.at[i, f"b{b}"] = int(np.clip(round(cur + strength * (target - cur)), 0, 255))

    # 2) Timing: mild jitter to push IAT mean/std toward target (keep order)
    ts = df["timestamp"].to_numpy(dtype=np.float64).copy()
    if len(ts) >= 2:
        # small monotone jitter
        noise = rng.normal(0.0, 1e-5, size=len(ts))
        noise[0] = 0.0
        ts = ts + np.cumsum(noise * strength)
        # enforce strictly increasing
        for j in range(1, len(ts)):
            if ts[j] <= ts[j - 1]:
                ts[j] = ts[j - 1] + 1e-6
        df["timestamp"] = ts

    # 3) Family-specific rate softening toward normal counts (keep objective floor)
    if family == "flood":
        flood_id = 0x100
        idx = np.where(df["can_id"].to_numpy() == flood_id)[0]
        # target count from features if available
        t_count = float(target_features[top_index[flood_id]]) if flood_id in top_index else 25.0
        # keep at least flood_min objective
        keep_n = max(20, int(round(t_count)))
        if len(idx) > keep_n:
            # drop excess flood frames (prefer dropping middle ones)
            drop = idx[keep_n:]
            df = df.drop(df.index[drop]).reset_index(drop=True)
    if family == "injection":
        idx = np.where(df["can_id"].to_numpy() == INJECT_ID)[0]
        keep_n = max(2, int(len(idx) * (1.0 - 0.3 * strength)))
        if len(idx) > keep_n:
            drop = idx[keep_n:]
            df = df.drop(df.index[drop]).reset_index(drop=True)

    # Ensure bytes still in 0..255
    for b in range(8):
        df[f"b{b}"] = df[f"b{b}"].astype(int).clip(0, 255)

    return df


def _pad_or_trim_window(df: pd.DataFrame, n: int = WINDOW_N) -> pd.DataFrame:
    """Ensure exactly n frames for fair window compare (trim or repeat-last)."""
    if len(df) == n:
        return df.reset_index(drop=True)
    if len(df) > n:
        return df.iloc[:n].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("empty window")
    rows = df.to_dict("records")
    while len(rows) < n:
        last = dict(rows[-1])
        last["timestamp"] = float(last["timestamp"]) + 1e-4
        rows.append(last)
    return pd.DataFrame(rows).reset_index(drop=True)


def two_pass_realize(
    window_df: pd.DataFrame,
    target_features: np.ndarray,
    family: str,
    normal_byte_means: dict[int, float],
    rng: np.random.Generator,
    strength: float = 0.7,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Morph frames → re-extract features (honest two-pass)."""
    morphed = _morph_window_frames(
        window_df, target_features, family, normal_byte_means, rng, strength=strength,
    )
    morphed = _pad_or_trim_window(morphed, WINDOW_N)
    ts, ids, data = frames_from_df(morphed)
    feats = extract_window_features(ts, ids, data)
    feats = project_features(feats)
    return morphed, feats


def estimate_normal_byte_means(normal_df: pd.DataFrame) -> dict[int, float]:
    means: dict[int, float] = {}
    for cid in ALL_IDS:
        rows = normal_df[normal_df["can_id"] == cid]
        if len(rows) == 0:
            means[cid] = 0.0
            continue
        payload = rows[[f"b{i}" for i in range(8)]].to_numpy(dtype=np.float64)
        means[cid] = float(payload.mean())
    return means


# ---------------------------------------------------------------------------
# Mimicry (black-box)
# ---------------------------------------------------------------------------

@dataclass
class EvasionResult:
    method: str
    family: str
    naive_errors: np.ndarray
    evasive_errors: np.ndarray
    naive_features: np.ndarray
    evasive_features: np.ndarray
    objective_retention: float
    constraint_violation_rate: float
    mean_naive_error: float
    mean_evasive_error: float

    @property
    def error_reduction(self) -> float:
        return float(self.mean_naive_error - self.mean_evasive_error)


def mimicry_evade(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    family: str,
    seed: int = 0,
    strengths: Sequence[float] = (0.3, 0.5, 0.7, 0.85),
) -> EvasionResult:
    """Black-box mimicry: blend frame stats toward NORMAL; pick best strength."""
    rng = np.random.default_rng(seed)
    normal_byte_means = estimate_normal_byte_means(normal_df)
    # normal feature centroid (raw)
    normal_batch = extract_features(normal_df)
    normal_centroid = normal_batch.X.mean(axis=0) if len(normal_batch.X) else np.zeros(FEATURE_DIM)

    spans = window_indices(len(attack_df))
    naive_feats = []
    evasive_feats = []
    naive_errs = []
    evasive_errs = []
    retained = []
    violations = []

    labels = attack_df["label"].to_numpy()

    for s, e in spans:
        wdf = attack_df.iloc[s:e].reset_index(drop=True)
        # only evade attack-majority windows
        if float((labels[s:e] == "attack").mean()) < 0.02:
            continue
        ts, ids, data = frames_from_df(wdf)
        feat0 = extract_window_features(ts, ids, data)
        err0 = float(reconstruction_errors(model, scaler.transform(feat0[None, :]))[0])

        best_feat = feat0
        best_err = err0
        best_df = wdf
        for strength in strengths:
            # feature-space blend target
            target = (1.0 - strength) * feat0 + strength * normal_centroid
            target = project_features(target)
            morphed, feat_r = two_pass_realize(
                wdf, target, family, normal_byte_means, rng, strength=strength,
            )
            if not objective_retained(morphed, family):
                continue
            err = float(reconstruction_errors(model, scaler.transform(feat_r[None, :]))[0])
            if err < best_err:
                best_err = err
                best_feat = feat_r
                best_df = morphed

        naive_feats.append(feat0)
        evasive_feats.append(best_feat)
        naive_errs.append(err0)
        evasive_errs.append(best_err)
        retained.append(1.0 if objective_retained(best_df, family) else 0.0)
        violations.append(constraint_violations(best_feat))

    return EvasionResult(
        method="mimicry",
        family=family,
        naive_errors=np.asarray(naive_errs, dtype=np.float64),
        evasive_errors=np.asarray(evasive_errs, dtype=np.float64),
        naive_features=np.stack(naive_feats) if naive_feats else np.zeros((0, FEATURE_DIM)),
        evasive_features=np.stack(evasive_feats) if evasive_feats else np.zeros((0, FEATURE_DIM)),
        objective_retention=float(np.mean(retained)) if retained else float("nan"),
        constraint_violation_rate=float(np.mean([v > 0 for v in violations])) if violations else float("nan"),
        mean_naive_error=float(np.mean(naive_errs)) if naive_errs else float("nan"),
        mean_evasive_error=float(np.mean(evasive_errs)) if evasive_errs else float("nan"),
    )


# ---------------------------------------------------------------------------
# Constrained PGD (white-box) + two-pass
# ---------------------------------------------------------------------------

def pgd_feature_step(
    model: Autoencoder,
    x_scaled: torch.Tensor,
    steps: int = 40,
    step_size: float = 0.05,
    freeze_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Minimize reconstruction MSE via PGD in scaled feature space.

    ``freeze_mask`` (D,) bool — True means do not update that feature (objective).
    """
    x_adv = x_scaled.clone().detach().requires_grad_(True)
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        recon = model(x_adv)
        loss = (recon - x_adv).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            grad = x_adv.grad
            # descend on reconstruction error
            x_adv = x_adv - step_size * grad.sign()
            if freeze_mask is not None:
                x_adv = torch.where(freeze_mask.bool(), x_scaled, x_adv)
            x_adv = x_adv.detach().requires_grad_(True)
    return x_adv.detach()


def pgd_evade(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    family: str,
    seed: int = 0,
    steps: int = 40,
    step_size: float = 0.08,
) -> EvasionResult:
    """White-box constrained PGD + Π_C + two-pass frame verification."""
    rng = np.random.default_rng(seed)
    normal_byte_means = estimate_normal_byte_means(normal_df)
    model.eval()

    # freeze count features that encode the attack objective
    freeze = np.zeros(FEATURE_DIM, dtype=bool)
    id_to_idx = {cid: i for i, cid in enumerate(TOP_IDS)}
    if family == "flood" and 0x100 in id_to_idx:
        freeze[id_to_idx[0x100]] = True  # keep flood count from collapsing fully
    if family == "drop" and 0x100 in id_to_idx:
        freeze[id_to_idx[0x100]] = True
    if family == "injection":
        # other-bucket often holds INJECT_ID
        freeze[K] = True  # count_other
    freeze_t = torch.from_numpy(freeze)

    spans = window_indices(len(attack_df))
    labels = attack_df["label"].to_numpy()

    naive_feats, evasive_feats = [], []
    naive_errs, evasive_errs = [], []
    retained, violations = [], []

    for s, e in spans:
        wdf = attack_df.iloc[s:e].reset_index(drop=True)
        if float((labels[s:e] == "attack").mean()) < 0.02:
            continue
        ts, ids, data = frames_from_df(wdf)
        feat0 = extract_window_features(ts, ids, data)
        x0_s = scaler.transform(feat0[None, :]).astype(np.float32)
        err0 = float(reconstruction_errors(model, x0_s)[0])

        x_t = torch.from_numpy(x0_s)
        x_adv_s = pgd_feature_step(model, x_t, steps=steps, step_size=step_size, freeze_mask=freeze_t)
        # unscale + project
        x_adv_raw = x_adv_s.numpy()[0] * np.where(scaler.std < 1e-8, 1.0, scaler.std) + scaler.mean
        x_adv_raw = project_features(x_adv_raw)

        # two-pass realize
        morphed, feat_r = two_pass_realize(
            wdf, x_adv_raw, family, normal_byte_means, rng, strength=0.75,
        )
        # if objective lost, fall back to milder morph of PGD features
        if not objective_retained(morphed, family):
            morphed, feat_r = two_pass_realize(
                wdf, x_adv_raw, family, normal_byte_means, rng, strength=0.4,
            )
        # final projection on re-extracted
        feat_r = project_features(feat_r)
        err_r = float(reconstruction_errors(model, scaler.transform(feat_r[None, :]))[0])

        # if two-pass made things worse than naive, keep feature-space PGD score
        # but still report two-pass features only if objective held; else mark
        # — honesty: always report two-pass (realized) error
        naive_feats.append(feat0)
        evasive_feats.append(feat_r)
        naive_errs.append(err0)
        evasive_errs.append(err_r)
        retained.append(1.0 if objective_retained(morphed, family) else 0.0)
        violations.append(constraint_violations(feat_r))

    return EvasionResult(
        method="pgd",
        family=family,
        naive_errors=np.asarray(naive_errs, dtype=np.float64),
        evasive_errors=np.asarray(evasive_errs, dtype=np.float64),
        naive_features=np.stack(naive_feats) if naive_feats else np.zeros((0, FEATURE_DIM)),
        evasive_features=np.stack(evasive_feats) if evasive_feats else np.zeros((0, FEATURE_DIM)),
        objective_retention=float(np.mean(retained)) if retained else float("nan"),
        constraint_violation_rate=float(np.mean([v > 0 for v in violations])) if violations else float("nan"),
        mean_naive_error=float(np.mean(naive_errs)) if naive_errs else float("nan"),
        mean_evasive_error=float(np.mean(evasive_errs)) if evasive_errs else float("nan"),
    )


def run_evasion_suite(
    model: Autoencoder,
    scaler: FeatureScaler,
    tau: float,
    seed: int = 0,
    duration_s: float = 10.0,
) -> dict:
    """Run mimicry + PGD on each attack family; return tables for the report."""
    from src.generate import ATTACK_FAMILIES, generate_attack, generate_normal
    from src.evaluate import detection_rate

    normal_df = generate_normal(duration_s=duration_s, seed=seed + 50)
    tables = {}
    for i, fam in enumerate(ATTACK_FAMILIES):
        kwargs = {}
        if fam == "spoof":
            kwargs["target_phys"] = 115.0
        atk = generate_attack(fam, duration_s=duration_s, seed=seed + 20 + i, **kwargs)
        mim = mimicry_evade(model, scaler, atk, normal_df, fam, seed=seed + 100 + i)
        pgd = pgd_evade(model, scaler, atk, normal_df, fam, seed=seed + 200 + i)
        tpr_naive = detection_rate(mim.naive_errors, tau)
        tpr_mim = detection_rate(mim.evasive_errors, tau)
        tpr_pgd = detection_rate(pgd.evasive_errors, tau)
        tables[fam] = {
            "n_windows": int(len(mim.naive_errors)),
            "mean_recon_naive": mim.mean_naive_error,
            "mean_recon_mimicry": mim.mean_evasive_error,
            "mean_recon_pgd": pgd.mean_evasive_error,
            "tpr_naive": tpr_naive,
            "tpr_mimicry": tpr_mim,
            "tpr_pgd": tpr_pgd,
            "delta_tpr_mimicry": float(tpr_naive - tpr_mim) if tpr_naive == tpr_naive else float("nan"),
            "delta_tpr_pgd": float(tpr_naive - tpr_pgd) if tpr_naive == tpr_naive else float("nan"),
            "obj_retention_mimicry": mim.objective_retention,
            "obj_retention_pgd": pgd.objective_retention,
            "constraint_viol_mimicry": mim.constraint_violation_rate,
            "constraint_viol_pgd": pgd.constraint_violation_rate,
        }
    return tables
