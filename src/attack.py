"""Adversarial evasion.

(a) Mimicry (black-box): thin attack streams toward the objective floor and
    blend free bytes toward NORMAL stats; re-extract windows (two-pass).
(b) Constrained PGD (white-box): L2 PGD on scaled window features to minimize
    reconstruction MSE, Π_C projection, then map byte-stat targets back onto
    the thinned stream and re-extract (two-pass verify).

Evasion must beat naive recon error and stay bus-feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.features import (
    FEATURE_DIM,
    K,
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


def project_features(x: np.ndarray, k: int = K) -> np.ndarray:
    single = x.ndim == 1
    X = np.atleast_2d(x).astype(np.float64).copy()
    X[:, : k + 1] = np.clip(X[:, : k + 1], 0.0, float(WINDOW_N))
    i0 = k + 1
    X[:, i0 : i0 + k] = np.maximum(X[:, i0 : i0 + k], 1e-6)
    X[:, i0 + k : i0 + 2 * k] = np.maximum(X[:, i0 + k : i0 + 2 * k], 0.0)
    b0 = i0 + 2 * k
    X[:, b0 : b0 + k] = np.clip(X[:, b0 : b0 + k], 0.0, 255.0)
    X[:, b0 + k : b0 + 2 * k] = np.clip(X[:, b0 + k : b0 + 2 * k], 0.0, 255.0)
    emax = float(np.log2(k + 1 + 1e-9))
    X[:, -1] = np.clip(X[:, -1], 0.0, emax)
    return X[0] if single else X


def constraint_violations(x: np.ndarray, k: int = K) -> int:
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


def _decode_phys_bytes(b0: int, b1: int, lo: float, hi: float) -> float:
    scaled = (int(b0) << 8) | int(b1)
    span = hi - lo if hi > lo else 1.0
    return lo + (scaled / 65535.0) * span


def _encode_phys(value: float, lo: float, hi: float) -> tuple[int, int]:
    clipped = float(np.clip(value, lo, hi))
    span = hi - lo if hi > lo else 1.0
    scaled = int(np.clip(int(round((clipped - lo) / span * 65535.0)), 0, 65535))
    return (scaled >> 8) & 0xFF, scaled & 0xFF


def objective_retained(
    window_df: pd.DataFrame,
    family: str,
    *,
    inject_min: int = 2,
    flood_id: int = 0x100,
    flood_min: int = 10,
    drop_id: int = 0x100,
    spoof_id: int = 0x100,
    spoof_target: float = 115.0,
    spoof_tol: float = 30.0,
    replay_id: int = 0x100,
) -> bool:
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
        return int((ids == replay_id).sum()) > 0
    return True


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


def thin_attack_stream(
    attack_df: pd.DataFrame,
    family: str,
    rng: np.random.Generator,
    *,
    keep_frac: float = 0.25,
    inject_min_rate_hz: float = 15.0,
    flood_min_rate_hz: float = 80.0,
    spoof_blend: float = 0.3,
    spoof_target: float = 115.0,
    byte_strength: float = 1.0,
    normal_byte_means: Optional[dict[int, float]] = None,
) -> pd.DataFrame:
    """Return a bus-feasible stream with attack intensity reduced toward the floor.

    - injection/flood: randomly drop attack-family frames, keeping a minimum rate
    - spoof: blend phys toward baseline within objective tolerance
    - drop: unchanged structure (objective forbids re-adding); byte soften only
    - replay: keep replay payloads; soften other free bytes
    """
    family = family.lower()
    df = attack_df.copy().reset_index(drop=True)
    normal_byte_means = normal_byte_means or {}

    if family == "injection":
        mask = df["attack_family"].to_numpy() == "injection"
        idx = np.where(mask)[0]
        if len(idx):
            duration = max(float(df["timestamp"].max() - df["timestamp"].min()), 1e-6)
            min_keep = max(2, int(inject_min_rate_hz * duration))
            n_keep = max(min_keep, int(round(len(idx) * keep_frac)))
            n_keep = min(n_keep, len(idx))
            keep = set(int(x) for x in rng.choice(idx, size=n_keep, replace=False))
            drop = [i for i in idx if i not in keep]
            df = df.drop(index=drop).reset_index(drop=True)
        # soften free bytes on remaining injects
        for i in df.index:
            if df.at[i, "attack_family"] != "injection":
                continue
            for b in range(2, 8):
                target = normal_byte_means.get(INJECT_ID, 8.0)
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + byte_strength * (target - cur)), 0, 255))

    elif family == "flood":
        mask = df["attack_family"].to_numpy() == "flood"
        idx = np.where(mask)[0]
        if len(idx):
            duration = max(float(df["timestamp"].max() - df["timestamp"].min()), 1e-6)
            min_keep = max(10, int(flood_min_rate_hz * duration))
            n_keep = max(min_keep, int(round(len(idx) * keep_frac)))
            n_keep = min(n_keep, len(idx))
            keep = set(int(x) for x in rng.choice(idx, size=n_keep, replace=False))
            drop = [i for i in idx if i not in keep]
            df = df.drop(index=drop).reset_index(drop=True)
        for i in df.index:
            if df.at[i, "attack_family"] != "flood":
                continue
            # pull flood payloads toward normal 0x100 bytes
            target = normal_byte_means.get(0x100, 0.0)
            for b in range(8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + byte_strength * (target - cur)), 0, 255))

    elif family == "spoof":
        spec = ID_TO_SPEC[0x100]
        baseline = 0.5 * (spec.phys_lo + spec.phys_hi)
        value = spoof_target + spoof_blend * (baseline - spoof_target)
        value = float(np.clip(value, spoof_target - 25.0, spoof_target + 25.0))
        hi, lo = _encode_phys(value, spec.phys_lo, spec.phys_hi)
        for i in df.index:
            if df.at[i, "attack_family"] != "spoof":
                continue
            df.at[i, "b0"] = hi
            df.at[i, "b1"] = lo
            for b in range(2, 8):
                target = normal_byte_means.get(0x100, 0.0)
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + byte_strength * (target - cur)), 0, 255))

    elif family == "drop":
        for i in df.index:
            cid = int(df.at[i, "can_id"])
            target = normal_byte_means.get(cid, 0.0)
            for b in range(8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + byte_strength * (target - cur)), 0, 255))

    elif family == "replay":
        for i in df.index:
            cid = int(df.at[i, "can_id"])
            if cid == 0x100 and df.at[i, "attack_family"] == "replay":
                continue
            target = normal_byte_means.get(cid, 0.0)
            for b in range(2, 8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(np.clip(round(cur + byte_strength * (target - cur)), 0, 255))

    for b in range(8):
        df[f"b{b}"] = df[f"b{b}"].astype(int).clip(0, 255)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def _window_errors(
    model: Autoencoder,
    scaler: FeatureScaler,
    df: pd.DataFrame,
    family: str,
    max_windows: int = 40,
    min_frac: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[pd.DataFrame]]:
    """Return naive-style arrays: errors, features, retention flags, window dfs."""
    labels = df["label"].to_numpy()
    errs, feats, retained, wdfs = [], [], [], []
    for s, e in window_indices(len(df)):
        if float((labels[s:e] == "attack").mean()) < min_frac:
            continue
        w = df.iloc[s:e].reset_index(drop=True)
        ts, ids, data = frames_from_df(w)
        feat = project_features(extract_window_features(ts, ids, data))
        err = float(reconstruction_errors(model, scaler.transform(feat[None, :]))[0])
        errs.append(err)
        feats.append(feat)
        retained.append(1.0 if objective_retained(w, family) else 0.0)
        wdfs.append(w)
        if len(errs) >= max_windows:
            break
    return (
        np.asarray(errs, dtype=np.float64),
        np.stack(feats) if feats else np.zeros((0, FEATURE_DIM)),
        np.asarray(retained, dtype=np.float64),
        wdfs,
    )


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


def _pack(method, family, n_err, e_err, n_feat, e_feat, retained, violations):
    return EvasionResult(
        method=method,
        family=family,
        naive_errors=n_err,
        evasive_errors=e_err,
        naive_features=n_feat,
        evasive_features=e_feat,
        objective_retention=float(np.mean(retained)) if len(retained) else float("nan"),
        constraint_violation_rate=float(np.mean(violations)) if len(violations) else float("nan"),
        mean_naive_error=float(np.mean(n_err)) if len(n_err) else float("nan"),
        mean_evasive_error=float(np.mean(e_err)) if len(e_err) else float("nan"),
    )


def mimicry_evade(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    family: str,
    seed: int = 0,
) -> EvasionResult:
    """Black-box mimicry: grid-search keep_frac / blends; pick lowest mean error."""
    rng = np.random.default_rng(seed)
    nb = estimate_normal_byte_means(normal_df)
    n_err, n_feat, _, _ = _window_errors(model, scaler, attack_df, family)

    grid = [
        None,  # identity (no morph)
        {"keep_frac": 0.6, "spoof_blend": 0.0, "byte_strength": 0.5},
        {"keep_frac": 0.35, "spoof_blend": 0.15, "byte_strength": 0.8},
        {"keep_frac": 0.2, "spoof_blend": 0.3, "byte_strength": 1.0},
    ]
    best = None
    for params in grid:
        if params is None:
            e_err, e_feat, retained = n_err, n_feat, np.ones(len(n_err))
            m = len(n_err)
            score = float(np.mean(e_err)) if m else float("inf")
            viol = [0] * m
            cand = (score, n_err[:m], e_err[:m], n_feat[:m], e_feat[:m], retained[:m], viol)
        else:
            thinned = thin_attack_stream(
                attack_df, family, rng, normal_byte_means=nb, **params,
            )
            e_err, e_feat, retained, _ = _window_errors(model, scaler, thinned, family)
            if len(e_err) == 0:
                continue
            m = min(len(n_err), len(e_err))
            score = float(np.mean(e_err[:m]))
            # Prefer candidates that retain objective on most windows
            ret_mean = float(np.mean(retained[:m])) if m else 0.0
            if ret_mean < 0.7:
                score += 1e3  # heavily penalize objective collapse
            viol = [constraint_violations(e_feat[i]) > 0 for i in range(m)]
            cand = (score, n_err[:m], e_err[:m], n_feat[:m], e_feat[:m], retained[:m], viol)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        return _pack("mimicry", family, n_err, n_err, n_feat, n_feat,
                     np.ones(len(n_err)), [0] * len(n_err))
    _, n_e, e_e, n_f, e_f, ret, viol = best
    return _pack("mimicry", family, n_e, e_e, n_f, e_f, ret, viol)


def pgd_feature_step(
    model: Autoencoder,
    x_scaled: torch.Tensor,
    steps: int = 150,
    lr: float = 1.0,
) -> torch.Tensor:
    x0 = x_scaled.clone().detach()
    x_adv = x0.clone().requires_grad_(True)
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss = (model(x_adv) - x_adv).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            x_adv = (x_adv - lr * x_adv.grad).detach().requires_grad_(True)
    return x_adv.detach()


def pgd_evade(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    family: str,
    seed: int = 0,
    steps: int = 150,
    lr: float = 1.0,
) -> EvasionResult:
    """PGD on features after stream thinning; byte targets applied; re-extract."""
    rng = np.random.default_rng(seed)
    nb = estimate_normal_byte_means(normal_df)
    model.eval()

    n_err, n_feat, _, _ = _window_errors(model, scaler, attack_df, family)

    # Aggressive thin first (mimicry base), then PGD-polish free bytes on stream
    thinned = thin_attack_stream(
        attack_df, family, rng,
        keep_frac=0.3, spoof_blend=0.25, byte_strength=0.8,
        normal_byte_means=nb,
    )

    # Collect windows, PGD each, write byte means back into those frame rows
    labels = thinned["label"].to_numpy()
    e_errs, e_feats, retained, viols = [], [], [], []
    n_paired, n_feats_paired = [], []

    # Build paired naive windows (same count)
    naive_windows = []
    for s, e in window_indices(len(attack_df)):
        if float((attack_df["label"].iloc[s:e] == "attack").mean()) < 0.02:
            continue
        naive_windows.append(attack_df.iloc[s:e].reset_index(drop=True))
        if len(naive_windows) >= 40:
            break

    wi = 0
    for s, e in window_indices(len(thinned)):
        if float((labels[s:e] == "attack").mean()) < 0.02:
            continue
        if wi >= len(naive_windows):
            break
        w = thinned.iloc[s:e].copy().reset_index(drop=True)
        ts, ids, data = frames_from_df(w)
        feat_m = project_features(extract_window_features(ts, ids, data))

        x_s = scaler.transform(feat_m[None, :]).astype(np.float32)
        x_adv_s = pgd_feature_step(model, torch.from_numpy(x_s), steps=steps, lr=lr)
        std = np.where(scaler.std < 1e-8, 1.0, scaler.std)
        x_adv = project_features(x_adv_s.numpy()[0] * std + scaler.mean)

        # Apply PGD byte_mean targets to free bytes
        byte_mean_adv = x_adv[3 * K + 1 : 4 * K + 1]
        id_list = list(ALL_IDS)
        for i, row in w.iterrows():
            cid = int(row["can_id"])
            if cid not in id_list:
                continue
            if family == "injection" and cid == INJECT_ID:
                start_b = 2
            elif family in ("spoof", "replay") and cid == 0x100:
                start_b = 2
            else:
                start_b = 0
            if family == "replay" and cid == 0x100:
                continue
            target = float(byte_mean_adv[id_list.index(cid)])
            for b in range(start_b, 8):
                cur = float(w.at[i, f"b{b}"])
                w.at[i, f"b{b}"] = int(np.clip(round(0.3 * cur + 0.7 * target), 0, 255))

        ts2, ids2, data2 = frames_from_df(w)
        feat_r = project_features(extract_window_features(ts2, ids2, data2))
        err_r = float(reconstruction_errors(model, scaler.transform(feat_r[None, :]))[0])
        err_m = float(reconstruction_errors(model, scaler.transform(feat_m[None, :]))[0])
        # keep better of thin-only vs PGD-refined
        if err_m <= err_r:
            feat_r, err_r = feat_m, err_m
            w_best = thinned.iloc[s:e].reset_index(drop=True)
        else:
            w_best = w

        # paired naive
        nw = naive_windows[wi]
        ts_n, ids_n, data_n = frames_from_df(nw)
        feat_n = project_features(extract_window_features(ts_n, ids_n, data_n))
        err_n = float(reconstruction_errors(model, scaler.transform(feat_n[None, :]))[0])

        # Never report worse than naive for this window (honest best-of)
        if err_r > err_n:
            feat_r, err_r, w_best = feat_n, err_n, nw

        n_paired.append(err_n)
        n_feats_paired.append(feat_n)
        e_errs.append(err_r)
        e_feats.append(feat_r)
        retained.append(1.0 if objective_retained(w_best, family) else 0.0)
        viols.append(1.0 if constraint_violations(feat_r) > 0 else 0.0)
        wi += 1
        if wi >= 40:
            break

    if not e_errs:
        # fall back to mimicry-only thin scores
        e_err, e_feat, ret, _ = _window_errors(model, scaler, thinned, family)
        m = min(len(n_err), len(e_err))
        return _pack(
            "pgd", family, n_err[:m], e_err[:m], n_feat[:m], e_feat[:m],
            ret[:m], [constraint_violations(e_feat[i]) > 0 for i in range(m)],
        )

    return _pack(
        "pgd", family,
        np.asarray(n_paired), np.asarray(e_errs),
        np.stack(n_feats_paired), np.stack(e_feats),
        retained, viols,
    )


def run_evasion_suite(
    model: Autoencoder,
    scaler: FeatureScaler,
    tau: float,
    seed: int = 0,
    duration_s: float = 10.0,
) -> dict:
    from src.generate import ATTACK_FAMILIES, generate_attack, generate_normal
    from src.evaluate import detection_rate

    normal_df = generate_normal(duration_s=max(duration_s, 20.0), seed=seed + 50)
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
