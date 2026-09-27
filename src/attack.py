"""Adversarial evasion.

(a) Mimicry (black-box): thin attack streams toward the objective floor and
    blend free bytes toward per-ID per-position NORMAL stats; re-extract
    (two-pass).
(b) Constrained frame-space PGD (white-box): PGD directly on FREE payload
    bytes of attack frames. A differentiable torch path computes byte
    mean/std per ID; counts/IAT/entropy stay fixed from the window structure.
    Every step projects onto [0, 255] ∩ L∞ ball around the original bytes
    (Π_C in-loop). After optimization, quantize to uint8 and re-verify with
    the original numpy feature extractor (two-pass).

Thinning is mimicry only — never labeled as PGD. No never-worse-than-naive
clamp: report actual recon MSE and frac_fallback.
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


def estimate_normal_byte_means(normal_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per-ID, per-byte-position means (shape (8,)).

    One scalar across all 8 bytes flattens counters / constants / phys encodings;
    thinning and blending must preserve per-position structure.
    """
    means: dict[int, np.ndarray] = {}
    cols = [f"b{i}" for i in range(8)]
    for cid in ALL_IDS:
        rows = normal_df[normal_df["can_id"] == cid]
        if len(rows) == 0:
            means[cid] = np.zeros(8, dtype=np.float64)
            continue
        payload = rows[cols].to_numpy(dtype=np.float64)
        means[cid] = payload.mean(axis=0).astype(np.float64)
    return means


def _byte_target_vec(
    normal_byte_means: dict,
    cid: int,
    default: float = 0.0,
) -> np.ndarray:
    """Return length-8 float target vector for ``cid``."""
    raw = normal_byte_means.get(cid)
    if raw is None:
        return np.full(8, default, dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        # Back-compat: scalar-per-ID → broadcast (should not occur after rebuild)
        return np.full(8, float(arr[0]), dtype=np.float64)
    out = np.zeros(8, dtype=np.float64)
    n = min(8, arr.size)
    out[:n] = arr[:n]
    return out


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
    normal_byte_means: Optional[dict[int, np.ndarray]] = None,
) -> pd.DataFrame:
    """Return a bus-feasible stream with attack intensity reduced toward the floor.

    - injection/flood: randomly drop attack-family frames, keeping a minimum rate
    - spoof: blend phys toward baseline within objective tolerance
    - drop: unchanged structure (objective forbids re-adding); byte soften only
    - replay: keep replay payloads; soften other free bytes

    Byte blending uses per-ID, per-position NORMAL means (not one scalar per ID).
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
        # soften free bytes on remaining injects (per-position; inject ID absent from NORMAL)
        inj_target = _byte_target_vec(normal_byte_means, INJECT_ID, default=8.0)
        for i in df.index:
            if df.at[i, "attack_family"] != "injection":
                continue
            for b in range(2, 8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(
                    np.clip(round(cur + byte_strength * (inj_target[b] - cur)), 0, 255)
                )

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
        flood_target = _byte_target_vec(normal_byte_means, 0x100, default=0.0)
        for i in df.index:
            if df.at[i, "attack_family"] != "flood":
                continue
            # pull flood payloads toward normal 0x100 per-position bytes
            for b in range(8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(
                    np.clip(round(cur + byte_strength * (flood_target[b] - cur)), 0, 255)
                )

    elif family == "spoof":
        spec = ID_TO_SPEC[0x100]
        baseline = 0.5 * (spec.phys_lo + spec.phys_hi)
        value = spoof_target + spoof_blend * (baseline - spoof_target)
        value = float(np.clip(value, spoof_target - 25.0, spoof_target + 25.0))
        hi, lo = _encode_phys(value, spec.phys_lo, spec.phys_hi)
        spoof_target_bytes = _byte_target_vec(normal_byte_means, 0x100, default=0.0)
        for i in df.index:
            if df.at[i, "attack_family"] != "spoof":
                continue
            df.at[i, "b0"] = hi
            df.at[i, "b1"] = lo
            for b in range(2, 8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(
                    np.clip(round(cur + byte_strength * (spoof_target_bytes[b] - cur)), 0, 255)
                )

    elif family == "drop":
        for i in df.index:
            cid = int(df.at[i, "can_id"])
            target = _byte_target_vec(normal_byte_means, cid, default=0.0)
            for b in range(8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(
                    np.clip(round(cur + byte_strength * (target[b] - cur)), 0, 255)
                )

    elif family == "replay":
        for i in df.index:
            cid = int(df.at[i, "can_id"])
            if cid == 0x100 and df.at[i, "attack_family"] == "replay":
                continue
            target = _byte_target_vec(normal_byte_means, cid, default=0.0)
            for b in range(2, 8):
                cur = float(df.at[i, f"b{b}"])
                df.at[i, f"b{b}"] = int(
                    np.clip(round(cur + byte_strength * (target[b] - cur)), 0, 255)
                )

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


PATH_THIN = "thin"
PATH_TRUE_PGD = "true_pgd"
PATH_NAIVE_FALLBACK = "naive_fallback"


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
    retained_mask: np.ndarray
    path_labels: np.ndarray
    frac_improved: float
    frac_fallback: float
    path_counts: dict

    @property
    def error_reduction(self) -> float:
        return float(self.mean_naive_error - self.mean_evasive_error)


def _path_counts(labels: np.ndarray) -> dict:
    counts = {PATH_THIN: 0, PATH_TRUE_PGD: 0, PATH_NAIVE_FALLBACK: 0}
    for lab in labels:
        key = str(lab)
        if key in counts:
            counts[key] += 1
    return counts


def _pack(
    method,
    family,
    n_err,
    e_err,
    n_feat,
    e_feat,
    retained,
    violations,
    path_labels=None,
):
    n_err = np.asarray(n_err, dtype=np.float64)
    e_err = np.asarray(e_err, dtype=np.float64)
    retained_arr = np.asarray(retained, dtype=np.float64)
    if path_labels is None:
        # Mimicry is stream-thinning / blend; label every window as thin.
        path_labels = np.full(len(e_err), PATH_THIN if method == "mimicry" else PATH_TRUE_PGD, dtype=object)
    else:
        path_labels = np.asarray(path_labels, dtype=object)
    m = len(e_err)
    if m and len(n_err) == m:
        improved = e_err < n_err
        # frac_fallback: windows that did not beat naive recon (honest, no clamp)
        fallback = e_err >= n_err
        frac_improved = float(improved.mean())
        frac_fallback = float(fallback.mean())
    else:
        frac_improved = float("nan")
        frac_fallback = float("nan")
    return EvasionResult(
        method=method,
        family=family,
        naive_errors=n_err,
        evasive_errors=e_err,
        naive_features=n_feat,
        evasive_features=e_feat,
        objective_retention=float(np.mean(retained_arr)) if len(retained_arr) else float("nan"),
        constraint_violation_rate=float(np.mean(violations)) if len(violations) else float("nan"),
        mean_naive_error=float(np.mean(n_err)) if len(n_err) else float("nan"),
        mean_evasive_error=float(np.mean(e_err)) if len(e_err) else float("nan"),
        retained_mask=retained_arr.astype(bool) if len(retained_arr) else np.zeros(0, dtype=bool),
        path_labels=path_labels,
        frac_improved=frac_improved,
        frac_fallback=frac_fallback,
        path_counts=_path_counts(path_labels),
    )


def gated_detection_rate(
    errors: np.ndarray,
    tau: float,
    retained_mask: np.ndarray,
    path_labels: Optional[np.ndarray] = None,
    require_path: Optional[str] = None,
) -> float:
    """TPR on windows that retain the attack objective (and optional path filter)."""
    if len(errors) == 0:
        return float("nan")
    mask = np.asarray(retained_mask, dtype=bool)
    if require_path is not None:
        if path_labels is None:
            return float("nan")
        mask = mask & (np.asarray(path_labels, dtype=object) == require_path)
    if mask.sum() == 0:
        return float("nan")
    return float((errors[mask] > tau).mean())


def _mimicry_grid() -> list:
    # inject_min_rate_hz may be lowered vs default 15 so thinning can meet the
    # window objective (≥2 inject frames) without a 15Hz floor that keeps recon>τ
    # on sensitive detectors. Still selected on select_df only (not score seed).
    return [
        None,  # identity (no morph)
        {"keep_frac": 0.6, "spoof_blend": 0.0, "byte_strength": 0.5},
        {"keep_frac": 0.35, "spoof_blend": 0.15, "byte_strength": 0.8},
        {"keep_frac": 0.2, "spoof_blend": 0.3, "byte_strength": 1.0},
        {"keep_frac": 0.25, "spoof_blend": 0.0, "byte_strength": 1.0, "inject_min_rate_hz": 5.0},
        {"keep_frac": 0.15, "spoof_blend": 0.0, "byte_strength": 1.0, "inject_min_rate_hz": 3.0},
        {"keep_frac": 0.1, "spoof_blend": 0.0, "byte_strength": 1.0, "inject_min_rate_hz": 2.0},
    ]


def _score_mimicry_params(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    family: str,
    params: Optional[dict],
    nb: dict,
    rng: np.random.Generator,
) -> tuple[float, Optional[tuple]]:
    """Return (score, packed_arrays_or_None). Lower score is better."""
    n_err, n_feat, _, _ = _window_errors(model, scaler, attack_df, family)
    if params is None:
        e_err, e_feat, retained = n_err, n_feat, np.ones(len(n_err))
        m = len(n_err)
        if m == 0:
            return float("inf"), None
        score = float(np.mean(e_err))
        viol = [0] * m
        return score, (n_err[:m], e_err[:m], n_feat[:m], e_feat[:m], retained[:m], viol)

    thinned = thin_attack_stream(
        attack_df, family, rng, normal_byte_means=nb, **params,
    )
    e_err, e_feat, retained, _ = _window_errors(model, scaler, thinned, family)
    if len(e_err) == 0:
        return float("inf"), None
    m = min(len(n_err), len(e_err))
    score = float(np.mean(e_err[:m]))
    ret_mean = float(np.mean(retained[:m])) if m else 0.0
    # Soft retention signal only (tiered selection in _select_mimicry_params is authoritative)
    if ret_mean < 0.7:
        score += 10.0 * (0.7 - ret_mean)
    viol = [constraint_violations(e_feat[i]) > 0 for i in range(m)]
    return score, (n_err[:m], e_err[:m], n_feat[:m], e_feat[:m], retained[:m], viol)


def _select_mimicry_params(
    model: Autoencoder,
    scaler: FeatureScaler,
    select_stream: pd.DataFrame,
    family: str,
    nb: dict,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Choose grid params on select_stream without hard-penalizing into identity.

    Tier 1: among candidates with obj retention ≥ 0.7, minimize mean recon.
    Tier 2: if empty, among *non-identity* morphs with retention ≥ 0.5, maximize
            retention then minimize recon (avoids seed0 identity collapse when
            select-stream retention sits just under 0.7).
    Tier 3: identity / lowest recon fallback.
    """
    scored: list[tuple[Optional[dict], float, float]] = []
    for params in _mimicry_grid():
        score, packed = _score_mimicry_params(
            model, scaler, select_stream, family, params, nb, rng,
        )
        if packed is None or score == float("inf"):
            continue
        # Undo hard penalty inside _score for tiering; use raw recon + ret
        _n_e, e_e, _n_f, _e_f, ret, _viol = packed
        mean_recon = float(np.mean(e_e)) if len(e_e) else float("inf")
        ret_mean = float(np.mean(ret)) if len(ret) else 0.0
        scored.append((params, mean_recon, ret_mean))
    if not scored:
        return None
    # Prefer real morphs; identity always has ret=1 and must not dominate tier1.
    tier1 = [s for s in scored if s[0] is not None and s[2] >= 0.7]
    if tier1:
        return min(tier1, key=lambda s: s[1])[0]
    tier2 = [s for s in scored if s[0] is not None and s[2] >= 0.5]
    if tier2:
        # max retention, then min recon
        return min(tier2, key=lambda s: (-s[2], s[1]))[0]
    # Last resort: identity (or whatever has lowest recon)
    return min(scored, key=lambda s: s[1])[0]


def mimicry_evade(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    family: str,
    seed: int = 0,
    select_df: Optional[pd.DataFrame] = None,
) -> EvasionResult:
    """Black-box mimicry: select grid params on ``select_df``, score on ``attack_df``.

    When ``select_df`` is provided (recommended), grid-search uses that stream so
    parameter choice is not fit on the evaluated attack seed (selection-bias guard).
    If omitted, falls back to selecting on ``attack_df`` (legacy / unit tests).
    """
    rng_select = np.random.default_rng(seed)
    rng_score = np.random.default_rng(seed + 7919)
    nb = estimate_normal_byte_means(normal_df)
    select_stream = select_df if select_df is not None else attack_df

    best_params = _select_mimicry_params(
        model, scaler, select_stream, family, nb, rng_select,
    )

    # Fresh RNG for scoring morph so thinning draws are independent of select
    _, packed = _score_mimicry_params(
        model, scaler, attack_df, family, best_params, nb, rng_score,
    )
    if packed is None:
        n_err, n_feat, _, _ = _window_errors(model, scaler, attack_df, family)
        return _pack(
            "mimicry", family, n_err, n_err, n_feat, n_feat,
            np.ones(len(n_err)), [0] * len(n_err),
        )
    n_e, e_e, n_f, e_f, ret, viol = packed
    return _pack("mimicry", family, n_e, e_e, n_f, e_f, ret, viol)


def _free_byte_mask(window_df: pd.DataFrame, family: str) -> np.ndarray:
    """Boolean mask (N, 8): True where payload bytes may be optimized by PGD.

    Objective-critical bytes stay fixed (spoof phys b0/b1; full replay payload
    on the replayed ID). On-bus adversary may also morph free mid-bytes of
    in-TOP_IDS background frames in the window so gradients reach byte_mean/std
    (INJECT_ID 0x7FF is intentionally outside TOP_IDS / feature byte stats;
    free background TOP_IDS mid-bytes provide the differentiable lever).
    """
    family = family.lower()
    n = len(window_df)
    mask = np.zeros((n, 8), dtype=bool)
    if n == 0:
        return mask
    ids = window_df["can_id"].to_numpy(dtype=np.int64)
    fams = window_df["attack_family"].to_numpy()
    top_set = set(TOP_IDS)

    for i in range(n):
        cid = int(ids[i])
        af = str(fams[i]) if fams[i] else ""
        # --- objective-critical freezes ---
        if family == "spoof" and cid == 0x100 and af == "spoof":
            mask[i, 2:8] = True  # b0/b1 = spoofed phys
            continue
        if family == "replay" and cid == 0x100 and af == "replay":
            continue  # entire replayed payload fixed
        if family == "flood" and af == "flood":
            mask[i, :] = True
            continue
        if family == "injection" and cid == INJECT_ID:
            mask[i, 2:8] = True  # free inject padding (may not affect TOP_IDS stats)
            continue
        if family == "drop" and cid in top_set:
            mask[i, :] = True
            continue
        # Background / remaining frames with known IDs: free mid-bytes
        if cid in top_set:
            mask[i, 2:8] = True
    return mask


def torch_byte_mean_std(
    payloads: torch.Tensor,
    can_ids: np.ndarray,
    top_ids: tuple[int, ...] = TOP_IDS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable per-ID byte mean and std over an (N, 8) payload tensor.

    Matches ``extract_window_features`` aggregation: flatten all bytes of frames
    with that ID, then mean / std (std=0 if fewer than 2 values).
    """
    k = len(top_ids)
    device = payloads.device
    dtype = payloads.dtype
    byte_mean = torch.zeros(k, device=device, dtype=dtype)
    byte_std = torch.zeros(k, device=device, dtype=dtype)
    id_to_idx = {cid: i for i, cid in enumerate(top_ids)}
    for cid, idx in id_to_idx.items():
        sel = np.where(can_ids == cid)[0]
        if sel.size == 0:
            continue
        flat = payloads[torch.from_numpy(sel.astype(np.int64)).to(device)].reshape(-1)
        byte_mean[idx] = flat.mean()
        if flat.numel() > 1:
            byte_std[idx] = flat.std(unbiased=False)
    return byte_mean, byte_std


def assemble_torch_features(
    base_feat: np.ndarray,
    payloads: torch.Tensor,
    can_ids: np.ndarray,
    k: int = K,
) -> torch.Tensor:
    """Replace byte_mean/std slots in ``base_feat`` with differentiable values."""
    byte_mean, byte_std = torch_byte_mean_std(payloads, can_ids)
    # Layout: counts(k+1) | iat_mean(k) | iat_std(k) | byte_mean(k) | byte_std(k) | ent
    b0 = 3 * k + 1
    parts = [
        torch.as_tensor(base_feat[:b0], dtype=payloads.dtype, device=payloads.device),
        byte_mean,
        byte_std,
        torch.as_tensor(base_feat[-1:], dtype=payloads.dtype, device=payloads.device),
    ]
    return torch.cat(parts, dim=0)


def pgd_frame_bytes(
    model: Autoencoder,
    scaler: FeatureScaler,
    window_df: pd.DataFrame,
    family: str,
    *,
    steps: int = 100,
    step_size: float = 4.0,
    eps: float = 48.0,
) -> tuple[pd.DataFrame, str]:
    """Frame-space sign-PGD on free bytes; Π_C = L∞ then [0,255] every step.

    Returns (morphed_window_df, path_label). Path is ``true_pgd`` only when at
    least one free byte lies on a TOP_IDS frame (differentiable lever); otherwise
    ``naive_fallback``. Scoring must use the original numpy extractor after a
    single final round — never the torch forward alone.
    """
    w = window_df.copy().reset_index(drop=True)
    free = _free_byte_mask(w, family)
    ids_arr = w["can_id"].to_numpy(dtype=np.int64)
    top_rows = np.isin(ids_arr, list(TOP_IDS))
    effective = free & top_rows[:, None]
    if not effective.any():
        return w, PATH_NAIVE_FALLBACK

    ts, ids, data = frames_from_df(w)
    base_feat = project_features(extract_window_features(ts, ids, data))
    data0 = data.astype(np.float64).copy()
    free_t = torch.from_numpy(free.astype(np.float32))
    x0 = torch.from_numpy(data0.astype(np.float32))
    x = x0.clone().requires_grad_(True)

    mean_t = torch.from_numpy(scaler.mean.astype(np.float32))
    std_np = np.where(scaler.std < 1e-8, 1.0, scaler.std).astype(np.float32)
    std_t = torch.from_numpy(std_np)
    model.eval()

    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        feat = assemble_torch_features(base_feat, x, ids)
        feat_s = (feat - mean_t) / std_t
        recon = model(feat_s.unsqueeze(0)).squeeze(0)
        loss = (recon - feat_s).pow(2).mean()
        loss.backward()
        with torch.no_grad():
            grad = x.grad
            # Sign-PGD: step_size is in byte levels (interpretable)
            x_step = x - step_size * grad.sign() * free_t
            # Π_C order: L∞ ball around x0, then [0, 255]; non-free frozen
            delta = (x_step - x0).clamp(-float(eps), float(eps))
            x_proj = torch.where(
                free_t.bool(),
                (x0 + delta).clamp(0.0, 255.0),
                x0,
            )
            x = x_proj.detach().requires_grad_(True)

    # Single final quantization of free bytes only; caller re-extracts with numpy
    final = x.detach().cpu().numpy()
    for i in range(len(w)):
        for b in range(8):
            if free[i, b]:
                w.at[i, f"b{b}"] = int(np.clip(int(round(final[i, b])), 0, 255))
    for b in range(8):
        w[f"b{b}"] = w[f"b{b}"].astype(int).clip(0, 255)
    return w, PATH_TRUE_PGD


def pgd_evade(
    model: Autoencoder,
    scaler: FeatureScaler,
    attack_df: pd.DataFrame,
    normal_df: pd.DataFrame,
    family: str,
    seed: int = 0,
    steps: int = 100,
    lr: float = 4.0,
    eps: float = 48.0,
) -> EvasionResult:
    """Frame-space PGD on free attack-frame bytes; two-pass re-extract verify.

    Path labels (per window):
      - true_pgd: frame-space PGD ran (Π_C in-loop on bytes); these count as PGD
      - naive_fallback: no free bytes to optimize; raw naive window kept
    Thinning is **not** part of this path (mimicry-only). Actual recon MSE is
    reported even when worse than naive (no clamp).
    """
    del normal_df, seed  # byte targets unused; signature kept for call-site stability
    model.eval()

    n_errs, e_errs, n_feats, e_feats, retained, viols, paths = (
        [], [], [], [], [], [], [],
    )

    labels = attack_df["label"].to_numpy()
    for s, e in window_indices(len(attack_df)):
        if float((labels[s:e] == "attack").mean()) < 0.02:
            continue
        nw = attack_df.iloc[s:e].reset_index(drop=True)
        ts_n, ids_n, data_n = frames_from_df(nw)
        feat_n = project_features(extract_window_features(ts_n, ids_n, data_n))
        err_n = float(reconstruction_errors(model, scaler.transform(feat_n[None, :]))[0])

        w_adv, path = pgd_frame_bytes(
            model, scaler, nw, family, steps=steps, step_size=lr, eps=eps,
        )
        ts_a, ids_a, data_a = frames_from_df(w_adv)
        feat_a = project_features(extract_window_features(ts_a, ids_a, data_a))
        err_a = float(reconstruction_errors(model, scaler.transform(feat_a[None, :]))[0])

        n_errs.append(err_n)
        n_feats.append(feat_n)
        e_errs.append(err_a)
        e_feats.append(feat_a)
        retained.append(1.0 if objective_retained(w_adv, family) else 0.0)
        viols.append(1.0 if constraint_violations(feat_a) > 0 else 0.0)
        paths.append(path)
        if len(e_errs) >= 40:
            break

    if not e_errs:
        n_err, n_feat, ret, _ = _window_errors(model, scaler, attack_df, family)
        path_labels = np.full(len(n_err), PATH_NAIVE_FALLBACK, dtype=object)
        return _pack(
            "pgd", family, n_err, n_err, n_feat, n_feat, ret,
            [0] * len(n_err), path_labels=path_labels,
        )

    return _pack(
        "pgd", family,
        np.asarray(n_errs), np.asarray(e_errs),
        np.stack(n_feats), np.stack(e_feats),
        retained, viols,
        path_labels=np.asarray(paths, dtype=object),
    )


def run_evasion_suite(
    model: Autoencoder,
    scaler: FeatureScaler,
    tau: float,
    seed: int = 0,
    duration_s: float = 10.0,
    tau_fpr1: float | None = None,
) -> dict:
    from src.generate import ATTACK_FAMILIES, generate_attack, generate_normal
    from src.evaluate import detection_rate

    normal_df = generate_normal(duration_s=max(duration_s, 20.0), seed=seed + 50)
    tables = {}
    for i, fam in enumerate(ATTACK_FAMILIES):
        kwargs = {}
        if fam == "spoof":
            kwargs["target_phys"] = 115.0
        # Separate seeds: select mimicry params on one stream, score on another
        select_seed = seed + 20 + i
        score_seed = seed + 120 + i
        assert select_seed != score_seed, "mimicry select/score seeds must differ"
        atk_select = generate_attack(
            fam, duration_s=duration_s, seed=select_seed, **kwargs,
        )
        atk = generate_attack(
            fam, duration_s=duration_s, seed=score_seed, **kwargs,
        )
        mim = mimicry_evade(
            model, scaler, atk, normal_df, fam, seed=seed + 100 + i,
            select_df=atk_select,
        )
        pgd = pgd_evade(model, scaler, atk, normal_df, fam, seed=seed + 200 + i)

        # Ungated (secondary): all windows
        tpr_naive_ungated = detection_rate(mim.naive_errors, tau)
        tpr_mim_ungated = detection_rate(mim.evasive_errors, tau)
        tpr_pgd_ungated = detection_rate(pgd.evasive_errors, tau)

        # Primary: objective-retained gate (spirit of mimicry >=0.7 retention).
        # PGD primary further requires path == true_pgd so thinning is not credited.
        tpr_naive = gated_detection_rate(mim.naive_errors, tau, mim.retained_mask)
        # Naive matched to PGD true_pgd ∩ retained windows for fair ΔTPR
        tpr_naive_pgd_gate = gated_detection_rate(
            pgd.naive_errors, tau, pgd.retained_mask,
            path_labels=pgd.path_labels, require_path=PATH_TRUE_PGD,
        )
        tpr_mim = gated_detection_rate(mim.evasive_errors, tau, mim.retained_mask)
        tpr_pgd = gated_detection_rate(
            pgd.evasive_errors, tau, pgd.retained_mask,
            path_labels=pgd.path_labels, require_path=PATH_TRUE_PGD,
        )

        # Matched 1% FPR operating point (from fresh-NORMAL ROC), if provided
        if tau_fpr1 is not None and tau_fpr1 == tau_fpr1:
            tpr_naive_fpr1 = gated_detection_rate(mim.naive_errors, tau_fpr1, mim.retained_mask)
            tpr_mim_fpr1 = gated_detection_rate(mim.evasive_errors, tau_fpr1, mim.retained_mask)
            tpr_naive_pgd_fpr1 = gated_detection_rate(
                pgd.naive_errors, tau_fpr1, pgd.retained_mask,
                path_labels=pgd.path_labels, require_path=PATH_TRUE_PGD,
            )
            tpr_pgd_fpr1 = gated_detection_rate(
                pgd.evasive_errors, tau_fpr1, pgd.retained_mask,
                path_labels=pgd.path_labels, require_path=PATH_TRUE_PGD,
            )
        else:
            tpr_naive_fpr1 = tpr_mim_fpr1 = tpr_naive_pgd_fpr1 = tpr_pgd_fpr1 = float("nan")

        # Mean recon on true_pgd windows only (PGD headline honesty)
        pgd_mask = (pgd.path_labels == PATH_TRUE_PGD) if len(pgd.path_labels) else np.zeros(0, dtype=bool)
        if pgd_mask.any():
            mean_recon_pgd_true = float(np.mean(pgd.evasive_errors[pgd_mask]))
            mean_recon_naive_pgd_matched = float(np.mean(pgd.naive_errors[pgd_mask]))
        else:
            mean_recon_pgd_true = float("nan")
            mean_recon_naive_pgd_matched = float("nan")

        def _delta(a, b):
            if a != a or b != b:
                return float("nan")
            return float(a - b)

        tables[fam] = {
            "n_windows": int(len(mim.naive_errors)),
            "n_windows_pgd": int(len(pgd.naive_errors)),
            "n_true_pgd": int(pgd.path_counts.get(PATH_TRUE_PGD, 0)),
            "n_thin": int(pgd.path_counts.get(PATH_THIN, 0)),
            "n_naive_fallback": int(pgd.path_counts.get(PATH_NAIVE_FALLBACK, 0)),
            "path_counts_pgd": dict(pgd.path_counts),
            "mean_recon_naive": mim.mean_naive_error,
            "mean_recon_mimicry": mim.mean_evasive_error,
            # Headline PGD recon: true_pgd path only (nan if none)
            "mean_recon_pgd": mean_recon_pgd_true,
            "mean_recon_pgd_all_paths": pgd.mean_evasive_error,
            "mean_recon_naive_pgd_matched": mean_recon_naive_pgd_matched,
            "frac_improved_mimicry": mim.frac_improved,
            "frac_fallback_mimicry": mim.frac_fallback,
            "frac_improved_pgd": pgd.frac_improved,
            "frac_fallback_pgd": pgd.frac_fallback,
            # Primary (objective-gated; PGD also path-gated)
            "tpr_naive": tpr_naive if tpr_naive == tpr_naive else tpr_naive_ungated,
            "tpr_naive_pgd_matched": tpr_naive_pgd_gate,
            "tpr_mimicry": tpr_mim,
            "tpr_pgd": tpr_pgd,
            "delta_tpr_mimicry": _delta(tpr_naive if tpr_naive == tpr_naive else tpr_naive_ungated, tpr_mim),
            "delta_tpr_pgd": _delta(tpr_naive_pgd_gate, tpr_pgd),
            # Matched 1% FPR (ROC) operating point
            "tpr_naive_fpr1": tpr_naive_fpr1 if tpr_naive_fpr1 == tpr_naive_fpr1 else tpr_naive_ungated,
            "tpr_mimicry_fpr1": tpr_mim_fpr1,
            "tpr_naive_pgd_matched_fpr1": tpr_naive_pgd_fpr1,
            "tpr_pgd_fpr1": tpr_pgd_fpr1,
            "delta_tpr_mimicry_fpr1": _delta(
                tpr_naive_fpr1 if tpr_naive_fpr1 == tpr_naive_fpr1 else float("nan"), tpr_mim_fpr1,
            ),
            "delta_tpr_pgd_fpr1": _delta(tpr_naive_pgd_fpr1, tpr_pgd_fpr1),
            # Secondary ungated
            "tpr_naive_ungated": tpr_naive_ungated,
            "tpr_mimicry_ungated": tpr_mim_ungated,
            "tpr_pgd_ungated": tpr_pgd_ungated,
            "delta_tpr_mimicry_ungated": _delta(tpr_naive_ungated, tpr_mim_ungated),
            "delta_tpr_pgd_ungated": _delta(tpr_naive_ungated, tpr_pgd_ungated),
            "obj_retention_mimicry": mim.objective_retention,
            "obj_retention_pgd": pgd.objective_retention,
            "constraint_viol_mimicry": mim.constraint_violation_rate,
            "constraint_viol_pgd": pgd.constraint_violation_rate,
        }
    return tables
