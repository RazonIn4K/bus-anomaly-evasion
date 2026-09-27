"""Windowing + feature extraction.

Fixed-frame-count windows -> compact feature vector (per-ID counts, inter-arrival
mean/std, byte value stats, ID entropy).

Single code path for train / eval / attack. K locked to the generator ID set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.generate import ALL_IDS, FRAME_COLUMNS

# Locked windowing (DESIGN.md §7)
WINDOW_N = 50
WINDOW_STRIDE = 25  # 50% overlap for training

# Locked K: full generator ID set (15) + "other" bucket
TOP_IDS: tuple[int, ...] = tuple(ALL_IDS)
K = len(TOP_IDS)


@dataclass
class FeatureScaler:
    """Simple z-score scaler fit on NORMAL training features only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        std = np.where(self.std < 1e-8, 1.0, self.std)
        return (X - self.mean) / std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    @staticmethod
    def fit(X: np.ndarray) -> "FeatureScaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return FeatureScaler(mean=mean, std=std)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "FeatureScaler":
        return FeatureScaler(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
        )


def feature_dim(k: int = K) -> int:
    """counts(K+1) + iat_mean(K) + iat_std(K) + byte_mean(K) + byte_std(K) + entropy(1)."""
    return (k + 1) + k + k + k + k + 1


FEATURE_DIM = feature_dim(K)


def feature_names(ids: Sequence[int] = TOP_IDS) -> list[str]:
    names: list[str] = []
    for cid in ids:
        names.append(f"count_{cid:#x}")
    names.append("count_other")
    for cid in ids:
        names.append(f"iat_mean_{cid:#x}")
    for cid in ids:
        names.append(f"iat_std_{cid:#x}")
    for cid in ids:
        names.append(f"byte_mean_{cid:#x}")
    for cid in ids:
        names.append(f"byte_std_{cid:#x}")
    names.append("id_entropy")
    return names


def _id_entropy(ids: np.ndarray) -> float:
    if len(ids) == 0:
        return 0.0
    _, counts = np.unique(ids, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def extract_window_features(
    timestamps: np.ndarray,
    can_ids: np.ndarray,
    data: np.ndarray,
    top_ids: Sequence[int] = TOP_IDS,
) -> np.ndarray:
    """Extract a single feature vector from one window of frames.

    Parameters
    ----------
    timestamps : (N,) float64
    can_ids : (N,) int
    data : (N, 8) uint8 / float
    """
    top_ids = tuple(top_ids)
    k = len(top_ids)
    id_to_idx = {cid: i for i, cid in enumerate(top_ids)}

    counts = np.zeros(k + 1, dtype=np.float64)
    iat_mean = np.zeros(k, dtype=np.float64)
    iat_std = np.zeros(k, dtype=np.float64)
    byte_mean = np.zeros(k, dtype=np.float64)
    byte_std = np.zeros(k, dtype=np.float64)

    # group indices per ID
    for i, cid in enumerate(can_ids):
        if cid in id_to_idx:
            counts[id_to_idx[cid]] += 1
        else:
            counts[k] += 1

    data_f = data.astype(np.float64)
    for cid, idx in id_to_idx.items():
        mask = can_ids == cid
        n = int(mask.sum())
        if n == 0:
            continue
        ts = timestamps[mask]
        if n >= 2:
            iats = np.diff(ts)
            # guard non-positive (should not happen after sort)
            iats = np.maximum(iats, 1e-9)
            iat_mean[idx] = float(iats.mean())
            iat_std[idx] = float(iats.std()) if n >= 3 else 0.0
        elif n == 1:
            iat_mean[idx] = 0.0
            iat_std[idx] = 0.0
        payload = data_f[mask]  # (n, 8)
        flat = payload.ravel()
        byte_mean[idx] = float(flat.mean())
        byte_std[idx] = float(flat.std()) if flat.size > 1 else 0.0

    ent = _id_entropy(can_ids)
    vec = np.concatenate([counts, iat_mean, iat_std, byte_mean, byte_std, [ent]])
    assert vec.shape == (feature_dim(k),)
    return vec.astype(np.float64)


def frames_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull timestamp / can_id / data arrays from a frame DataFrame."""
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    ids = df["can_id"].to_numpy(dtype=np.int64)
    data = df[[f"b{i}" for i in range(8)]].to_numpy(dtype=np.float64)
    return ts, ids, data


def window_indices(
    n_frames: int,
    window_n: int = WINDOW_N,
    stride: int = WINDOW_STRIDE,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for sliding windows."""
    if n_frames < window_n:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    while start + window_n <= n_frames:
        out.append((start, start + window_n))
        start += stride
    return out


@dataclass
class WindowBatch:
    """Features + labels for a set of windows."""

    X: np.ndarray            # (M, D)
    y: np.ndarray            # (M,) 0=normal 1=attack
    family: np.ndarray       # (M,) str
    starts: np.ndarray       # (M,) start frame index
    ends: np.ndarray         # (M,) end frame index


def extract_features(
    df: pd.DataFrame,
    window_n: int = WINDOW_N,
    stride: int = WINDOW_STRIDE,
    top_ids: Sequence[int] = TOP_IDS,
    attack_majority: float = 0.02,
) -> WindowBatch:
    """Slide windows over ``df`` and extract features.

    A window is labeled attack if the fraction of frames with label=='attack'
    is >= ``attack_majority`` (default 2% so sparse injection still flags).
    Family = mode of non-empty attack_family among attack frames, else "".
    """
    ts, ids, data = frames_from_df(df)
    labels = df["label"].to_numpy()
    families = df["attack_family"].to_numpy()
    spans = window_indices(len(df), window_n=window_n, stride=stride)

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    fams: list[str] = []
    starts: list[int] = []
    ends: list[int] = []

    for s, e in spans:
        vec = extract_window_features(ts[s:e], ids[s:e], data[s:e], top_ids=top_ids)
        lab_slice = labels[s:e]
        fam_slice = families[s:e]
        attack_frac = float((lab_slice == "attack").mean())
        is_attack = attack_frac >= attack_majority
        if is_attack:
            atk_fams = [f for f in fam_slice if f]
            if atk_fams:
                # mode
                vals, cnts = np.unique(atk_fams, return_counts=True)
                fam = str(vals[cnts.argmax()])
            else:
                fam = "unknown"
            y = 1
        else:
            fam = ""
            y = 0
        Xs.append(vec)
        ys.append(y)
        fams.append(fam)
        starts.append(s)
        ends.append(e)

    if not Xs:
        D = feature_dim(len(top_ids))
        return WindowBatch(
            X=np.zeros((0, D), dtype=np.float64),
            y=np.zeros((0,), dtype=np.int64),
            family=np.array([], dtype=object),
            starts=np.zeros((0,), dtype=np.int64),
            ends=np.zeros((0,), dtype=np.int64),
        )

    return WindowBatch(
        X=np.stack(Xs, axis=0),
        y=np.asarray(ys, dtype=np.int64),
        family=np.asarray(fams, dtype=object),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
    )



def effective_independent_windows(
    n_windows: int,
    window_n: int = WINDOW_N,
    stride: int = WINDOW_STRIDE,
) -> int:
    """Overlap-aware count of approximately independent windows.

    With stride < window_n, consecutive windows share frames. Scale by
    ``stride / window_n`` so reported sample sizes are not inflated by 50% overlap.
    """
    if n_windows <= 0:
        return 0
    return max(1, int(round(n_windows * float(stride) / float(window_n))))

def reconstruct_window_dataframe(
    df: pd.DataFrame,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Return a copy of frames [start:end] for two-pass verify."""
    return df.iloc[start:end].copy().reset_index(drop=True)
