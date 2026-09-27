"""Feature extraction tests."""

from __future__ import annotations

import numpy as np

from src.features import (
    FEATURE_DIM,
    K,
    WINDOW_N,
    WINDOW_STRIDE,
    extract_features,
    extract_window_features,
    feature_names,
    frames_from_df,
    window_indices,
)
from src.generate import generate_attack, generate_normal


def test_feature_dim_locked():
    assert K == 15
    assert FEATURE_DIM == (K + 1) + 4 * K + 1
    assert len(feature_names()) == FEATURE_DIM


def test_windowing_stride():
    spans = window_indices(200, window_n=WINDOW_N, stride=WINDOW_STRIDE)
    assert spans[0] == (0, 50)
    assert spans[1] == (25, 75)
    assert all(e - s == WINDOW_N for s, e in spans)


def test_extract_deterministic():
    df = generate_normal(duration_s=5.0, seed=1)
    a = extract_features(df)
    b = extract_features(df)
    np.testing.assert_allclose(a.X, b.X)
    assert a.X.shape[1] == FEATURE_DIM
    assert (a.y == 0).all()


def test_attack_windows_labeled():
    df = generate_attack("flood", duration_s=6.0, seed=3, flood_rate_hz=400.0)
    batch = extract_features(df)
    assert (batch.y == 1).any()
    assert (batch.family == "flood").any()


def test_single_window_bounds():
    df = generate_normal(duration_s=3.0, seed=0)
    ts, ids, data = frames_from_df(df.iloc[:WINDOW_N])
    vec = extract_window_features(ts, ids, data)
    assert vec.shape == (FEATURE_DIM,)
    assert np.isfinite(vec).all()
    assert abs(vec[: K + 1].sum() - WINDOW_N) < 1e-6
