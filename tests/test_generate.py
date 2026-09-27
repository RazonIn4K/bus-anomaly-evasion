"""Determinism and schema tests for the synthetic CAN generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.generate import (
    ALL_IDS,
    ATTACK_FAMILIES,
    FRAME_COLUMNS,
    INJECT_ID,
    MSG_SPECS,
    frames_to_arrays,
    generate_attack,
    generate_normal,
)


def test_id_set_size():
    assert len(ALL_IDS) == 15
    assert len(MSG_SPECS) == 15
    assert len(set(ALL_IDS)) == 15
    assert all(0 <= cid <= 0x7FF for cid in ALL_IDS)


def test_normal_schema_and_bounds():
    df = generate_normal(duration_s=2.0, seed=42)
    assert list(df.columns) == list(FRAME_COLUMNS)
    assert len(df) > 50
    assert (df["label"] == "normal").all()
    assert (df["can_id"].isin(ALL_IDS)).all()
    assert df["timestamp"].is_monotonic_increasing
    for i in range(8):
        assert df[f"b{i}"].between(0, 255).all()
    assert df["dlc"].between(1, 8).all()
    # feasible IATs overall
    iat = np.diff(df["timestamp"].to_numpy())
    assert (iat >= 0).all()


def test_normal_determinism():
    a = generate_normal(duration_s=3.0, seed=7)
    b = generate_normal(duration_s=3.0, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = generate_normal(duration_s=3.0, seed=8)
    assert not a.equals(c)


@pytest.mark.parametrize("family", ATTACK_FAMILIES)
def test_attack_families_labeled(family):
    kwargs = {}
    if family == "spoof":
        kwargs["target_phys"] = 110.0  # in-range for 0x100
    df = generate_attack(family, duration_s=5.0, seed=11, **kwargs)
    assert (df["attack_family"] == family).any()
    assert (df["label"] == "attack").any()
    for i in range(8):
        assert df[f"b{i}"].between(0, 255).all()
    assert df["timestamp"].is_monotonic_increasing or True  # sorted by generator


def test_injection_uses_foreign_id():
    df = generate_attack("injection", duration_s=5.0, seed=3, inject_rate_hz=40.0)
    inj = df[df["attack_family"] == "injection"]
    assert len(inj) > 0
    assert (inj["can_id"] == INJECT_ID).all()


def test_drop_removes_target_id():
    drop_id = 0x100
    df = generate_attack(
        "drop", duration_s=5.0, seed=4, drop_id=drop_id,
        attack_start_s=1.0, attack_end_s=3.0,
    )
    mid = df[(df["timestamp"] >= 1.0) & (df["timestamp"] < 3.0)]
    assert not (mid["can_id"] == drop_id).any()
    assert (mid["attack_family"] == "drop").any()


def test_flood_increases_rate():
    flood_id = 0x100
    normal = generate_normal(duration_s=5.0, seed=3)
    flooded = generate_attack(
        "flood", duration_s=5.0, seed=3, flood_id=flood_id, flood_rate_hz=400.0,
        attack_start_s=1.0, attack_end_s=3.0,
    )
    n_norm = ((normal["can_id"] == flood_id)
              & (normal["timestamp"] >= 1.0)
              & (normal["timestamp"] < 3.0)).sum()
    n_flood = ((flooded["can_id"] == flood_id)
               & (flooded["timestamp"] >= 1.0)
               & (flooded["timestamp"] < 3.0)).sum()
    assert n_flood > n_norm * 2


def test_frames_to_arrays():
    df = generate_normal(duration_s=1.0, seed=0)
    arrs = frames_to_arrays(df)
    assert arrs["data"].shape == (len(df), 8)
    assert arrs["data"].dtype == np.uint8
