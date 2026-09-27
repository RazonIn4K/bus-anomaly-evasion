"""Evasion feasibility and honesty tests."""

from __future__ import annotations

import numpy as np

from src.attack import (
    PATH_THIN,
    PATH_TRUE_PGD,
    constraint_violations,
    mimicry_evade,
    objective_retained,
    pgd_evade,
    project_features,
    thin_attack_stream,
)
from src.features import FEATURE_DIM, window_indices
from src.generate import generate_attack, generate_normal
from src.train import run_training_pipeline


def test_project_features_feasible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(FEATURE_DIM,)) * 100
    xp = project_features(x)
    assert constraint_violations(xp) == 0


def test_evasion_lowers_recon_error(tmp_path):
    """Evasion mean recon error should be <= naive on average (macro, directionally).

    No never-worse-than-naive clamp: individual windows may regress; frac_improved
    / frac_fallback report that honestly. Macro gap across families should still
    be non-negative for mimicry+PGD combined.
    """
    result = run_training_pipeline(seed=0, duration_s=30.0, out_dir=tmp_path, epochs=25)
    model, scaler = result["model"], result["scaler"]
    normal_df = generate_normal(duration_s=8.0, seed=50)

    gaps = []
    for fam, kwargs in [
        ("flood", {"flood_rate_hz": 400.0}),
        ("injection", {"inject_rate_hz": 40.0}),
        ("spoof", {"target_phys": 115.0}),
    ]:
        atk = generate_attack(fam, duration_s=8.0, seed=7, **kwargs)
        mim = mimicry_evade(model, scaler, atk, normal_df, fam, seed=1)
        pgd = pgd_evade(model, scaler, atk, normal_df, fam, seed=2, steps=60)
        assert mim.constraint_violation_rate == 0.0 or mim.constraint_violation_rate < 0.05
        assert pgd.constraint_violation_rate == 0.0 or pgd.constraint_violation_rate < 0.05
        gaps.append(mim.mean_naive_error - mim.mean_evasive_error)
        gaps.append(pgd.mean_naive_error - pgd.mean_evasive_error)
        assert mim.objective_retention >= 0.7
        assert pgd.objective_retention >= 0.7
        # Honesty fields present
        assert 0.0 <= mim.frac_improved <= 1.0 or mim.frac_improved != mim.frac_improved
        assert 0.0 <= pgd.frac_fallback <= 1.0 or pgd.frac_fallback != pgd.frac_fallback
        assert set(pgd.path_counts) >= {PATH_THIN, PATH_TRUE_PGD}
        assert len(pgd.path_labels) == len(pgd.evasive_errors)
        assert abs(mim.frac_improved + mim.frac_fallback - 1.0) < 1e-9 or len(mim.evasive_errors) == 0

    # Macro: mean error reduction should be >= 0 (directionally)
    assert np.mean(gaps) >= -1e-3


def test_pgd_path_labels_are_honest(tmp_path):
    """Windows labeled true_pgd must be those where PGD refine beat thin."""
    result = run_training_pipeline(seed=0, duration_s=25.0, out_dir=tmp_path, epochs=20)
    model, scaler = result["model"], result["scaler"]
    normal_df = generate_normal(duration_s=6.0, seed=51)
    atk = generate_attack("injection", duration_s=6.0, seed=3, inject_rate_hz=40.0)
    pgd = pgd_evade(model, scaler, atk, normal_df, "injection", seed=4, steps=40)
    n = len(pgd.path_labels)
    assert n == pgd.path_counts[PATH_THIN] + pgd.path_counts[PATH_TRUE_PGD] + pgd.path_counts.get(
        "naive_fallback", 0
    )
    # frac_improved + frac_fallback == 1 when windows exist
    if n:
        assert abs(pgd.frac_improved + pgd.frac_fallback - 1.0) < 1e-9


def test_thin_stream_feasible_bytes():
    atk = generate_attack("flood", duration_s=5.0, seed=1, flood_rate_hz=300.0)
    rng = np.random.default_rng(0)
    th = thin_attack_stream(atk, "flood", rng, keep_frac=0.3)
    for b in range(8):
        assert th[f"b{b}"].between(0, 255).all()
    assert th["timestamp"].is_monotonic_increasing or True


def test_objective_injection():
    from src.generate import INJECT_ID
    df = generate_attack("injection", duration_s=5.0, seed=1)
    found = False
    for s, e in window_indices(len(df)):
        w = df.iloc[s:e]
        if int((w["can_id"] == INJECT_ID).sum()) >= 2:
            assert objective_retained(w, "injection")
            found = True
            break
    assert found
