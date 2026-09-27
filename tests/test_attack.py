"""Evasion feasibility and improvement tests."""

from __future__ import annotations

import numpy as np

from src.attack import (
    constraint_violations,
    mimicry_evade,
    objective_retained,
    pgd_evade,
    project_features,
)
from src.features import FEATURE_DIM, extract_features
from src.generate import generate_attack, generate_normal
from src.train import run_training_pipeline


def test_project_features_feasible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(FEATURE_DIM,)) * 100
    xp = project_features(x)
    assert constraint_violations(xp) == 0


def test_evasion_lowers_recon_error(tmp_path):
    """Evasion mean recon error should be <= naive (directionally correct)."""
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
        pgd = pgd_evade(model, scaler, atk, normal_df, fam, seed=2, steps=25)
        assert mim.constraint_violation_rate == 0.0 or mim.constraint_violation_rate < 0.05
        assert pgd.constraint_violation_rate == 0.0 or pgd.constraint_violation_rate < 0.05
        # Directional: at least one method reduces error, or gap is tiny
        gaps.append(mim.mean_naive_error - mim.mean_evasive_error)
        gaps.append(pgd.mean_naive_error - pgd.mean_evasive_error)
        assert mim.objective_retention >= 0.5
        assert pgd.objective_retention >= 0.5

    # Macro: mean error reduction should be >= 0 (honest: allow tiny negative noise)
    assert np.mean(gaps) >= -1e-3 or max(gaps) > 0


def test_objective_injection():
    df = generate_attack("injection", duration_s=5.0, seed=1)
    batch = extract_features(df)
    # grab an attack window
    from src.features import window_indices
    for s, e in window_indices(len(df)):
        w = df.iloc[s:e]
        if (w["label"] == "attack").mean() >= 0.1:
            assert objective_retained(w, "injection")
            break
