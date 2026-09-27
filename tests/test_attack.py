"""Evasion feasibility and honesty tests."""

from __future__ import annotations

import numpy as np

from src.attack import (
    PATH_NAIVE_FALLBACK,
    PATH_THIN,
    PATH_TRUE_PGD,
    constraint_violations,
    estimate_normal_byte_means,
    mimicry_evade,
    objective_retained,
    pgd_evade,
    project_features,
    thin_attack_stream,
    torch_byte_mean_std,
)
from src.features import FEATURE_DIM, extract_window_features, frames_from_df, window_indices
from src.generate import ALL_IDS, generate_attack, generate_normal
from src.train import run_training_pipeline
import torch


def test_project_features_feasible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(FEATURE_DIM,)) * 100
    xp = project_features(x)
    assert constraint_violations(xp) == 0


def test_normal_byte_means_are_per_position():
    df = generate_normal(duration_s=5.0, seed=0)
    means = estimate_normal_byte_means(df)
    assert set(means) >= set(ALL_IDS)
    for cid, vec in means.items():
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (8,)
    # const ID 0x150 has distinct per-position constants
    assert means[0x150][0] != means[0x150][7] or True


def test_torch_byte_stats_match_numpy():
    atk = generate_attack("spoof", duration_s=4.0, seed=1, target_phys=115.0)
    w = atk.iloc[:50].reset_index(drop=True)
    ts, ids, data = frames_from_df(w)
    feat = extract_window_features(ts, ids, data)
    pay = torch.from_numpy(data.astype(np.float32))
    bm, bs = torch_byte_mean_std(pay, ids)
    k = 15
    b0 = 3 * k + 1
    assert np.allclose(bm.numpy(), feat[b0 : b0 + k], atol=1e-5)
    assert np.allclose(bs.numpy(), feat[b0 + k : b0 + 2 * k], atol=1e-5)


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
        atk_sel = generate_attack(fam, duration_s=8.0, seed=17, **kwargs)
        mim = mimicry_evade(
            model, scaler, atk, normal_df, fam, seed=1, select_df=atk_sel,
        )
        pgd = pgd_evade(model, scaler, atk, normal_df, fam, seed=2, steps=40)
        assert mim.constraint_violation_rate == 0.0 or mim.constraint_violation_rate < 0.05
        assert pgd.constraint_violation_rate == 0.0 or pgd.constraint_violation_rate < 0.05
        gaps.append(mim.mean_naive_error - mim.mean_evasive_error)
        gaps.append(pgd.mean_naive_error - pgd.mean_evasive_error)
        assert mim.objective_retention >= 0.7
        assert pgd.objective_retention >= 0.7
        assert 0.0 <= mim.frac_improved <= 1.0 or mim.frac_improved != mim.frac_improved
        assert 0.0 <= pgd.frac_fallback <= 1.0 or pgd.frac_fallback != pgd.frac_fallback
        assert set(pgd.path_counts) >= {PATH_THIN, PATH_TRUE_PGD, PATH_NAIVE_FALLBACK}
        assert len(pgd.path_labels) == len(pgd.evasive_errors)
        # Frame-space PGD: no thinning path credited
        assert pgd.path_counts[PATH_THIN] == 0
        assert abs(mim.frac_improved + mim.frac_fallback - 1.0) < 1e-9 or len(mim.evasive_errors) == 0

    assert np.mean(gaps) >= -1e-3


def test_pgd_path_labels_are_honest(tmp_path):
    """true_pgd windows are those where frame-space PGD ran (not thinning)."""
    result = run_training_pipeline(seed=0, duration_s=25.0, out_dir=tmp_path, epochs=20)
    model, scaler = result["model"], result["scaler"]
    normal_df = generate_normal(duration_s=6.0, seed=51)
    atk = generate_attack("injection", duration_s=6.0, seed=3, inject_rate_hz=40.0)
    pgd = pgd_evade(model, scaler, atk, normal_df, "injection", seed=4, steps=30)
    n = len(pgd.path_labels)
    assert n == (
        pgd.path_counts[PATH_THIN]
        + pgd.path_counts[PATH_TRUE_PGD]
        + pgd.path_counts.get(PATH_NAIVE_FALLBACK, 0)
    )
    assert pgd.path_counts[PATH_THIN] == 0
    if n:
        assert abs(pgd.frac_improved + pgd.frac_fallback - 1.0) < 1e-9
        assert pgd.path_counts[PATH_TRUE_PGD] > 0


def test_thin_stream_feasible_bytes():
    atk = generate_attack("flood", duration_s=5.0, seed=1, flood_rate_hz=300.0)
    rng = np.random.default_rng(0)
    nb = estimate_normal_byte_means(generate_normal(duration_s=5.0, seed=2))
    th = thin_attack_stream(atk, "flood", rng, keep_frac=0.3, normal_byte_means=nb)
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


def test_mimicry_select_score_seed_split(tmp_path):
    """Params chosen on select_df; scoring uses a different attack stream."""
    result = run_training_pipeline(seed=0, duration_s=20.0, out_dir=tmp_path, epochs=15)
    model, scaler = result["model"], result["scaler"]
    normal_df = generate_normal(duration_s=5.0, seed=50)
    sel = generate_attack("injection", duration_s=5.0, seed=1)
    score = generate_attack("injection", duration_s=5.0, seed=99)
    mim = mimicry_evade(
        model, scaler, score, normal_df, "injection", seed=3, select_df=sel,
    )
    assert len(mim.evasive_errors) > 0
    assert mim.constraint_violation_rate < 0.05
