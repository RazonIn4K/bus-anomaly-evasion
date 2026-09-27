"""Train the AE on NORMAL windows only; calibrate threshold at p99 of a
held-out NORMAL split. Save model + threshold. No leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features import (
    FEATURE_DIM,
    FeatureScaler,
    WINDOW_N,
    WINDOW_STRIDE,
    effective_independent_windows,
    extract_features,
)
from src.generate import generate_normal
from src.model import Autoencoder, build_model


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_normal_windows(
    X: np.ndarray,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.20,
    temporal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split NORMAL windows into train / val / test (disjoint).

    Default ``temporal=True``: contiguous time order (early→train, mid→val for τ,
    late→test FPR). Prefer temporal held-out NORMAL for τ calibration per DESIGN.
    ``seed`` is unused when temporal but kept for API stability.
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if temporal:
        return X[:n_train], X[n_train : n_train + n_val], X[n_train + n_val :]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return X[train_idx], X[val_idx], X[test_idx]


def train_autoencoder(
    model: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> dict:
    """Train on NORMAL only; early-stop on NORMAL val reconstruction MSE."""
    set_all_seeds(seed)
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            n += len(batch)
        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_t), val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_ran"] = len(history["train_loss"])
    return history


@torch.no_grad()
def reconstruction_errors(model: Autoencoder, X: np.ndarray, device: Optional[torch.device] = None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    t = torch.from_numpy(X.astype(np.float32)).to(device)
    err = model.reconstruction_error(t, reduction="none")
    return err.cpu().numpy()


def calibrate_threshold(errors: np.ndarray, percentile: float = 99.0) -> float:
    """τ = percentile of reconstruction MSE on held-out NORMAL only."""
    return float(np.percentile(errors, percentile))


def _block_seeds(seed: int, n_blocks: int = 12) -> list[int]:
    """Deterministic NORMAL block seeds, disjoint from attack/fresh offsets.

    Avoids collisions with generate_attack internal ``seed+1000`` backgrounds,
    pipeline fresh ``seed+777``, and mimicry select/score offsets.
    """
    return [10_000 + seed * 100 + i for i in range(n_blocks)]


def run_training_pipeline(
    seed: int = 0,
    duration_s: float = 200.0,
    out_dir: str | Path = "results",
    epochs: int = 60,
    n_blocks: int = 12,
    block_duration_s: float = 200.0,
    n_train_blocks: int = 6,
    n_earlystop_blocks: int = 1,
    n_calib_blocks: int = 3,
    n_test_blocks: int = 2,
) -> dict:
    """End-to-end: independent NORMAL blocks → train → τ=p99(calib) → save.

    Ordered temporal split of **independent** ``generate_normal`` blocks (each
    reset+walk like fresh eval), not one uninterrupted mega-stream. This matches
    the fresh-NORMAL distribution that FPR is measured on, avoiding mid-stream
    vs reset mismatch while keeping τ = p99 of held-out NORMAL only.

    Block roles (chronological, predetermined — never chosen by scores):
      train (scaler+AE) | early-stop | calibration (τ) | NORMAL test (FPR check)
    Features are extracted **per block** then concatenated (no cross-block windows).
    """
    set_all_seeds(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compact layout for unit tests that pass a short duration_s without block kwargs
    if duration_s < 60.0 and n_blocks == 12 and block_duration_s == 200.0:
        n_blocks = 4
        block_duration_s = max(float(duration_s), 6.0)
        n_train_blocks, n_earlystop_blocks, n_calib_blocks, n_test_blocks = 1, 1, 1, 1

    assert n_train_blocks + n_earlystop_blocks + n_calib_blocks + n_test_blocks == n_blocks

    seeds = _block_seeds(seed, n_blocks=n_blocks)

    def _feats(block_seed: int) -> np.ndarray:
        df = generate_normal(duration_s=block_duration_s, seed=block_seed)
        batch = extract_features(df, window_n=WINDOW_N, stride=WINDOW_STRIDE)
        assert (batch.y == 0).all()
        return batch.X

    train_seeds = seeds[:n_train_blocks]
    early_seeds = seeds[n_train_blocks : n_train_blocks + n_earlystop_blocks]
    calib_seeds = seeds[
        n_train_blocks + n_earlystop_blocks :
        n_train_blocks + n_earlystop_blocks + n_calib_blocks
    ]
    test_seeds = seeds[n_train_blocks + n_earlystop_blocks + n_calib_blocks :]

    X_train = np.concatenate([_feats(s) for s in train_seeds], axis=0)
    X_early = np.concatenate([_feats(s) for s in early_seeds], axis=0)
    X_calib = np.concatenate([_feats(s) for s in calib_seeds], axis=0)
    X_test = np.concatenate([_feats(s) for s in test_seeds], axis=0)

    scaler = FeatureScaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_early_s = scaler.transform(X_early)
    X_calib_s = scaler.transform(X_calib)
    X_test_s = scaler.transform(X_test)

    model = build_model(input_dim=FEATURE_DIM, seed=seed)
    history = train_autoencoder(
        model, X_train_s, X_early_s, epochs=epochs, patience=15, seed=seed,
    )

    # τ from calibration blocks ONLY (not early-stop) — DESIGN: held-out NORMAL p99
    calib_err = reconstruction_errors(model, X_calib_s)
    tau = calibrate_threshold(calib_err, percentile=99.0)
    test_err = reconstruction_errors(model, X_test_s)
    fpr_test = float((test_err > tau).mean()) if len(test_err) else float("nan")

    model_path = out_dir / f"ae_seed{seed}.pt"
    meta_path = out_dir / f"ae_seed{seed}_meta.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": FEATURE_DIM,
            "seed": seed,
        },
        model_path,
    )
    meta = {
        "seed": seed,
        "tau": tau,
        "percentile": 99.0,
        "tau_calibration_split": "ordered_independent_NORMAL_calib_blocks",
        "fpr_split_description": (
            "FPR on held-out independent NORMAL test blocks; "
            "τ = p99 of pooled independent NORMAL calibration blocks "
            "(predetermined seeds; never chosen by scores; never attack)"
        ),
        "fpr_test_normal": fpr_test,
        "n_train": int(len(X_train)),
        "n_earlystop": int(len(X_early)),
        "n_val": int(len(X_calib)),  # calib = τ source
        "n_val_effective_indep": int(effective_independent_windows(len(X_calib))),
        "n_calib_blocks": int(n_calib_blocks),
        "n_test": int(len(X_test)),
        "n_test_effective_indep": int(effective_independent_windows(len(X_test))),
        "block_duration_s": float(block_duration_s),
        "n_blocks": int(n_blocks),
        "block_seeds": {
            "train": train_seeds,
            "earlystop": early_seeds,
            "calib": calib_seeds,
            "test": test_seeds,
        },
        "feature_dim": FEATURE_DIM,
        "scaler": scaler.to_dict(),
        "history": {
            "best_val_loss": history["best_val_loss"],
            "epochs_ran": history["epochs_ran"],
        },
        "model_path": str(model_path),
        "disclaimer": "synthetic only",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {
        "model": model,
        "scaler": scaler,
        "tau": tau,
        "meta": meta,
        "X_test_s": X_test_s,
        "test_err": test_err,
        "model_path": model_path,
        "meta_path": meta_path,
    }



def load_trained(meta_path: str | Path, device: Optional[torch.device] = None) -> dict:
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    model = build_model(input_dim=meta["feature_dim"], seed=meta["seed"])
    ckpt = torch.load(meta["model_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model = model.to(device)
    scaler = FeatureScaler.from_dict(meta["scaler"])
    return {"model": model, "scaler": scaler, "tau": meta["tau"], "meta": meta}


if __name__ == "__main__":
    result = run_training_pipeline(seed=0, duration_s=60.0, epochs=60)
    print(json.dumps({k: result["meta"][k] for k in ("tau", "fpr_test_normal", "n_train", "n_val", "n_test", "history")}, indent=2))
