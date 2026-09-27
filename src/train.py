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
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split NORMAL windows into train / val / test (disjoint)."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
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


def run_training_pipeline(
    seed: int = 0,
    duration_s: float = 60.0,
    out_dir: str | Path = "results",
    epochs: int = 80,
) -> dict:
    """End-to-end: generate NORMAL → features → train → τ=p99(val) → save."""
    set_all_seeds(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_normal(duration_s=duration_s, seed=seed)
    batch = extract_features(df, window_n=WINDOW_N, stride=WINDOW_STRIDE)
    assert (batch.y == 0).all(), "NORMAL stream must yield only normal windows"
    X = batch.X
    X_train, X_val, X_test = split_normal_windows(X, seed=seed)

    scaler = FeatureScaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = build_model(input_dim=FEATURE_DIM, seed=seed)
    history = train_autoencoder(
        model, X_train_s, X_val_s, epochs=epochs, seed=seed,
    )

    val_err = reconstruction_errors(model, X_val_s)
    tau = calibrate_threshold(val_err, percentile=99.0)
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
        "fpr_test_normal": fpr_test,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
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
