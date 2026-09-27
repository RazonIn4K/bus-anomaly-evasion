"""Synthetic CAN-bus-style telemetry.

Emit labeled normal traffic and attack streams (injection, spoof, flood, drop,
replay). Normal = ~15 message IDs, each with a nominal period + jitter, carrying
bounded signals (random-walk physical values, counters, constants).

Frames: timestamp (float s), can_id (11-bit int), dlc (1-8), data (uint8[8]),
label (str), attack_family (str|None).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locked ID set (K = 15) — periods in seconds, payload roles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MsgSpec:
    can_id: int
    period_s: float
    dlc: int
    kind: str  # "phys" | "counter" | "const" | "mixed"
    phys_lo: float = 0.0
    phys_hi: float = 100.0
    phys_step: float = 0.5  # max random-walk step per tick


MSG_SPECS: tuple[MsgSpec, ...] = (
    MsgSpec(0x100, 0.010, 8, "phys", 0.0, 120.0, 1.0),   # engine RPM proxy
    MsgSpec(0x101, 0.010, 8, "phys", 0.0, 100.0, 0.8),   # throttle
    MsgSpec(0x110, 0.020, 8, "phys", 0.0, 250.0, 2.0),   # wheel speed FL
    MsgSpec(0x111, 0.020, 8, "phys", 0.0, 250.0, 2.0),   # wheel speed FR
    MsgSpec(0x120, 0.050, 8, "phys", -40.0, 150.0, 0.3), # coolant temp
    MsgSpec(0x130, 0.100, 8, "mixed", 0.0, 100.0, 0.5),  # battery / status
    MsgSpec(0x140, 0.050, 4, "counter"),                  # alive counter
    MsgSpec(0x150, 0.100, 8, "const"),                    # VIN fragment-ish
    MsgSpec(0x160, 0.020, 8, "phys", 0.0, 100.0, 1.5),   # brake pressure
    MsgSpec(0x170, 0.200, 8, "phys", 0.0, 360.0, 3.0),   # steering angle
    MsgSpec(0x180, 0.010, 8, "phys", 0.0, 50.0, 0.5),    # accel pedal
    MsgSpec(0x190, 0.050, 8, "mixed", 0.0, 255.0, 1.0),  # lights / dig
    MsgSpec(0x1A0, 0.100, 8, "phys", 0.0, 100.0, 0.4),   # fuel level
    MsgSpec(0x1B0, 1.000, 8, "const"),                    # slow heartbeat
    MsgSpec(0x1C0, 0.040, 8, "phys", 0.0, 200.0, 1.0),   # yaw rate proxy
)

ALL_IDS: tuple[int, ...] = tuple(s.can_id for s in MSG_SPECS)
ID_TO_SPEC: dict[int, MsgSpec] = {s.can_id: s for s in MSG_SPECS}

# Attacker-injected ID used by injection/flood (not in normal set)
INJECT_ID = 0x7FF

FRAME_COLUMNS = (
    "timestamp",
    "can_id",
    "dlc",
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7",
    "label",
    "attack_family",
)


def set_seed(seed: int) -> np.random.Generator:
    """Return a seeded Generator; also seed the global numpy RNG for callers."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _encode_phys(value: float, lo: float, hi: float) -> tuple[int, int]:
    """Encode a physical float into two bytes (uint16 scaled to [lo, hi])."""
    clipped = float(np.clip(value, lo, hi))
    span = hi - lo if hi > lo else 1.0
    scaled = int(round((clipped - lo) / span * 65535.0))
    scaled = int(np.clip(scaled, 0, 65535))
    return (scaled >> 8) & 0xFF, scaled & 0xFF


def _payload_for(
    spec: MsgSpec,
    phys: float,
    counter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build an 8-byte payload (unused bytes zeroed; DLC truncates semantically)."""
    data = np.zeros(8, dtype=np.uint8)
    if spec.kind == "phys":
        hi, lo = _encode_phys(phys, spec.phys_lo, spec.phys_hi)
        data[0], data[1] = hi, lo
        # mild noise in remaining used bytes
        if spec.dlc > 2:
            data[2:spec.dlc] = rng.integers(0, 16, size=spec.dlc - 2, dtype=np.uint8)
    elif spec.kind == "counter":
        data[0] = counter & 0xFF
        data[1] = (counter >> 8) & 0xFF
    elif spec.kind == "const":
        # deterministic constant per ID
        base = (spec.can_id * 17) & 0xFF
        for i in range(spec.dlc):
            data[i] = (base + i * 3) & 0xFF
    elif spec.kind == "mixed":
        hi, lo = _encode_phys(phys, spec.phys_lo, spec.phys_hi)
        data[0], data[1] = hi, lo
        data[2] = counter & 0xFF
        if spec.dlc > 3:
            data[3:spec.dlc] = rng.integers(0, 32, size=spec.dlc - 3, dtype=np.uint8)
    return data


def _frame_row(
    ts: float,
    can_id: int,
    dlc: int,
    data: np.ndarray,
    label: str,
    attack_family: Optional[str],
) -> dict:
    row = {
        "timestamp": float(ts),
        "can_id": int(can_id),
        "dlc": int(dlc),
        "label": label,
        "attack_family": attack_family if attack_family else "",
    }
    for i in range(8):
        row[f"b{i}"] = int(data[i]) if i < len(data) else 0
    return row


def generate_normal(
    duration_s: float = 30.0,
    seed: int = 0,
    jitter_frac: float = 0.02,
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Generate a normal (benign) CAN stream for ``duration_s`` seconds.

    Each ID emits at its nominal period with relative jitter ``jitter_frac``.
    Physical signals follow a clipped random walk. Labels are all ``normal``.
    """
    rng = set_seed(seed)
    rows: list[dict] = []

    # Per-ID state
    phys: dict[int, float] = {}
    counters: dict[int, int] = {}
    next_t: dict[int, float] = {}
    for spec in MSG_SPECS:
        mid = 0.5 * (spec.phys_lo + spec.phys_hi)
        phys[spec.can_id] = mid + float(rng.uniform(-0.1, 0.1) * (spec.phys_hi - spec.phys_lo))
        counters[spec.can_id] = int(rng.integers(0, 256))
        # stagger first emission
        next_t[spec.can_id] = start_time + float(rng.uniform(0.0, spec.period_s))

    t_end = start_time + duration_s
    # Event-driven: always emit the soonest next message
    while True:
        cid = min(next_t, key=next_t.get)
        t = next_t[cid]
        if t > t_end:
            break
        spec = ID_TO_SPEC[cid]
        # update phys
        if spec.kind in ("phys", "mixed"):
            step = float(rng.uniform(-spec.phys_step, spec.phys_step))
            phys[cid] = float(np.clip(phys[cid] + step, spec.phys_lo, spec.phys_hi))
        counters[cid] = (counters[cid] + 1) & 0xFFFF
        data = _payload_for(spec, phys[cid], counters[cid], rng)
        rows.append(_frame_row(t, cid, spec.dlc, data, "normal", None))
        # schedule next with jitter
        jitter = float(rng.uniform(-jitter_frac, jitter_frac)) * spec.period_s
        next_t[cid] = t + spec.period_s + jitter
        # keep IAT feasible (> 0)
        if next_t[cid] <= t:
            next_t[cid] = t + 1e-6

    df = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def _copy_normal_segment(
    duration_s: float,
    seed: int,
    start_time: float = 0.0,
) -> pd.DataFrame:
    return generate_normal(duration_s=duration_s, seed=seed, start_time=start_time)


def generate_injection(
    duration_s: float = 10.0,
    seed: int = 1,
    inject_rate_hz: float = 50.0,
    attack_start_s: float = 2.0,
    attack_end_s: Optional[float] = None,
) -> pd.DataFrame:
    """Inject frames with a foreign ID at ``inject_rate_hz`` during the attack window."""
    rng = set_seed(seed)
    base = _copy_normal_segment(duration_s, seed=seed + 1000)
    if attack_end_s is None:
        attack_end_s = duration_s - 1.0
    rows = base.to_dict("records")
    t = attack_start_s
    period = 1.0 / inject_rate_hz
    while t < attack_end_s:
        data = rng.integers(0, 256, size=8, dtype=np.uint8)
        # plant a distinctive pattern in first two bytes for objective checks
        data[0], data[1] = 0xDE, 0xAD
        rows.append(_frame_row(t, INJECT_ID, 8, data, "attack", "injection"))
        jitter = float(rng.uniform(-0.05, 0.05)) * period
        t = t + period + jitter
    df = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def generate_spoof(
    duration_s: float = 10.0,
    seed: int = 2,
    target_id: int = 0x100,
    target_phys: float = 200.0,
    attack_start_s: float = 2.0,
    attack_end_s: Optional[float] = None,
    gradual: bool = False,
    gradual_steps: int = 20,
) -> pd.DataFrame:
    """Spoof ``target_id`` payloads toward ``target_phys`` (optionally gradual)."""
    rng = set_seed(seed)
    base = _copy_normal_segment(duration_s, seed=seed + 2000)
    if attack_end_s is None:
        attack_end_s = duration_s - 1.0
    spec = ID_TO_SPEC[target_id]
    # baseline phys from first matching frame (approx mid)
    baseline = 0.5 * (spec.phys_lo + spec.phys_hi)

    records = base.to_dict("records")
    attack_mask_times = []
    for i, row in enumerate(records):
        if row["can_id"] != target_id:
            continue
        ts = row["timestamp"]
        if ts < attack_start_s or ts >= attack_end_s:
            continue
        if gradual:
            frac = (ts - attack_start_s) / max(attack_end_s - attack_start_s, 1e-9)
            # discrete staircase over gradual_steps
            step_i = min(int(frac * gradual_steps), gradual_steps - 1)
            frac_q = (step_i + 1) / gradual_steps
            value = baseline + frac_q * (target_phys - baseline)
        else:
            value = target_phys
        value = float(np.clip(value, spec.phys_lo, spec.phys_hi))
        # if target outside range, clamp but still mark attack (objective may fail —
        # caller can request in-range targets)
        hi, lo = _encode_phys(value, spec.phys_lo, spec.phys_hi)
        records[i]["b0"] = hi
        records[i]["b1"] = lo
        records[i]["label"] = "attack"
        records[i]["attack_family"] = "spoof"
        attack_mask_times.append(ts)

    # If target_phys is outside phys range, also inject override frames with
    # unconstrained encoding so the attack objective is measurable.
    # Prefer in-range targets; for out-of-range, stretch encoding.
    if target_phys > spec.phys_hi or target_phys < spec.phys_lo:
        # re-encode using extended range for objective demonstration
        ext_lo, ext_hi = min(spec.phys_lo, target_phys), max(spec.phys_hi, target_phys)
        for i, row in enumerate(records):
            if row["attack_family"] != "spoof":
                continue
            ts = row["timestamp"]
            if gradual:
                frac = (ts - attack_start_s) / max(attack_end_s - attack_start_s, 1e-9)
                step_i = min(int(frac * gradual_steps), gradual_steps - 1)
                frac_q = (step_i + 1) / gradual_steps
                value = baseline + frac_q * (target_phys - baseline)
            else:
                value = target_phys
            hi, lo = _encode_phys(value, ext_lo, ext_hi)
            records[i]["b0"] = hi
            records[i]["b1"] = lo

    df = pd.DataFrame(records, columns=list(FRAME_COLUMNS))
    return df


def generate_flood(
    duration_s: float = 10.0,
    seed: int = 3,
    flood_id: int = 0x100,
    flood_rate_hz: float = 500.0,
    attack_start_s: float = 2.0,
    attack_end_s: Optional[float] = None,
) -> pd.DataFrame:
    """Flood the bus with high-rate frames of ``flood_id``."""
    rng = set_seed(seed)
    base = _copy_normal_segment(duration_s, seed=seed + 3000)
    if attack_end_s is None:
        attack_end_s = duration_s - 1.0
    rows = base.to_dict("records")
    spec = ID_TO_SPEC.get(flood_id)
    dlc = spec.dlc if spec else 8
    t = attack_start_s
    period = 1.0 / flood_rate_hz
    while t < attack_end_s:
        data = rng.integers(0, 256, size=8, dtype=np.uint8)
        rows.append(_frame_row(t, flood_id, dlc, data, "attack", "flood"))
        t = t + period
    df = pd.DataFrame(rows, columns=list(FRAME_COLUMNS))
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def generate_drop(
    duration_s: float = 10.0,
    seed: int = 4,
    drop_id: int = 0x100,
    attack_start_s: float = 2.0,
    attack_end_s: Optional[float] = None,
) -> pd.DataFrame:
    """Suppress (drop) all frames of ``drop_id`` during the attack window."""
    set_seed(seed)
    base = _copy_normal_segment(duration_s, seed=seed + 4000)
    if attack_end_s is None:
        attack_end_s = duration_s - 1.0
    records = base.to_dict("records")
    kept: list[dict] = []
    for row in records:
        if (
            row["can_id"] == drop_id
            and attack_start_s <= row["timestamp"] < attack_end_s
        ):
            # drop it; leave a marker row? No — true suppress. Mark neighbors?
            # For labeling windows we need attack labels on remaining frames in
            # the window. Tag nearby frames of other IDs in the drop interval.
            continue
        if attack_start_s <= row["timestamp"] < attack_end_s:
            # mark ambient frames in the drop interval so windows are labeled
            row = dict(row)
            row["label"] = "attack"
            row["attack_family"] = "drop"
        kept.append(row)
    df = pd.DataFrame(kept, columns=list(FRAME_COLUMNS))
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def generate_replay(
    duration_s: float = 10.0,
    seed: int = 5,
    replay_id: int = 0x100,
    attack_start_s: float = 2.0,
    attack_end_s: Optional[float] = None,
    capture_start_s: float = 0.5,
    capture_end_s: float = 1.5,
) -> pd.DataFrame:
    """Replay previously captured payloads of ``replay_id`` during the attack window."""
    set_seed(seed)
    base = _copy_normal_segment(duration_s, seed=seed + 5000)
    if attack_end_s is None:
        attack_end_s = duration_s - 1.0

    captured = base[
        (base["can_id"] == replay_id)
        & (base["timestamp"] >= capture_start_s)
        & (base["timestamp"] < capture_end_s)
    ]
    if len(captured) == 0:
        # fallback: capture first few of that ID
        captured = base[base["can_id"] == replay_id].head(10)
    payloads = captured[[f"b{i}" for i in range(8)]].to_numpy()
    if len(payloads) == 0:
        payloads = np.zeros((1, 8), dtype=np.uint8)

    records = base.to_dict("records")
    idx = 0
    for i, row in enumerate(records):
        if row["can_id"] != replay_id:
            continue
        if not (attack_start_s <= row["timestamp"] < attack_end_s):
            continue
        p = payloads[idx % len(payloads)]
        for b in range(8):
            records[i][f"b{b}"] = int(p[b])
        records[i]["label"] = "attack"
        records[i]["attack_family"] = "replay"
        idx += 1

    df = pd.DataFrame(records, columns=list(FRAME_COLUMNS))
    return df


def generate_attack(
    family: str,
    duration_s: float = 10.0,
    seed: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Dispatch to a family-specific generator."""
    family = family.lower()
    dispatch = {
        "injection": generate_injection,
        "spoof": generate_spoof,
        "flood": generate_flood,
        "drop": generate_drop,
        "replay": generate_replay,
    }
    if family not in dispatch:
        raise ValueError(f"Unknown attack family: {family}")
    return dispatch[family](duration_s=duration_s, seed=seed, **kwargs)


def frames_to_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convenience: export core columns as numpy arrays."""
    data = df[[f"b{i}" for i in range(8)]].to_numpy(dtype=np.uint8)
    return {
        "timestamp": df["timestamp"].to_numpy(dtype=np.float64),
        "can_id": df["can_id"].to_numpy(dtype=np.int64),
        "dlc": df["dlc"].to_numpy(dtype=np.int64),
        "data": data,
        "label": df["label"].to_numpy(),
        "attack_family": df["attack_family"].to_numpy(),
    }


ATTACK_FAMILIES: tuple[str, ...] = (
    "injection",
    "spoof",
    "flood",
    "drop",
    "replay",
)
