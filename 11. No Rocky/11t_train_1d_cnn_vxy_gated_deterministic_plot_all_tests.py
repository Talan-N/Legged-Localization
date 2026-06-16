#!/usr/bin/env python3
"""
Version 11T: V11T plus avg+max temporal pooling for Go2.

This script is designed to answer the next question in your project:

    Can the CNN learn slip corrections across multiple slippery/low-traction terrains
    without needing a manually tuned alpha for each run?

Compared with 10_train_1d_cnn_vxy_terrain_gated.py:

  1. Still predicts only vx/vy residuals:
       residual_vx = vicon_vx - odom_vx
       residual_vy = vicon_vy - odom_vy

  2. Still leaves wz unchanged:
       corrected_wz = odom_wz

     This avoids the large wz RMSE problem you observed.

  3. Adds terrain context inferred from run_name:
       terrain_watery, terrain_slippery, terrain_rocky, terrain_whiteboard, terrain_other

  4. Adds maneuver context inferred from run_name:
       maneuver_m_path, maneuver_circle, maneuver_s_turn, maneuver_rectangle,
       maneuver_square, maneuver_sideways_square, maneuver_sideways, maneuver_rough, maneuver_other

  5. Adds a learned correction gate:
       gate_vx, gate_vy in [0, 1]

     Final correction:
       corrected_vx = odom_vx + alpha * gate_vx * clipped_pred_residual_vx
       corrected_vy = odom_vy + alpha * gate_vy * clipped_pred_residual_vy

     Here alpha is now a safety ceiling, not the real hand-tuned gain.

  6. Trains using corrected-velocity loss plus a residual-shape auxiliary loss:
       loss = velocity_loss + residual_loss_weight * residual_loss + gate_penalty * mean_gate

     This encourages the model to learn both a useful residual direction and when to apply it.

  7. Supports group-balanced loss so one terrain/maneuver group does not dominate.

  8. Saves per-run position metrics in addition to velocity metrics:
       run_position_metrics_train.csv
       run_position_metrics_test.csv

Suggested first run, matching your good 08 setup but with learned gate:

python 11_train_1d_cnn_vxy_terrain_maneuver_gated.py \
  --csv processed_csv_all/all_runs_aligned_50hz_body_vicon.csv \
  --out-dir models/v11_terrain_maneuver_gated_slip \
  --include-runs slippery,water,wet \
  --alpha 0.6 \
  --clip-residual 0.5 \
  --window 128 \
  --residual-loss-weight 0.20 \
  --gate-penalty 0.002

If you want to include every terrain:

python 11_train_1d_cnn_vxy_terrain_maneuver_gated.py \
  --csv processed_csv_all/all_runs_aligned_50hz_body_vicon.csv \
  --out-dir models/v11_terrain_maneuver_gated_all_runs \
  --alpha 0.6 \
  --clip-residual 0.5 \
  --window 128
"""

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_RAW_FEATURE_COLS = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "roll", "pitch", "yaw",

    *[f"q_{i}" for i in range(12)],
    *[f"dq_{i}" for i in range(12)],
    *[f"tau_{i}" for i in range(12)],

    *[f"foot_force_{i}" for i in range(4)],
    *[f"foot_force_est_{i}" for i in range(4)],

    "odom_vx", "odom_vy", "odom_wz",
]

TERRAIN_NAMES = ["watery", "slippery", "rocky", "whiteboard", "other"]
TERRAIN_CONTEXT_COLS = [f"terrain_{name}" for name in TERRAIN_NAMES]

MANEUVER_NAMES = [
    "m_path", "circle", "s_turn", "rectangle", "square",
    "sideways_square", "sideways", "rough", "other",
]
MANEUVER_CONTEXT_COLS = [f"maneuver_{name}" for name in MANEUVER_NAMES]

DERIVED_MOTION_CONTEXT_COLS = [
    "odom_speed_xy",
    "abs_odom_vx",
    "abs_odom_vy",
    "abs_odom_wz",
]

BASELINE_COLS = ["odom_vx", "odom_vy", "odom_wz"]
TRUTH_COLS = ["vicon_vx", "vicon_vy", "vicon_wz"]
TARGET_RESIDUAL_COLS = ["residual_vx", "residual_vy"]
ALL_RESIDUAL_COLS = ["residual_vx", "residual_vy", "residual_wz"]


# -----------------------------
# Data containers / dataset
# -----------------------------

@dataclass
class WindowedData:
    X: np.ndarray
    baseline: np.ndarray
    residual_vxy: np.ndarray
    truth: np.ndarray
    sample_weight: np.ndarray
    meta: pd.DataFrame


class Go2WindowDataset(Dataset):
    def __init__(self, data: WindowedData):
        self.X = torch.tensor(data.X, dtype=torch.float32)
        self.baseline = torch.tensor(data.baseline, dtype=torch.float32)
        self.residual_vxy = torch.tensor(data.residual_vxy, dtype=torch.float32)
        self.truth = torch.tensor(data.truth, dtype=torch.float32)
        self.sample_weight = torch.tensor(data.sample_weight, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.baseline[idx],
            self.residual_vxy[idx],
            self.truth[idx],
            self.sample_weight[idx],
        )


# -----------------------------
# Model
# -----------------------------

class TerrainManeuverGatedCNN1D(nn.Module):
    """Shared CNN encoder with two heads:

    residual_head: predicts scaled vx/vy residuals.
    gate_head: predicts per-axis gate values in [gate_min, 1].

    The gate is the critical difference from 08/09. It lets the network learn
    when to correct strongly and when to trust Go2 odometry.
    """

    def __init__(self, input_features: int, dropout: float = 0.10, gate_min: float = 0.0, gate_bias_init: float = 0.0):
        super().__init__()
        if not (0.0 <= gate_min < 1.0):
            raise ValueError("gate_min must be in [0, 1).")
        self.gate_min = float(gate_min)

        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # V11T change: preserve both average activation and peak activation over
        # the 128-sample window. V11M used only AdaptiveAvgPool1d(1), which can
        # wash out short slip/foot-contact events. Avg+max pooling keeps the
        # architecture simple while giving the heads access to temporal peaks.
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        self.gate_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        # Optional initialization: negative means conservative initial gates;
        # 0.0 means initial sigmoid gate roughly 0.5.
        final_gate_layer = self.gate_head[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.constant_(final_gate_layer.bias, float(gate_bias_init))

    def forward(self, x):
        h = self.encoder(x)          # (B, 128, T)
        z_avg = torch.mean(h, dim=-1)
        z_max = torch.amax(h, dim=-1)
        z = torch.cat([z_avg, z_max], dim=1)  # (B, 256)
        z = self.shared_fc(z)
        residual_scaled = self.residual_head(z)
        gate_logits = self.gate_head(z)
        gate = self.gate_min + (1.0 - self.gate_min) * torch.sigmoid(gate_logits)
        return residual_scaled, gate


# -----------------------------
# Utility functions
# -----------------------------

def set_seed(seed: int, deterministic: bool = True):
    """Set RNG seeds and optionally request deterministic PyTorch behavior.

    Fixed train/test splits only keep the same runs in each split. This function
    also reduces run-to-run training variation from random initialization, batch
    shuffling, dropout, and CUDA/cuDNN algorithm selection.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions may not support warn_only.
            torch.use_deterministic_algorithms(True)




def seed_worker(worker_id: int):
    """Make DataLoader workers deterministic when num_workers > 0."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def wrap_pi(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    if not np.isfinite(angle):
        return np.nan
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def require_columns(df: pd.DataFrame, cols: List[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def infer_terrain_from_run_name(run_name: str) -> str:
    r = str(run_name).lower()
    if "water" in r or "watery" in r or "wet" in r:
        return "watery"
    if "slip" in r or "slippery" in r:
        return "slippery"
    if "rock" in r or "pebble" in r or "stone" in r:
        return "rocky"
    if "whiteboard" in r or "white_board" in r or "white" in r or "board" in r:
        return "whiteboard"
    return "other"


def infer_maneuver_from_run_name(run_name: str) -> str:
    """Infer a coarse maneuver class from run_name.

    This is intentionally simple and transparent. It is an oracle/diagnostic
    label just like terrain_name: useful for testing whether maneuver context
    helps before you try to infer it automatically from sensors.
    """
    r = str(run_name).lower()
    tokens = re.split(r"[^a-z0-9]+", r)

    if "s" in tokens and "turn" in tokens:
        return "s_turn"
    if "s_turn" in r or "sturn" in r or "s-turn" in r:
        return "s_turn"
    if "circle" in tokens or "circle" in r:
        return "circle"
    if "rectangle" in tokens or "rect" in tokens or "rectangle" in r:
        return "rectangle"
    if "sideways" in tokens and "square" in tokens:
        return "sideways_square"
    if "sideways" in tokens or "sideways" in r:
        return "sideways"
    if "square" in tokens or "square" in r:
        return "square"
    if "pebble" in r or "pebbles" in r or "stone" in r or "stones" in r:
        return "rough"
    if "m" in tokens:
        return "m_path"
    return "other"


def add_terrain_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["terrain_name"] = df["run_name"].map(infer_terrain_from_run_name)
    for name in TERRAIN_NAMES:
        df[f"terrain_{name}"] = (df["terrain_name"] == name).astype(float)
    return df


def add_maneuver_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["maneuver_name"] = df["run_name"].map(infer_maneuver_from_run_name)
    for name in MANEUVER_NAMES:
        df[f"maneuver_{name}"] = (df["maneuver_name"] == name).astype(float)
    return df


def add_derived_motion_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["odom_speed_xy"] = np.sqrt(df["odom_vx"].astype(float) ** 2 + df["odom_vy"].astype(float) ** 2)
    df["abs_odom_vx"] = np.abs(df["odom_vx"].astype(float))
    df["abs_odom_vy"] = np.abs(df["odom_vy"].astype(float))
    df["abs_odom_wz"] = np.abs(df["odom_wz"].astype(float))
    return df


def filtered_run_names(all_runs: List[str], include_arg: str, exclude_arg: str) -> List[str]:
    runs = list(all_runs)

    include_terms = [s.strip().lower() for s in include_arg.split(",") if s.strip()]
    exclude_terms = [s.strip().lower() for s in exclude_arg.split(",") if s.strip()]

    if include_terms:
        runs = [r for r in runs if any(term in r.lower() for term in include_terms)]

    if exclude_terms:
        runs = [r for r in runs if not any(term in r.lower() for term in exclude_terms)]

    if not runs:
        raise ValueError(
            "No runs left after include/exclude filtering. "
            f"include={include_terms}, exclude={exclude_terms}, available={all_runs}"
        )

    return runs


def split_single_group(group_runs: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    runs = list(group_runs)
    rng.shuffle(runs)
    n = len(runs)

    if n <= 1:
        return runs, [], []
    if n == 2:
        return runs[1:], [], runs[:1]
    if n == 3:
        return runs[2:], runs[1:2], runs[:1]

    n_test = max(1, int(round(0.20 * n)))
    remaining_n = n - n_test
    n_val = max(1, int(round(val_fraction * remaining_n))) if remaining_n > 2 else 0

    test_runs = runs[:n_test]
    val_runs = runs[n_test:n_test + n_val]
    train_runs = runs[n_test + n_val:]

    if len(train_runs) == 0:
        # Keep the split valid even for tiny groups.
        train_runs = val_runs[-1:]
        val_runs = val_runs[:-1]

    return train_runs, val_runs, test_runs


def split_runs(
    run_names: List[str],
    test_runs_arg: str,
    val_fraction: float,
    seed: int,
    stratified_by_terrain: bool,
) -> Tuple[List[str], List[str], List[str]]:
    all_runs = list(run_names)

    if test_runs_arg:
        test_runs = [r.strip() for r in test_runs_arg.split(",") if r.strip()]
        unknown = [r for r in test_runs if r not in all_runs]
        if unknown:
            raise ValueError(f"Requested --test-runs not found after filtering: {unknown}\nAvailable runs: {all_runs}")
        remaining = [r for r in all_runs if r not in test_runs]
        train_runs, val_runs, _ = split_runs(remaining, "", val_fraction, seed, stratified_by_terrain)
        return train_runs, val_runs, test_runs

    if not stratified_by_terrain:
        rng = random.Random(seed)
        shuffled = all_runs[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(0.20 * len(shuffled)))) if len(shuffled) > 1 else 0
        test_runs = shuffled[:n_test]
        remaining = shuffled[n_test:]

        rng = random.Random(seed + 1)
        rng.shuffle(remaining)
        n_val = max(1, int(round(val_fraction * len(remaining)))) if len(remaining) > 2 else 0
        val_runs = remaining[:n_val]
        train_runs = remaining[n_val:]
        if len(train_runs) == 0:
            raise ValueError("No training runs left after split. Use fewer --test-runs or loosen include/exclude filters.")
        return train_runs, val_runs, test_runs

    by_terrain: Dict[str, List[str]] = {name: [] for name in TERRAIN_NAMES}
    for r in all_runs:
        by_terrain[infer_terrain_from_run_name(r)].append(r)

    train_runs: List[str] = []
    val_runs: List[str] = []
    test_runs: List[str] = []
    for i, name in enumerate(TERRAIN_NAMES):
        group = sorted(by_terrain[name])
        if not group:
            continue
        tr, va, te = split_single_group(group, val_fraction, seed + 100 * i)
        train_runs.extend(tr)
        val_runs.extend(va)
        test_runs.extend(te)

    train_runs = sorted(train_runs)
    val_runs = sorted(val_runs)
    test_runs = sorted(test_runs)

    if len(train_runs) == 0:
        raise ValueError("No training runs left after stratified split. Use --random-split or fewer --test-runs.")

    return train_runs, val_runs, test_runs


def select_feature_cols(
    df: pd.DataFrame,
    use_vicon_pose_context: bool,
    terrain_context: bool,
    maneuver_context: bool,
    derived_motion_context: bool,
) -> List[str]:
    feature_cols = BASE_RAW_FEATURE_COLS[:]

    for c in ["odom_x", "odom_y", "odom_yaw"]:
        if c in df.columns:
            feature_cols.append(c)

    command_candidates = [
        "cmd_vx", "cmd_vy", "cmd_wz",
        "command_vx", "command_vy", "command_wz",
        "target_vx", "target_vy", "target_wz",
    ]
    for c in command_candidates:
        if c in df.columns:
            feature_cols.append(c)

    if terrain_context:
        feature_cols.extend([c for c in TERRAIN_CONTEXT_COLS if c in df.columns])

    if maneuver_context:
        feature_cols.extend([c for c in MANEUVER_CONTEXT_COLS if c in df.columns])

    if derived_motion_context:
        feature_cols.extend([c for c in DERIVED_MOTION_CONTEXT_COLS if c in df.columns])

    # Usually do NOT use vicon pose as an input, because it will not exist in deployment.
    if use_vicon_pose_context:
        for c in ["vicon_x", "vicon_y", "vicon_yaw"]:
            if c in df.columns:
                feature_cols.append(c)

    return list(dict.fromkeys(feature_cols))


def fit_scalers(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(train_df[feature_cols].values)
    y_scaler.fit(train_df[TARGET_RESIDUAL_COLS].values)
    return x_scaler, y_scaler


def make_windows_by_run(
    df: pd.DataFrame,
    run_names: List[str],
    feature_cols: List[str],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    window: int,
) -> WindowedData:
    X_windows = []
    baseline_targets = []
    residual_targets = []
    truth_targets = []
    sample_weights = []
    meta_rows = []

    for run_name in run_names:
        run_df = df[df["run_name"] == run_name].copy().sort_values("time_s")
        if len(run_df) <= window:
            print(f"WARNING: skipping run {run_name!r}; length {len(run_df)} <= window {window}")
            continue

        X_scaled = x_scaler.transform(run_df[feature_cols].values)
        residual_scaled = y_scaler.transform(run_df[TARGET_RESIDUAL_COLS].values)
        baseline_raw = run_df[BASELINE_COLS].values
        truth_raw = run_df[TRUTH_COLS].values

        for i in range(window, len(run_df)):
            X_windows.append(X_scaled[i - window:i])
            baseline_targets.append(baseline_raw[i])
            residual_targets.append(residual_scaled[i])
            truth_targets.append(truth_raw[i])
            sample_weights.append(1.0)

            row = run_df.iloc[i]
            terrain_name = row.get("terrain_name", infer_terrain_from_run_name(run_name))
            maneuver_name = row.get("maneuver_name", infer_maneuver_from_run_name(run_name))
            meta_rows.append({
                "run_name": run_name,
                "terrain_name": terrain_name,
                "maneuver_name": maneuver_name,
                "time_s": row.get("time_s", np.nan),
                "odom_x": row.get("odom_x", np.nan),
                "odom_y": row.get("odom_y", np.nan),
                "odom_yaw": row.get("odom_yaw", np.nan),
                "vicon_x": row.get("vicon_x", np.nan),
                "vicon_y": row.get("vicon_y", np.nan),
                "vicon_yaw": row.get("vicon_yaw", np.nan),
            })

    if not X_windows:
        raise ValueError("No windows were created. Check run split and window length.")

    X_windows = np.asarray(X_windows, dtype=np.float32)
    baseline_targets = np.asarray(baseline_targets, dtype=np.float32)
    residual_targets = np.asarray(residual_targets, dtype=np.float32)
    truth_targets = np.asarray(truth_targets, dtype=np.float32)
    sample_weights = np.asarray(sample_weights, dtype=np.float32)

    # PyTorch Conv1d expects (N, C, T).
    X_windows = np.transpose(X_windows, (0, 2, 1))
    meta = pd.DataFrame(meta_rows)

    return WindowedData(X_windows, baseline_targets, residual_targets, truth_targets, sample_weights, meta)


def apply_group_balanced_weights(data: WindowedData, group_col: str = "terrain_name", cap: float = 5.0) -> WindowedData:
    """Inverse-frequency sample weighting by terrain, maneuver, or terrain+maneuver.

    group_col can be:
      - terrain_name
      - maneuver_name
      - terrain_maneuver

    This prevents one group with more windows from dominating training.
    """
    data = WindowedData(
        X=data.X,
        baseline=data.baseline,
        residual_vxy=data.residual_vxy,
        truth=data.truth,
        sample_weight=data.sample_weight.copy(),
        meta=data.meta.copy(),
    )

    if group_col == "terrain_maneuver":
        groups = data.meta["terrain_name"].astype(str) + "__" + data.meta["maneuver_name"].astype(str)
    elif group_col in data.meta.columns:
        groups = data.meta[group_col].astype(str)
    else:
        raise ValueError(f"Unknown group_col for balanced weights: {group_col}")

    counts = groups.value_counts().to_dict()
    if not counts:
        return data

    n_total = len(groups)
    n_groups = len(counts)
    weights_by_group = {}
    for group, count in counts.items():
        w = n_total / max(1, n_groups * count)
        weights_by_group[group] = min(float(w), float(cap))

    weights = groups.map(weights_by_group).astype(float).values
    weights = weights / max(1e-8, float(np.mean(weights)))
    data.sample_weight = weights.astype(np.float32)
    data.meta["sample_weight"] = data.sample_weight
    data.meta["balance_group"] = groups.values
    return data


# -----------------------------
# Loss / train / eval
# -----------------------------

@dataclass
class LossConfig:
    alpha: float
    clip_residual: float
    residual_loss_weight: float
    gate_penalty: float
    y_mean: torch.Tensor
    y_scale: torch.Tensor


def weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    # values shape can be (B,) or (B, D). We weight per sample.
    if values.ndim == 2:
        per_sample = values.mean(dim=1)
    else:
        per_sample = values
    w = sample_weight.view(-1)
    return (per_sample * w).sum() / torch.clamp(w.sum(), min=1e-8)


def compute_loss(model, batch, config: LossConfig, device: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    X_batch, baseline_batch, residual_batch, truth_batch, weight_batch = batch
    X_batch = X_batch.to(device)
    baseline_batch = baseline_batch.to(device)
    residual_batch = residual_batch.to(device)
    truth_batch = truth_batch.to(device)
    weight_batch = weight_batch.to(device)

    pred_scaled, gate = model(X_batch)

    y_mean = config.y_mean.to(device)
    y_scale = config.y_scale.to(device)

    pred_raw = pred_scaled * y_scale + y_mean
    if config.clip_residual > 0:
        pred_raw_for_correction = torch.clamp(pred_raw, -config.clip_residual, config.clip_residual)
    else:
        pred_raw_for_correction = pred_raw

    corrected_vxy = baseline_batch[:, :2] + config.alpha * gate * pred_raw_for_correction

    # Train the quantity we actually care about at velocity level, but normalize
    # by residual scale so vx and vy have comparable influence.
    velocity_error_scaled = (corrected_vxy - truth_batch[:, :2]) / y_scale
    velocity_loss = weighted_mean(velocity_error_scaled ** 2, weight_batch)

    # Auxiliary term: keep the residual head learning the residual shape even if
    # the gate chooses to suppress correction in noisy terrain.
    residual_loss = weighted_mean((pred_scaled - residual_batch) ** 2, weight_batch)

    # Small safety regularizer: prefer lower gates unless correction reduces loss.
    gate_loss = weighted_mean(gate, weight_batch)

    loss = velocity_loss + config.residual_loss_weight * residual_loss + config.gate_penalty * gate_loss

    parts = {
        "loss": float(loss.detach().cpu()),
        "velocity_loss": float(velocity_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "gate_mean": float(gate_loss.detach().cpu()),
    }
    return loss, parts


def train_one_epoch(model, loader, optimizer, config: LossConfig, device: str, grad_clip: float) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "velocity_loss": 0.0, "residual_loss": 0.0, "gate_mean": 0.0}
    n_batches = 0

    for batch in loader:
        loss, parts = compute_loss(model, batch, config, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in totals:
            totals[k] += parts[k]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def evaluate_loss(model, loader, config: LossConfig, device: str) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "velocity_loss": 0.0, "residual_loss": 0.0, "gate_mean": 0.0}
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            _, parts = compute_loss(model, batch, config, device)
            for k in totals:
                totals[k] += parts[k]
            n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# -----------------------------
# Prediction / metrics
# -----------------------------

def predict_dataset(
    model,
    data: WindowedData,
    y_scaler: StandardScaler,
    batch_size: int,
    device: str,
    alpha: float,
    clip_residual: float,
    turn_gate_wz: float = 0.0,
    turn_gate_power: float = 1.0,
) -> pd.DataFrame:
    ds = Go2WindowDataset(data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    pred_scaled_all = []
    pred_raw_all = []
    gate_all = []
    model.eval()

    y_mean = torch.tensor(y_scaler.mean_, dtype=torch.float32, device=device).view(1, -1)
    y_scale = torch.tensor(y_scaler.scale_, dtype=torch.float32, device=device).view(1, -1)

    with torch.no_grad():
        for X_batch, _, _, _, _ in loader:
            X_batch = X_batch.to(device)
            pred_scaled, gate = model(X_batch)
            pred_raw = pred_scaled * y_scale + y_mean
            pred_scaled_all.append(pred_scaled.cpu().numpy())
            pred_raw_all.append(pred_raw.cpu().numpy())
            gate_all.append(gate.cpu().numpy())

    pred_scaled_arr = np.vstack(pred_scaled_all)
    pred_residuals = np.vstack(pred_raw_all)
    learned_gate = np.vstack(gate_all)

    if clip_residual > 0:
        pred_residuals_clipped = np.clip(pred_residuals, -clip_residual, clip_residual)
    else:
        pred_residuals_clipped = pred_residuals.copy()

    baseline = data.baseline
    truth = data.truth

    if turn_gate_wz and turn_gate_wz > 0:
        turn_scale = 1.0 - (np.abs(baseline[:, 2]) / float(turn_gate_wz))
        turn_scale = np.clip(turn_scale, 0.0, 1.0)
        if turn_gate_power and turn_gate_power != 1.0:
            turn_scale = turn_scale ** float(turn_gate_power)
    else:
        turn_scale = np.ones(len(baseline), dtype=np.float32)

    applied_alpha_vx = alpha * learned_gate[:, 0] * turn_scale
    applied_alpha_vy = alpha * learned_gate[:, 1] * turn_scale

    corrected = baseline.copy()
    corrected[:, 0] = baseline[:, 0] + applied_alpha_vx * pred_residuals_clipped[:, 0]
    corrected[:, 1] = baseline[:, 1] + applied_alpha_vy * pred_residuals_clipped[:, 1]
    corrected[:, 2] = baseline[:, 2]

    out = data.meta.copy().reset_index(drop=True)

    out["true_vx"] = truth[:, 0]
    out["true_vy"] = truth[:, 1]
    out["true_wz"] = truth[:, 2]

    out["odom_vx"] = baseline[:, 0]
    out["odom_vy"] = baseline[:, 1]
    out["odom_wz"] = baseline[:, 2]

    out["true_residual_vx"] = truth[:, 0] - baseline[:, 0]
    out["true_residual_vy"] = truth[:, 1] - baseline[:, 1]

    out["pred_residual_vx_raw"] = pred_residuals[:, 0]
    out["pred_residual_vy_raw"] = pred_residuals[:, 1]
    out["pred_residual_vx_clipped"] = pred_residuals_clipped[:, 0]
    out["pred_residual_vy_clipped"] = pred_residuals_clipped[:, 1]
    out["pred_residual_vx_scaled"] = pred_scaled_arr[:, 0]
    out["pred_residual_vy_scaled"] = pred_scaled_arr[:, 1]

    out["learned_gate_vx"] = learned_gate[:, 0]
    out["learned_gate_vy"] = learned_gate[:, 1]
    out["turn_scale"] = turn_scale
    out["applied_alpha_vx"] = applied_alpha_vx
    out["applied_alpha_vy"] = applied_alpha_vy

    out["applied_residual_vx"] = applied_alpha_vx * pred_residuals_clipped[:, 0]
    out["applied_residual_vy"] = applied_alpha_vy * pred_residuals_clipped[:, 1]
    out["applied_residual_wz"] = 0.0

    out["corrected_vx"] = corrected[:, 0]
    out["corrected_vy"] = corrected[:, 1]
    out["corrected_wz"] = corrected[:, 2]

    return out


def velocity_metrics(pred_df: pd.DataFrame) -> Dict:
    axes = ["vx", "vy", "wz"]
    metrics = {"per_axis": {}, "overall": {}, "overall_vxy_only": {}}

    baseline_errs = []
    corrected_errs = []
    baseline_errs_vxy = []
    corrected_errs_vxy = []

    for axis in axes:
        b_err = pred_df[f"odom_{axis}"].values - pred_df[f"true_{axis}"].values
        c_err = pred_df[f"corrected_{axis}"].values - pred_df[f"true_{axis}"].values

        b_mse = float(np.mean(b_err ** 2))
        c_mse = float(np.mean(c_err ** 2))

        metrics["per_axis"][axis] = {
            "baseline_mse": b_mse,
            "corrected_mse": c_mse,
            "baseline_rmse": float(math.sqrt(b_mse)),
            "corrected_rmse": float(math.sqrt(c_mse)),
            "mse_improvement_percent": float(100.0 * (b_mse - c_mse) / b_mse) if b_mse > 0 else None,
        }

        baseline_errs.append(b_err)
        corrected_errs.append(c_err)

        if axis in ["vx", "vy"]:
            baseline_errs_vxy.append(b_err)
            corrected_errs_vxy.append(c_err)

    baseline_errs = np.vstack(baseline_errs).T
    corrected_errs = np.vstack(corrected_errs).T
    b_all = float(np.mean(baseline_errs ** 2))
    c_all = float(np.mean(corrected_errs ** 2))

    metrics["overall"] = {
        "baseline_mse": b_all,
        "corrected_mse": c_all,
        "baseline_rmse": float(math.sqrt(b_all)),
        "corrected_rmse": float(math.sqrt(c_all)),
        "mse_improvement_percent": float(100.0 * (b_all - c_all) / b_all) if b_all > 0 else None,
    }

    baseline_errs_vxy = np.vstack(baseline_errs_vxy).T
    corrected_errs_vxy = np.vstack(corrected_errs_vxy).T
    b_vxy = float(np.mean(baseline_errs_vxy ** 2))
    c_vxy = float(np.mean(corrected_errs_vxy ** 2))

    metrics["overall_vxy_only"] = {
        "baseline_mse": b_vxy,
        "corrected_mse": c_vxy,
        "baseline_rmse": float(math.sqrt(b_vxy)),
        "corrected_rmse": float(math.sqrt(c_vxy)),
        "mse_improvement_percent": float(100.0 * (b_vxy - c_vxy) / b_vxy) if b_vxy > 0 else None,
    }

    return metrics


def velocity_metrics_by_terrain(pred_df: pd.DataFrame) -> Dict[str, Dict]:
    out = {}
    if "terrain_name" not in pred_df.columns:
        return out
    for terrain, g in pred_df.groupby("terrain_name"):
        out[str(terrain)] = velocity_metrics(g)
    return out


def reconstruct_paths_for_predictions(pred_df: pd.DataFrame, velocity_frame: str = "body") -> pd.DataFrame:
    pieces = []

    for run_name, run_df in pred_df.groupby("run_name", sort=False):
        g = run_df.sort_values("time_s").copy().reset_index(drop=True)

        corrected_x = np.zeros(len(g), dtype=float)
        corrected_y = np.zeros(len(g), dtype=float)
        corrected_yaw = np.zeros(len(g), dtype=float)

        corrected_x[0] = g.loc[0, "odom_x"] if pd.notna(g.loc[0, "odom_x"]) else 0.0
        corrected_y[0] = g.loc[0, "odom_y"] if pd.notna(g.loc[0, "odom_y"]) else 0.0
        corrected_yaw[0] = g.loc[0, "odom_yaw"] if pd.notna(g.loc[0, "odom_yaw"]) else 0.0

        t = g["time_s"].values.astype(float)

        for i in range(1, len(g)):
            dt = t[i] - t[i - 1]
            if not np.isfinite(dt) or dt <= 0 or dt > 1.0:
                dt = 0.02

            vx = float(g.loc[i - 1, "corrected_vx"])
            vy = float(g.loc[i - 1, "corrected_vy"])
            wz = float(g.loc[i - 1, "corrected_wz"])

            yaw_prev = corrected_yaw[i - 1]
            corrected_yaw[i] = yaw_prev + wz * dt

            if velocity_frame == "body":
                dx = (vx * math.cos(yaw_prev) - vy * math.sin(yaw_prev)) * dt
                dy = (vx * math.sin(yaw_prev) + vy * math.cos(yaw_prev)) * dt
            elif velocity_frame == "world":
                dx = vx * dt
                dy = vy * dt
            else:
                raise ValueError("velocity_frame must be 'body' or 'world'")

            corrected_x[i] = corrected_x[i - 1] + dx
            corrected_y[i] = corrected_y[i - 1] + dy

        g["corrected_x"] = corrected_x
        g["corrected_y"] = corrected_y
        g["corrected_yaw"] = corrected_yaw

        pieces.append(g)

    return pd.concat(pieces, ignore_index=True) if pieces else pred_df.copy()


def run_position_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run pose metrics.

    V11 already reported trajectory RMSE and final position error. This edited
    version expands the report so each run shows:
      - trajectory RMSE improvement
      - final position error improvement
      - final position delta, where negative means CNN ended farther from truth
      - max position error improvement
      - final yaw error, even though wz is intentionally unchanged
      - simple boolean flags for whether trajectory/final/max errors improved
    """
    rows = []
    required_xy = {"vicon_x", "vicon_y", "odom_x", "odom_y", "corrected_x", "corrected_y"}
    if not required_xy.issubset(pred_df.columns):
        return pd.DataFrame()

    for run_name, g0 in pred_df.groupby("run_name", sort=True):
        g = g0.sort_values("time_s").copy()
        terrain = g["terrain_name"].iloc[0] if "terrain_name" in g.columns and len(g) else infer_terrain_from_run_name(run_name)
        maneuver = g["maneuver_name"].iloc[0] if "maneuver_name" in g.columns and len(g) else infer_maneuver_from_run_name(run_name)

        valid = g[list(required_xy)].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) == 0:
            continue

        odom_err = np.sqrt((valid["odom_x"] - valid["vicon_x"]) ** 2 + (valid["odom_y"] - valid["vicon_y"]) ** 2)
        corr_err = np.sqrt((valid["corrected_x"] - valid["vicon_x"]) ** 2 + (valid["corrected_y"] - valid["vicon_y"]) ** 2)

        final = valid.iloc[-1]
        odom_final = float(math.hypot(final["odom_x"] - final["vicon_x"], final["odom_y"] - final["vicon_y"]))
        corr_final = float(math.hypot(final["corrected_x"] - final["vicon_x"], final["corrected_y"] - final["vicon_y"]))

        odom_rmse = float(np.sqrt(np.mean(odom_err ** 2)))
        corr_rmse = float(np.sqrt(np.mean(corr_err ** 2)))
        odom_max = float(np.max(odom_err))
        corr_max = float(np.max(corr_err))

        traj_improvement = float(100.0 * (odom_rmse - corr_rmse) / odom_rmse) if odom_rmse > 0 else np.nan
        final_improvement = float(100.0 * (odom_final - corr_final) / odom_final) if odom_final > 0 else np.nan
        max_improvement = float(100.0 * (odom_max - corr_max) / odom_max) if odom_max > 0 else np.nan

        # Final yaw metrics. corrected_yaw exists after reconstruct_paths_for_predictions.
        odom_final_yaw_err_deg = np.nan
        corrected_final_yaw_err_deg = np.nan
        yaw_improvement_deg = np.nan
        yaw_improvement_percent = np.nan
        yaw_cols = {"odom_yaw", "corrected_yaw", "vicon_yaw"}
        if yaw_cols.issubset(g.columns):
            yaw_valid = g[list(yaw_cols)].replace([np.inf, -np.inf], np.nan).dropna()
            if len(yaw_valid) > 0:
                yf = yaw_valid.iloc[-1]
                odom_final_yaw_err_deg = abs(math.degrees(wrap_pi(float(yf["odom_yaw"] - yf["vicon_yaw"]))))
                corrected_final_yaw_err_deg = abs(math.degrees(wrap_pi(float(yf["corrected_yaw"] - yf["vicon_yaw"]))))
                yaw_improvement_deg = odom_final_yaw_err_deg - corrected_final_yaw_err_deg
                if odom_final_yaw_err_deg > 0:
                    yaw_improvement_percent = 100.0 * yaw_improvement_deg / odom_final_yaw_err_deg

        rows.append({
            "run_name": run_name,
            "terrain_name": terrain,
            "maneuver_name": maneuver,
            "n_samples": int(len(valid)),
            "odom_traj_rmse_m": odom_rmse,
            "corrected_traj_rmse_m": corr_rmse,
            "traj_rmse_improvement_percent": traj_improvement,
            "traj_rmse_delta_m": odom_rmse - corr_rmse,
            "traj_improved": bool(corr_rmse < odom_rmse),

            "odom_final_error_m": odom_final,
            "corrected_final_error_m": corr_final,
            "final_error_improvement_percent": final_improvement,
            "final_error_delta_m": odom_final - corr_final,
            "cnn_final_minus_odom_final_m": corr_final - odom_final,
            "final_position_improved": bool(corr_final < odom_final),

            "odom_max_position_error_m": odom_max,
            "corrected_max_position_error_m": corr_max,
            "max_position_error_improvement_percent": max_improvement,
            "max_position_error_delta_m": odom_max - corr_max,
            "max_position_improved": bool(corr_max < odom_max),

            "odom_final_yaw_error_deg": odom_final_yaw_err_deg,
            "corrected_final_yaw_error_deg": corrected_final_yaw_err_deg,
            "final_yaw_error_improvement_deg": yaw_improvement_deg,
            "final_yaw_error_improvement_percent": yaw_improvement_percent,

            "mean_gate_vx": float(g["learned_gate_vx"].mean()) if "learned_gate_vx" in g.columns else np.nan,
            "mean_gate_vy": float(g["learned_gate_vy"].mean()) if "learned_gate_vy" in g.columns else np.nan,
            "mean_applied_alpha_vx": float(g["applied_alpha_vx"].mean()) if "applied_alpha_vx" in g.columns else np.nan,
            "mean_applied_alpha_vy": float(g["applied_alpha_vy"].mean()) if "applied_alpha_vy" in g.columns else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["terrain_name", "maneuver_name", "traj_rmse_improvement_percent"],
        ascending=[True, True, False],
    )


def summarize_position_metrics_overall(run_metrics: pd.DataFrame) -> Dict:
    """Compact overall run-level summary for metrics.json."""
    if run_metrics.empty:
        return {}

    out = {
        "n_runs": int(len(run_metrics)),
        "num_traj_improved": int(run_metrics["traj_improved"].sum()) if "traj_improved" in run_metrics else None,
        "num_final_position_improved": int(run_metrics["final_position_improved"].sum()) if "final_position_improved" in run_metrics else None,
        "num_both_traj_and_final_improved": int((run_metrics["traj_improved"] & run_metrics["final_position_improved"]).sum()) if {"traj_improved", "final_position_improved"}.issubset(run_metrics.columns) else None,
        "mean_traj_rmse_improvement_percent": float(run_metrics["traj_rmse_improvement_percent"].mean()),
        "median_traj_rmse_improvement_percent": float(run_metrics["traj_rmse_improvement_percent"].median()),
        "worst_traj_rmse_improvement_percent": float(run_metrics["traj_rmse_improvement_percent"].min()),
        "mean_final_error_improvement_percent": float(run_metrics["final_error_improvement_percent"].mean()),
        "median_final_error_improvement_percent": float(run_metrics["final_error_improvement_percent"].median()),
        "worst_final_error_improvement_percent": float(run_metrics["final_error_improvement_percent"].min()),
        "mean_cnn_final_minus_odom_final_m": float(run_metrics["cnn_final_minus_odom_final_m"].mean()),
        "worst_cnn_final_minus_odom_final_m": float(run_metrics["cnn_final_minus_odom_final_m"].max()),
    }
    return out

def summarize_position_metrics_by_terrain(run_metrics: pd.DataFrame) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if run_metrics.empty:
        return out
    for terrain, g in run_metrics.groupby("terrain_name"):
        out[str(terrain)] = {
            "n_runs": int(len(g)),
            "mean_odom_traj_rmse_m": float(g["odom_traj_rmse_m"].mean()),
            "mean_corrected_traj_rmse_m": float(g["corrected_traj_rmse_m"].mean()),
            "mean_traj_rmse_improvement_percent": float(g["traj_rmse_improvement_percent"].mean()),
            "median_traj_rmse_improvement_percent": float(g["traj_rmse_improvement_percent"].median()),
            "num_runs_improved": int((g["traj_rmse_improvement_percent"] > 0).sum()),
            "mean_gate_vx": float(g["mean_gate_vx"].mean()),
            "mean_gate_vy": float(g["mean_gate_vy"].mean()),
        }
    return out


def summarize_position_metrics_by_maneuver(run_metrics: pd.DataFrame) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if run_metrics.empty or "maneuver_name" not in run_metrics.columns:
        return out
    for maneuver, g in run_metrics.groupby("maneuver_name"):
        out[str(maneuver)] = {
            "n_runs": int(len(g)),
            "mean_odom_traj_rmse_m": float(g["odom_traj_rmse_m"].mean()),
            "mean_corrected_traj_rmse_m": float(g["corrected_traj_rmse_m"].mean()),
            "mean_traj_rmse_improvement_percent": float(g["traj_rmse_improvement_percent"].mean()),
            "median_traj_rmse_improvement_percent": float(g["traj_rmse_improvement_percent"].median()),
            "num_runs_improved": int((g["traj_rmse_improvement_percent"] > 0).sum()),
            "mean_gate_vx": float(g["mean_gate_vx"].mean()),
            "mean_gate_vy": float(g["mean_gate_vy"].mean()),
        }
    return out


# -----------------------------
# Plotting
# -----------------------------

def plot_run(pred_df: pd.DataFrame, run_name: str, out_path: str):
    g = pred_df[pred_df["run_name"] == run_name].copy().sort_values("time_s")
    if len(g) == 0:
        print(f"WARNING: cannot plot run {run_name!r}; no predictions found.")
        return

    t = g["time_s"].values
    t_rel = t - t[0]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    ax_pos = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])

    if {"vicon_x", "vicon_y"}.issubset(g.columns):
        ax_pos.plot(g["vicon_x"], g["vicon_y"], label="Ground Truth")
    if {"odom_x", "odom_y"}.issubset(g.columns):
        ax_pos.plot(g["odom_x"], g["odom_y"], label="Go2 Odometry")
    ax_pos.plot(g["corrected_x"], g["corrected_y"], label="Corrected CNN")
    ax_pos.set_title("Position", fontsize=28)
    ax_pos.set_xlabel("X (meters)", fontsize=20)
    ax_pos.set_ylabel("Y (meters)", fontsize=20)
    ax_pos.grid(True)
    ax_pos.axis("equal")
    ax_pos.legend()

    if "vicon_x" in g.columns:
        ax_x.plot(t_rel, g["vicon_x"], label="Ground Truth")
    if "odom_x" in g.columns:
        ax_x.plot(t_rel, g["odom_x"], label="Go2 Odometry")
    ax_x.plot(t_rel, g["corrected_x"], label="Corrected CNN")
    ax_x.set_title("X over time", fontsize=28)
    ax_x.set_xlabel("Time (sec)", fontsize=20)
    ax_x.set_ylabel("X (meters)", fontsize=20)
    ax_x.grid(True)
    ax_x.legend()

    if "vicon_y" in g.columns:
        ax_y.plot(t_rel, g["vicon_y"], label="Ground Truth")
    if "odom_y" in g.columns:
        ax_y.plot(t_rel, g["odom_y"], label="Go2 Odometry")
    ax_y.plot(t_rel, g["corrected_y"], label="Corrected CNN")
    ax_y.set_title("Y over time", fontsize=28)
    ax_y.set_xlabel("Time (sec)", fontsize=20)
    ax_y.set_ylabel("Y (meters)", fontsize=20)
    ax_y.grid(True)
    ax_y.legend()

    terrain = g["terrain_name"].iloc[0] if "terrain_name" in g.columns else infer_terrain_from_run_name(run_name)
    maneuver = g["maneuver_name"].iloc[0] if "maneuver_name" in g.columns else infer_maneuver_from_run_name(run_name)
    fig.suptitle(f"{run_name} Estimate ({terrain}, {maneuver})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_velocity_diagnostics(pred_df: pd.DataFrame, run_name: str, out_path: str):
    g = pred_df[pred_df["run_name"] == run_name].copy().sort_values("time_s")
    if len(g) == 0:
        print(f"WARNING: cannot plot velocity diagnostics for {run_name!r}; no predictions found.")
        return

    t = g["time_s"].values
    t_rel = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, axis in zip(axes, ["vx", "vy", "wz"]):
        ax.plot(t_rel, g[f"true_{axis}"], label=f"Ground Truth {axis}")
        ax.plot(t_rel, g[f"odom_{axis}"], label=f"Go2 Odometry {axis}")
        ax.plot(t_rel, g[f"corrected_{axis}"], label=f"Corrected CNN {axis}")
        ax.set_ylabel(f"{axis} velocity")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (sec)")
    fig.suptitle(f"{run_name} Velocity Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residual_diagnostics(pred_df: pd.DataFrame, run_name: str, out_path: str):
    g = pred_df[pred_df["run_name"] == run_name].copy().sort_values("time_s")
    if len(g) == 0:
        print(f"WARNING: cannot plot residual diagnostics for {run_name!r}; no predictions found.")
        return

    t = g["time_s"].values
    t_rel = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_rel, g["true_residual_vx"], label="True residual vx")
    axes[0].plot(t_rel, g["pred_residual_vx_raw"], label="Predicted residual vx")
    axes[0].plot(t_rel, g["applied_residual_vx"], label="Applied residual vx")
    axes[0].set_ylabel("vx residual")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_rel, g["true_residual_vy"], label="True residual vy")
    axes[1].plot(t_rel, g["pred_residual_vy_raw"], label="Predicted residual vy")
    axes[1].plot(t_rel, g["applied_residual_vy"], label="Applied residual vy")
    axes[1].set_ylabel("vy residual")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_rel, g["learned_gate_vx"], label="Learned gate vx")
    axes[2].plot(t_rel, g["learned_gate_vy"], label="Learned gate vy")
    if "turn_scale" in g.columns:
        axes[2].plot(t_rel, g["turn_scale"], label="Optional turn scale")
    axes[2].plot(t_rel, g["applied_alpha_vx"], label="Applied alpha vx")
    axes[2].plot(t_rel, g["applied_alpha_vy"], label="Applied alpha vy")
    axes[2].set_ylabel("gate / alpha")
    axes[2].set_xlabel("Time (sec)")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(f"{run_name} Residual and Gate Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_index_html(run_names: List[str], out_dir: str):
    html_path = os.path.join(out_dir, "index.html")
    rows = []
    for run_name in run_names:
        s = safe_name(run_name)
        rows.append(f"""
        <h2>{run_name}</h2>
        <p>
          <a href="plot_{s}.png">Position plot</a> |
          <a href="velocity_diagnostics_{s}.png">Velocity diagnostics</a> |
          <a href="residual_diagnostics_{s}.png">Residual/gate diagnostics</a>
        </p>
        <img src="plot_{s}.png" style="width: 100%; max-width: 1200px;">
        <hr>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>V11 run plots</title>
</head>
<body>
  <h1>V11 run plots</h1>
  {''.join(rows)}
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    return html_path


def plot_all_runs(pred_df: pd.DataFrame, out_dir: str, position_only: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    run_names = sorted(pred_df["run_name"].dropna().unique().tolist())
    for i, run_name in enumerate(run_names, start=1):
        s = safe_name(run_name)
        print(f"  plotting [{i}/{len(run_names)}] {run_name}")
        plot_run(pred_df, run_name, os.path.join(out_dir, f"plot_{s}.png"))
        if not position_only:
            plot_velocity_diagnostics(pred_df, run_name, os.path.join(out_dir, f"velocity_diagnostics_{s}.png"))
            plot_residual_diagnostics(pred_df, run_name, os.path.join(out_dir, f"residual_diagnostics_{s}.png"))
    return make_index_html(run_names, out_dir)


# -----------------------------
# Reporting helpers
# -----------------------------

def print_velocity_metrics(metrics: Dict, split_name: str):
    print(f"\n{split_name} velocity MSE/RMSE:")
    for axis in ["vx", "vy", "wz"]:
        m = metrics["per_axis"][axis]
        print(
            f"  {axis}: baseline MSE={m['baseline_mse']:.6f}, corrected MSE={m['corrected_mse']:.6f}, "
            f"baseline RMSE={m['baseline_rmse']:.6f}, corrected RMSE={m['corrected_rmse']:.6f}, "
            f"improvement={m['mse_improvement_percent']:.2f}%"
        )

    m = metrics["overall_vxy_only"]
    print(
        f"  overall_vxy_only: baseline MSE={m['baseline_mse']:.6f}, corrected MSE={m['corrected_mse']:.6f}, "
        f"baseline RMSE={m['baseline_rmse']:.6f}, corrected RMSE={m['corrected_rmse']:.6f}, "
        f"improvement={m['mse_improvement_percent']:.2f}%"
    )

    m = metrics["overall"]
    print(
        f"  overall_with_wz_unchanged: baseline MSE={m['baseline_mse']:.6f}, corrected MSE={m['corrected_mse']:.6f}, "
        f"baseline RMSE={m['baseline_rmse']:.6f}, corrected RMSE={m['corrected_rmse']:.6f}, "
        f"improvement={m['mse_improvement_percent']:.2f}%"
    )


def print_position_summary(run_metrics: pd.DataFrame, split_name: str):
    print(f"\n{split_name} position metrics by run:")
    if run_metrics.empty:
        print("  Position metrics unavailable; missing pose columns.")
        return
    cols = [
        "run_name", "terrain_name", "maneuver_name",
        "odom_traj_rmse_m", "corrected_traj_rmse_m", "traj_rmse_improvement_percent",
        "odom_final_error_m", "corrected_final_error_m", "final_error_improvement_percent",
        "cnn_final_minus_odom_final_m", "odom_max_position_error_m", "corrected_max_position_error_m",
        "max_position_error_improvement_percent", "odom_final_yaw_error_deg", "corrected_final_yaw_error_deg",
        "mean_gate_vx", "mean_gate_vy",
    ]
    cols = [c for c in cols if c in run_metrics.columns]
    display = run_metrics[cols].sort_values("traj_rmse_improvement_percent", ascending=False)
    with pd.option_context("display.max_rows", 200, "display.width", 180):
        print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="processed_csv_all/all_runs_aligned_50hz_body_vicon.csv")
    parser.add_argument("--out-dir", default="models/v11_terrain_maneuver_gated")
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--test-runs", default="")
    parser.add_argument("--plot-run", default="")
    parser.add_argument("--include-runs", default="", help="Comma-separated substrings. Keep only run_name values containing at least one.")
    parser.add_argument("--exclude-runs", default="", help="Comma-separated substrings. Remove run_name values containing any.")
    parser.add_argument("--velocity-frame", choices=["body", "world"], default="body")

    # alpha is now max correction strength. Gate learns the actual applied fraction.
    parser.add_argument("--alpha", type=float, default=0.6, help="Maximum scale applied to predicted vx/vy residuals. Learned gates multiply this.")
    parser.add_argument("--clip-residual", type=float, default=0.5, help="Clip raw predicted vx/vy residuals in m/s. Set 0 to disable.")
    parser.add_argument("--gate-min", type=float, default=0.0, help="Minimum learned gate value. Use 0.0 for full suppressibility.")
    parser.add_argument("--gate-bias-init", type=float, default=0.0, help="Initial final gate bias. 0.0 -> gates near 0.5; negative -> more conservative.")
    parser.add_argument("--residual-loss-weight", type=float, default=0.20, help="Auxiliary residual-shape loss weight.")
    parser.add_argument("--gate-penalty", type=float, default=0.002, help="Small penalty on mean gate to prefer safe corrections unless useful.")

    # Optional hand-coded turn gate. Default off, because 09 hurt watery_m.
    parser.add_argument("--turn-gate-wz", type=float, default=0.0, help="Optional extra attenuation when abs(odom_wz) is high. 0 disables it.")
    parser.add_argument("--turn-gate-power", type=float, default=1.0)

    parser.add_argument("--use-vicon-pose-context", action="store_true", help="Diagnostic only. Do not use for deployable model.")
    parser.add_argument("--no-terrain-context", action="store_true", help="Disable terrain one-hot context features.")
    parser.add_argument("--no-maneuver-context", action="store_true", help="Disable maneuver one-hot context features.")
    parser.add_argument("--no-derived-motion-context", action="store_true", help="Disable derived motion context features.")
    parser.add_argument("--no-balanced-group-loss", action="store_true", help="Disable inverse-frequency group-balanced loss.")
    parser.add_argument("--balance-group", choices=["terrain_name", "maneuver_name", "terrain_maneuver"], default="terrain_maneuver", help="Group used for inverse-frequency training weights.")
    parser.add_argument("--group-weight-cap", type=float, default=5.0, help="Cap for inverse-frequency group sample weights.")
    parser.add_argument("--random-split", action="store_true", help="Use random run split instead of terrain-stratified split.")
    parser.add_argument("--plot-all-test-runs", action="store_true", default=True, help="Generate plots for every test run. Default: enabled.")
    parser.add_argument("--no-plot-all-test-runs", dest="plot_all_test_runs", action="store_false", help="Disable all-test plotting.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--non-deterministic", action="store_true", help="Disable deterministic PyTorch settings. Not recommended for model comparison.")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    deterministic = not args.non_deterministic
    set_seed(args.seed, deterministic=deterministic)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CSV:", args.csv)
    df = pd.read_csv(args.csv)

    require_columns(df, ["run_name", "time_s"], "run metadata")
    require_columns(df, BASE_RAW_FEATURE_COLS, "base raw input feature")
    require_columns(df, BASELINE_COLS + TRUTH_COLS, "baseline/truth")

    df = df.copy()
    df["residual_vx"] = df["vicon_vx"] - df["odom_vx"]
    df["residual_vy"] = df["vicon_vy"] - df["odom_vy"]
    df["residual_wz"] = df["vicon_wz"] - df["odom_wz"]

    df = add_terrain_context_columns(df)
    df = add_maneuver_context_columns(df)
    df = add_derived_motion_context_columns(df)

    all_runs = sorted(df["run_name"].dropna().unique().tolist())
    selected_runs = filtered_run_names(all_runs, args.include_runs, args.exclude_runs)
    df = df[df["run_name"].isin(selected_runs)].copy()

    terrain_context = not args.no_terrain_context
    maneuver_context = not args.no_maneuver_context
    derived_motion_context = not args.no_derived_motion_context
    balanced_group_loss = not args.no_balanced_group_loss

    feature_cols = select_feature_cols(df, args.use_vicon_pose_context, terrain_context, maneuver_context, derived_motion_context)
    require_columns(df, feature_cols, "selected feature")

    needed_cols = ["run_name", "terrain_name", "maneuver_name", "time_s"] + feature_cols + BASELINE_COLS + TRUTH_COLS + ALL_RESIDUAL_COLS
    needed_cols = list(dict.fromkeys(needed_cols))
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    run_names = sorted(df["run_name"].unique().tolist())
    train_runs, val_runs, test_runs = split_runs(
        run_names,
        args.test_runs,
        args.val_fraction,
        args.seed,
        stratified_by_terrain=not args.random_split,
    )

    print("\nRun filtering:")
    print("  Include substrings:", args.include_runs or "(none)")
    print("  Exclude substrings:", args.exclude_runs or "(none)")
    print("  Selected runs:", run_names)

    print("\nTerrain counts by run:")
    terrain_run_counts = pd.Series({r: infer_terrain_from_run_name(r) for r in run_names}).value_counts()
    for terrain, count in terrain_run_counts.items():
        print(f"  {terrain}: {count} runs")

    print("\nRun split:")
    print("  Train runs:", train_runs)
    print("  Val runs:  ", val_runs)
    print("  Test runs: ", test_runs)

    print(f"\nRows after filtering/dropna: {len(df)}")
    print(f"Input features: {len(feature_cols)}")
    print("Features:")
    for c in feature_cols:
        print("  ", c)

    print(f"\nWindow length: {args.window} samples ({args.window / 50.0:.2f} sec at 50 Hz, if data is 50 Hz)")
    print("Version 11T correction/evaluation:")
    print("  temporal readout: avg+max pooling over CNN feature map")
    print("  corrected_vxy = odom_vxy + alpha * learned_gate_vxy * clipped_pred_residual_vxy")
    print("  corrected_wz  = odom_wz")
    print(f"  alpha(max)={args.alpha}")
    print(f"  clip_residual={args.clip_residual}")
    print(f"  gate_min={args.gate_min}")
    print(f"  residual_loss_weight={args.residual_loss_weight}")
    print(f"  gate_penalty={args.gate_penalty}")
    print(f"  terrain_context={terrain_context}")
    print(f"  maneuver_context={maneuver_context}")
    print(f"  derived_motion_context={derived_motion_context}")
    print(f"  balanced_group_loss={balanced_group_loss} by {args.balance_group}")
    print(f"  optional turn gate: turn_gate_wz={args.turn_gate_wz}, turn_gate_power={args.turn_gate_power}")

    train_df = df[df["run_name"].isin(train_runs)].copy()
    x_scaler, y_scaler = fit_scalers(train_df, feature_cols)

    train_data = make_windows_by_run(df, train_runs, feature_cols, x_scaler, y_scaler, args.window)
    val_data = make_windows_by_run(df, val_runs, feature_cols, x_scaler, y_scaler, args.window) if val_runs else None
    test_data = make_windows_by_run(df, test_runs, feature_cols, x_scaler, y_scaler, args.window) if test_runs else None

    if balanced_group_loss:
        train_data = apply_group_balanced_weights(train_data, group_col=args.balance_group, cap=args.group_weight_cap)

    print("\nWindowed shapes:")
    print("  Train X:", train_data.X.shape)
    if val_data is not None:
        print("  Val X:  ", val_data.X.shape)
    if test_data is not None:
        print("  Test X: ", test_data.X.shape)

    if balanced_group_loss:
        print("\nTerrain-balanced train sample weights:")
        w_summary = train_data.meta.groupby("terrain_name")["sample_weight"].agg(["count", "mean", "min", "max"])
        with pd.option_context("display.width", 140):
            print(w_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        Go2WindowDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=loader_generator,
    )

    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(
            Go2WindowDataset(val_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TerrainManeuverGatedCNN1D(
        input_features=len(feature_cols),
        dropout=args.dropout,
        gate_min=args.gate_min,
        gate_bias_init=args.gate_bias_init,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    loss_config = LossConfig(
        alpha=args.alpha,
        clip_residual=args.clip_residual,
        residual_loss_weight=args.residual_loss_weight,
        gate_penalty=args.gate_penalty,
        y_mean=torch.tensor(y_scaler.mean_, dtype=torch.float32).view(1, -1),
        y_scale=torch.tensor(y_scaler.scale_, dtype=torch.float32).view(1, -1),
    )

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "v11t_terrain_maneuver_gated_avgmaxpool_best.pt")
    epochs_without_improvement = 0
    history = []

    print("\nTraining Version 11T terrain+maneuver-aware learned-gate VXY residual model")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Deterministic training: {deterministic}")

    for epoch in range(1, args.epochs + 1):
        train_parts = train_one_epoch(model, train_loader, optimizer, loss_config, device, grad_clip=1.0)
        val_parts = evaluate_loss(model, val_loader, loss_config, device) if val_loader is not None else train_parts
        scheduler.step(val_parts["loss"])

        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            **{f"train_{k}": v for k, v in train_parts.items()},
            **{f"val_{k}": v for k, v in val_parts.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_parts['loss']:.6f}, train_vel={train_parts['velocity_loss']:.6f}, "
            f"train_gate={train_parts['gate_mean']:.3f}, "
            f"val_loss={val_parts['loss']:.6f}, val_vel={val_parts['velocity_loss']:.6f}, "
            f"val_gate={val_parts['gate_mean']:.3f}, lr={lr_now:.2e}"
        )

        if val_parts["loss"] < best_val - 1e-5:
            best_val = val_parts["loss"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_features": feature_cols,
                "window": args.window,
                "train_runs": train_runs,
                "val_runs": val_runs,
                "test_runs": test_runs,
                "velocity_frame": args.velocity_frame,
                "alpha": args.alpha,
                "clip_residual": args.clip_residual,
                "gate_min": args.gate_min,
                "target": TARGET_RESIDUAL_COLS,
                "seed": args.seed,
                "deterministic": deterministic,
                "terrain_context": terrain_context,
                "maneuver_context": maneuver_context,
                "derived_motion_context": derived_motion_context,
                "terrain_names": TERRAIN_NAMES,
                "maneuver_names": MANEUVER_NAMES,
                "loss_config": {
                    "residual_loss_weight": args.residual_loss_weight,
                    "gate_penalty": args.gate_penalty,
                },
            }, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best val loss={best_val:.6f}")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_pred = predict_dataset(
        model, train_data, y_scaler, args.batch_size, device,
        args.alpha, args.clip_residual, args.turn_gate_wz, args.turn_gate_power,
    )
    train_pred = reconstruct_paths_for_predictions(train_pred, args.velocity_frame)

    if test_data is not None:
        test_pred = predict_dataset(
            model, test_data, y_scaler, args.batch_size, device,
            args.alpha, args.clip_residual, args.turn_gate_wz, args.turn_gate_power,
        )
        test_pred = reconstruct_paths_for_predictions(test_pred, args.velocity_frame)
    else:
        test_pred = pd.DataFrame()

    train_metrics = velocity_metrics(train_pred)
    test_metrics = velocity_metrics(test_pred) if not test_pred.empty else {}
    train_metrics_by_terrain = velocity_metrics_by_terrain(train_pred)
    test_metrics_by_terrain = velocity_metrics_by_terrain(test_pred) if not test_pred.empty else {}

    train_run_position_metrics = run_position_metrics(train_pred)
    test_run_position_metrics = run_position_metrics(test_pred) if not test_pred.empty else pd.DataFrame()

    print_velocity_metrics(train_metrics, "Train")
    if test_metrics:
        print_velocity_metrics(test_metrics, "Test")

    print_position_summary(train_run_position_metrics, "Train")
    if not test_run_position_metrics.empty:
        print_position_summary(test_run_position_metrics, "Test")

    train_pred_path = os.path.join(args.out_dir, "train_predictions.csv")
    test_pred_path = os.path.join(args.out_dir, "test_predictions.csv")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    history_path = os.path.join(args.out_dir, "training_history.csv")
    scaler_path = os.path.join(args.out_dir, "scalers.joblib")
    train_pos_metrics_path = os.path.join(args.out_dir, "run_position_metrics_train.csv")
    test_pos_metrics_path = os.path.join(args.out_dir, "run_position_metrics_test.csv")

    train_pred.to_csv(train_pred_path, index=False)
    if not test_pred.empty:
        test_pred.to_csv(test_pred_path, index=False)
    else:
        pd.DataFrame().to_csv(test_pred_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    train_run_position_metrics.to_csv(train_pos_metrics_path, index=False)
    test_run_position_metrics.to_csv(test_pos_metrics_path, index=False)

    joblib.dump({
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "features": feature_cols,
        "terrain_names": TERRAIN_NAMES,
        "maneuver_names": MANEUVER_NAMES,
        "terrain_context": terrain_context,
        "maneuver_context": maneuver_context,
        "derived_motion_context": derived_motion_context,
    }, scaler_path)

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "train_by_terrain": train_metrics_by_terrain,
        "test_by_terrain": test_metrics_by_terrain,
        "train_position_overall": summarize_position_metrics_overall(train_run_position_metrics),
        "test_position_overall": summarize_position_metrics_overall(test_run_position_metrics),
        "train_position_by_terrain": summarize_position_metrics_by_terrain(train_run_position_metrics),
        "test_position_by_terrain": summarize_position_metrics_by_terrain(test_run_position_metrics),
        "train_position_by_maneuver": summarize_position_metrics_by_maneuver(train_run_position_metrics),
        "test_position_by_maneuver": summarize_position_metrics_by_maneuver(test_run_position_metrics),
        "train_runs": train_runs,
        "val_runs": val_runs,
        "test_runs": test_runs,
        "selected_runs": run_names,
        "window": args.window,
        "features": feature_cols,
        "target": TARGET_RESIDUAL_COLS,
        "alpha_max": args.alpha,
        "clip_residual": args.clip_residual,
        "gate_min": args.gate_min,
        "residual_loss_weight": args.residual_loss_weight,
        "gate_penalty": args.gate_penalty,
        "balanced_group_loss": balanced_group_loss,
        "balance_group": args.balance_group,
        "terrain_context": terrain_context,
        "maneuver_context": maneuver_context,
        "derived_motion_context": derived_motion_context,
        "velocity_frame_for_position_integration": args.velocity_frame,
        "note": "Version 11T is V11M with avg+max temporal pooling. It predicts vx/vy residuals plus learned per-axis gates using terrain and maneuver context. wz is intentionally unchanged.",
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Default plot: requested run if available; else first test run; else first train run.
    available_plot_df = test_pred if not test_pred.empty else train_pred
    plot_run_name = args.plot_run.strip()
    if plot_run_name:
        combined_pred = pd.concat([train_pred, test_pred], ignore_index=True) if not test_pred.empty else train_pred
        plot_df = combined_pred
    else:
        plot_df = available_plot_df
        plot_run_name = test_runs[0] if test_runs else train_runs[0]

    s = safe_name(plot_run_name)
    plot_path = os.path.join(args.out_dir, f"plot_{s}.png")
    velocity_plot_path = os.path.join(args.out_dir, f"velocity_diagnostics_{s}.png")
    residual_plot_path = os.path.join(args.out_dir, f"residual_diagnostics_{s}.png")

    plot_run(plot_df, plot_run_name, plot_path)
    plot_velocity_diagnostics(plot_df, plot_run_name, velocity_plot_path)
    plot_residual_diagnostics(plot_df, plot_run_name, residual_plot_path)

    index_html_path: Optional[str] = None
    if args.plot_all_test_runs and not test_pred.empty:
        all_plots_dir = os.path.join(args.out_dir, "all_test_plots")
        print("\nGenerating all test plots:", all_plots_dir)
        index_html_path = plot_all_runs(test_pred, all_plots_dir, position_only=False)

    print("\nSaved files:")
    print("  Best model:              ", best_path)
    print("  Scalers:                 ", scaler_path)
    print("  Train predictions:       ", train_pred_path)
    print("  Test predictions:        ", test_pred_path)
    print("  Metrics:                 ", metrics_path)
    print("  History:                 ", history_path)
    print("  Train position metrics:  ", train_pos_metrics_path)
    print("  Test position metrics:   ", test_pos_metrics_path)
    print("  Position plot:           ", plot_path)
    print("  Velocity plot:           ", velocity_plot_path)
    print("  Residual/gate plot:      ", residual_plot_path)
    if index_html_path:
        print("  All test plots index:    ", index_html_path)

    if test_metrics and test_metrics["overall_vxy_only"]["corrected_mse"] > test_metrics["overall_vxy_only"]["baseline_mse"]:
        print("\nWARNING: Corrected VXY test MSE is worse than baseline.")
        print("Try lower --alpha, higher --gate-penalty, stronger --clip-residual, or inspect per-terrain metrics.")

    if not test_run_position_metrics.empty:
        n_improved = int((test_run_position_metrics["traj_rmse_improvement_percent"] > 0).sum())
        n_total = int(len(test_run_position_metrics))
        print(f"\nPosition summary: {n_improved}/{n_total} test runs improved in trajectory RMSE.")


if __name__ == "__main__":
    main()
