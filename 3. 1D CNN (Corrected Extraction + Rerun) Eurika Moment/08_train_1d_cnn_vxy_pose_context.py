#!/usr/bin/env python3
"""
Train a diagnostic 1D CNN residual odometry correction model for Go2.

This version is intentionally conservative:

  1. It predicts only vx/vy residuals:
       residual_vx = vicon_vx - odom_vx
       residual_vy = vicon_vy - odom_vy

  2. It does NOT learn wz correction:
       corrected_wz = odom_wz

  3. It includes odometry pose context as inputs when available:
       odom_x, odom_y, odom_yaw

  4. It supports terrain/run filtering so you can test easier surfaces first:
       --include-runs slippery,water
       --exclude-runs rocky

     Matching is substring-based against run_name.

  5. It has correction safety controls:
       --alpha scales the predicted residual before applying it
       --clip-residual clips predicted vx/vy residuals in m/s

  6. It saves two plots:
       plot_<run>.png                       integrated position plot
       velocity_diagnostics_<run>.png       vx/vy/wz time-series plot
       residual_diagnostics_<run>.png       true vs predicted vx/vy residuals

Recommended first run:

python 08_train_1d_cnn_vxy_pose_context.py \
  --csv processed_csv_all/all_runs_aligned_50hz.csv \
  --out-dir models/all_runs_1d_cnn_vxy_pose_context \
  --exclude-runs rocky \
  --alpha 0.3 \
  --clip-residual 0.5

For slippery/watery-only testing, try for example:

python 08_train_1d_cnn_vxy_pose_context.py \
  --csv processed_csv_all/all_runs_aligned_50hz.csv \
  --out-dir models/slippery_water_vxy_pose_context \
  --include-runs slippery,water,wet \
  --alpha 0.3 \
  --clip-residual 0.5
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

OPTIONAL_CONTEXT_COLS = [
    "odom_x", "odom_y", "odom_yaw",
    "vicon_x", "vicon_y", "vicon_yaw",   # only used if --use-vicon-pose-context is set; default false
    "cmd_vx", "cmd_vy", "cmd_wz",
    "command_vx", "command_vy", "command_wz",
    "target_vx", "target_vy", "target_wz",
]

BASELINE_COLS = ["odom_vx", "odom_vy", "odom_wz"]
TRUTH_COLS = ["vicon_vx", "vicon_vy", "vicon_wz"]
TARGET_RESIDUAL_COLS = ["residual_vx", "residual_vy"]
ALL_RESIDUAL_COLS = ["residual_vx", "residual_vy", "residual_wz"]
POSITION_COLS = ["odom_x", "odom_y", "odom_yaw", "vicon_x", "vicon_y", "vicon_yaw", "time_s"]


class Go2WindowDataset(Dataset):
    def __init__(self, X, baseline, residual_vxy, truth, meta):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.residual_vxy = torch.tensor(residual_vxy, dtype=torch.float32)
        self.truth = torch.tensor(truth, dtype=torch.float32)
        self.meta = meta

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.baseline[idx], self.residual_vxy[idx], self.truth[idx]


class SimpleCNN1D(nn.Module):
    def __init__(self, input_features: int, output_dim: int = 2, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
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

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class WindowedData:
    X: np.ndarray
    baseline: np.ndarray
    residual_vxy: np.ndarray
    truth: np.ndarray
    meta: pd.DataFrame


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_columns(df: pd.DataFrame, cols: List[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


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


def split_runs(run_names: List[str], test_runs_arg: str, val_fraction: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    all_runs = list(run_names)

    if test_runs_arg:
        test_runs = [r.strip() for r in test_runs_arg.split(",") if r.strip()]
        unknown = [r for r in test_runs if r not in all_runs]
        if unknown:
            raise ValueError(f"Requested --test-runs not found after filtering: {unknown}\nAvailable runs: {all_runs}")
        remaining = [r for r in all_runs if r not in test_runs]
    else:
        rng = random.Random(seed)
        shuffled = all_runs[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(0.20 * len(shuffled)))) if len(shuffled) > 1 else 1
        test_runs = shuffled[:n_test]
        remaining = shuffled[n_test:]

    rng = random.Random(seed + 1)
    rng.shuffle(remaining)
    n_val = max(1, int(round(val_fraction * len(remaining)))) if len(remaining) > 1 else 0
    val_runs = remaining[:n_val]
    train_runs = remaining[n_val:]

    if len(train_runs) == 0:
        raise ValueError("No training runs left after split. Use fewer --test-runs or loosen include/exclude filters.")

    return train_runs, val_runs, test_runs


def select_feature_cols(df: pd.DataFrame, use_vicon_pose_context: bool) -> List[str]:
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

    # Usually do NOT use vicon pose as an input, because it will not exist in deployment.
    if use_vicon_pose_context:
        for c in ["vicon_x", "vicon_y", "vicon_yaw"]:
            if c in df.columns:
                feature_cols.append(c)

    # Keep order but remove duplicates.
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

            row = run_df.iloc[i]
            meta_rows.append({
                "run_name": run_name,
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

    X_windows = np.transpose(X_windows, (0, 2, 1))
    meta = pd.DataFrame(meta_rows)

    return WindowedData(X_windows, baseline_targets, residual_targets, truth_targets, meta)


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, _, residual_batch, _ in loader:
        X_batch = X_batch.to(device)
        residual_batch = residual_batch.to(device)

        pred = model(X_batch)
        loss = loss_fn(pred, residual_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_scaled_loss(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, _, residual_batch, _ in loader:
            X_batch = X_batch.to(device)
            residual_batch = residual_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, residual_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def predict_dataset(
    model,
    data: WindowedData,
    y_scaler: StandardScaler,
    batch_size: int,
    device: str,
    alpha: float,
    clip_residual: float,
) -> pd.DataFrame:
    ds = Go2WindowDataset(data.X, data.baseline, data.residual_vxy, data.truth, data.meta)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    pred_residuals = []
    model.eval()

    with torch.no_grad():
        for X_batch, _, _, _ in loader:
            X_batch = X_batch.to(device)
            pred_scaled = model(X_batch).cpu().numpy()
            pred = y_scaler.inverse_transform(pred_scaled)
            pred_residuals.append(pred)

    pred_residuals = np.vstack(pred_residuals)

    if clip_residual > 0:
        pred_residuals = np.clip(pred_residuals, -clip_residual, clip_residual)

    baseline = data.baseline
    truth = data.truth

    corrected = baseline.copy()
    corrected[:, 0] = baseline[:, 0] + alpha * pred_residuals[:, 0]
    corrected[:, 1] = baseline[:, 1] + alpha * pred_residuals[:, 1]
    corrected[:, 2] = baseline[:, 2]  # intentionally unchanged

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
    out["applied_residual_vx"] = alpha * pred_residuals[:, 0]
    out["applied_residual_vy"] = alpha * pred_residuals[:, 1]
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

    fig.suptitle(f"{run_name} Estimate", fontsize=14)
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

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

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
    axes[1].set_xlabel("Time (sec)")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"{run_name} Residual Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_metrics(metrics: Dict, split_name: str):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="processed_csv_all/all_runs_aligned_50hz.csv")
    parser.add_argument("--out-dir", default="models/all_runs_1d_cnn_vxy_pose_context")
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
    parser.add_argument("--alpha", type=float, default=0.3, help="Scale applied to predicted vx/vy residuals.")
    parser.add_argument("--clip-residual", type=float, default=0.5, help="Clip raw predicted vx/vy residuals in m/s. Set 0 to disable.")
    parser.add_argument("--use-vicon-pose-context", action="store_true", help="Diagnostic only. Do not use for deployable model.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
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

    all_runs = sorted(df["run_name"].dropna().unique().tolist())
    selected_runs = filtered_run_names(all_runs, args.include_runs, args.exclude_runs)

    df = df[df["run_name"].isin(selected_runs)].copy()

    feature_cols = select_feature_cols(df, args.use_vicon_pose_context)
    require_columns(df, feature_cols, "selected feature")

    needed_cols = ["run_name", "time_s"] + feature_cols + BASELINE_COLS + TRUTH_COLS + ALL_RESIDUAL_COLS
    df = df.dropna(subset=list(dict.fromkeys(needed_cols))).reset_index(drop=True)

    run_names = sorted(df["run_name"].unique().tolist())
    train_runs, val_runs, test_runs = split_runs(run_names, args.test_runs, args.val_fraction, args.seed)

    print("\nRun filtering:")
    print("  Include substrings:", args.include_runs or "(none)")
    print("  Exclude substrings:", args.exclude_runs or "(none)")
    print("  Selected runs:", run_names)

    print("\nRun split:")
    print("  Train runs:", train_runs)
    print("  Val runs:  ", val_runs)
    print("  Test runs: ", test_runs)

    print(f"\nRows after filtering/dropna: {len(df)}")
    print(f"Input features: {len(feature_cols)}")
    print("Features:")
    for c in feature_cols:
        print("  ", c)

    print(f"\nWindow length: {args.window} samples ({args.window / 50.0:.2f} sec at 50 Hz)")
    print(f"Residual application: corrected_vxy = odom_vxy + alpha * clipped_pred_residual_vxy")
    print(f"  alpha={args.alpha}")
    print(f"  clip_residual={args.clip_residual}")
    print("  corrected_wz = odom_wz")

    train_df = df[df["run_name"].isin(train_runs)].copy()
    x_scaler, y_scaler = fit_scalers(train_df, feature_cols)

    train_data = make_windows_by_run(df, train_runs, feature_cols, x_scaler, y_scaler, args.window)
    val_data = make_windows_by_run(df, val_runs, feature_cols, x_scaler, y_scaler, args.window) if val_runs else None
    test_data = make_windows_by_run(df, test_runs, feature_cols, x_scaler, y_scaler, args.window)

    print("\nWindowed shapes:")
    print("  Train X:", train_data.X.shape)
    if val_data is not None:
        print("  Val X:  ", val_data.X.shape)
    print("  Test X: ", test_data.X.shape)

    train_loader = DataLoader(
        Go2WindowDataset(train_data.X, train_data.baseline, train_data.residual_vxy, train_data.truth, train_data.meta),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_data is not None:
        val_loader = DataLoader(
            Go2WindowDataset(val_data.X, val_data.baseline, val_data.residual_vxy, val_data.truth, val_data.meta),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN1D(input_features=len(feature_cols), output_dim=2, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "all_runs_1d_cnn_vxy_pose_context_best.pt")
    epochs_without_improvement = 0
    history = []

    print("\nTraining VXY-only 1D CNN residual correction model")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip=1.0)
        val_loss = evaluate_scaled_loss(model, val_loader, loss_fn, device) if val_loader is not None else train_loss
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss_scaled": train_loss, "val_loss_scaled": val_loss, "lr": lr_now})

        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={lr_now:.2e}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
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
                "target": TARGET_RESIDUAL_COLS,
            }, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best val loss={best_val:.6f}")
                break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_pred = predict_dataset(model, train_data, y_scaler, args.batch_size, device, args.alpha, args.clip_residual)
    test_pred = predict_dataset(model, test_data, y_scaler, args.batch_size, device, args.alpha, args.clip_residual)

    train_pred = reconstruct_paths_for_predictions(train_pred, args.velocity_frame)
    test_pred = reconstruct_paths_for_predictions(test_pred, args.velocity_frame)

    train_metrics = velocity_metrics(train_pred)
    test_metrics = velocity_metrics(test_pred)

    print_metrics(train_metrics, "Train")
    print_metrics(test_metrics, "Test")

    train_pred_path = os.path.join(args.out_dir, "train_predictions.csv")
    test_pred_path = os.path.join(args.out_dir, "test_predictions.csv")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    history_path = os.path.join(args.out_dir, "training_history.csv")
    scaler_path = os.path.join(args.out_dir, "scalers.joblib")

    train_pred.to_csv(train_pred_path, index=False)
    test_pred.to_csv(test_pred_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "features": feature_cols}, scaler_path)

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
        "train_runs": train_runs,
        "val_runs": val_runs,
        "test_runs": test_runs,
        "selected_runs": run_names,
        "window": args.window,
        "features": feature_cols,
        "target": TARGET_RESIDUAL_COLS,
        "alpha": args.alpha,
        "clip_residual": args.clip_residual,
        "velocity_frame_for_position_integration": args.velocity_frame,
        "note": "This diagnostic model corrects vx/vy only. wz is intentionally unchanged. Prefer overall_vxy_only for velocity evaluation.",
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    plot_run_name = args.plot_run.strip() if args.plot_run.strip() else test_runs[0]
    safe_name = plot_run_name.replace("/", "_").replace(" ", "_")

    plot_path = os.path.join(args.out_dir, f"plot_{safe_name}.png")
    velocity_plot_path = os.path.join(args.out_dir, f"velocity_diagnostics_{safe_name}.png")
    residual_plot_path = os.path.join(args.out_dir, f"residual_diagnostics_{safe_name}.png")

    plot_run(test_pred, plot_run_name, plot_path)
    plot_velocity_diagnostics(test_pred, plot_run_name, velocity_plot_path)
    plot_residual_diagnostics(test_pred, plot_run_name, residual_plot_path)

    print("\nSaved files:")
    print("  Best model:          ", best_path)
    print("  Scalers:             ", scaler_path)
    print("  Train predictions:   ", train_pred_path)
    print("  Test predictions:    ", test_pred_path)
    print("  Metrics:             ", metrics_path)
    print("  History:             ", history_path)
    print("  Position plot:       ", plot_path)
    print("  Velocity plot:       ", velocity_plot_path)
    print("  Residual plot:       ", residual_plot_path)

    if test_metrics["overall_vxy_only"]["corrected_mse"] > test_metrics["overall_vxy_only"]["baseline_mse"]:
        print("\nWARNING: Corrected VXY test MSE is worse than baseline.")
        print("Try lower --alpha, stronger --clip-residual, terrain-specific splits, or more train runs.")


if __name__ == "__main__":
    main()
