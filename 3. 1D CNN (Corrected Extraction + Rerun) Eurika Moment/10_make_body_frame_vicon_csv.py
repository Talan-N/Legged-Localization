#!/usr/bin/env python3
"""
Create a frame-consistent CSV for Go2 odometry correction.

Problem this fixes:
  Your current vicon_vx/vicon_vy appear to be WORLD-frame velocities, while
  odom_vx/odom_vy appear to be BODY-frame velocities.

This script loads all_runs_aligned_50hz.csv, computes Vicon BODY-frame velocity
from vicon_x, vicon_y, and vicon_yaw, then saves a corrected CSV with:

  vicon_body_vx
  vicon_body_vy
  vicon_body_wz

and optionally overwrites the training velocity columns:

  vicon_vx = vicon_body_vx
  vicon_vy = vicon_body_vy
  vicon_wz = vicon_body_wz

By default, it DOES NOT overwrite. Use --overwrite-vicon-velocity-cols if you
want the output CSV to be directly compatible with your existing training scripts.

Recommended use:

python 10_make_body_frame_vicon_csv.py \
  --csv processed_csv_all/all_runs_aligned_50hz.csv \
  --out processed_csv_all/all_runs_aligned_50hz_body_vicon.csv \
  --overwrite-vicon-velocity-cols

Then train using:

python 08_train_1d_cnn_vxy_pose_context.py \
  --csv processed_csv_all/all_runs_aligned_50hz_body_vicon.csv \
  --out-dir models/vxy_body_frame_test \
  --include-runs slippery,water,wet \
  --alpha 0.3 \
  --clip-residual 0.5
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLS = [
    "run_name", "time_s",
    "vicon_x", "vicon_y", "vicon_yaw",
]

ORIGINAL_VEL_COLS = [
    "vicon_vx", "vicon_vy", "vicon_wz",
    "odom_vx", "odom_vy", "odom_wz",
]


def unwrap_angle(theta):
    return np.unwrap(theta.astype(float))


def safe_gradient(y, t):
    y = y.astype(float)
    t = t.astype(float)

    dt = np.diff(t)
    if len(dt) == 0 or np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        return np.gradient(y)

    return np.gradient(y, t)


def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def world_to_body(vx_w, vy_w, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)

    vx_b = c * vx_w + s * vy_w
    vy_b = -s * vx_w + c * vy_w
    return vx_b, vy_b


def compute_body_vicon_for_run(g, smooth_window):
    g = g.sort_values("time_s").copy()

    t = g["time_s"].values.astype(float)
    yaw = unwrap_angle(g["vicon_yaw"].values.astype(float))

    vx_world = safe_gradient(g["vicon_x"].values, t)
    vy_world = safe_gradient(g["vicon_y"].values, t)
    wz_body = safe_gradient(yaw, t)

    vx_world = moving_average(vx_world, smooth_window)
    vy_world = moving_average(vy_world, smooth_window)
    wz_body = moving_average(wz_body, smooth_window)

    vx_body, vy_body = world_to_body(vx_world, vy_world, yaw)

    g["vicon_world_vx_from_pos"] = vx_world
    g["vicon_world_vy_from_pos"] = vy_world
    g["vicon_body_vx"] = vx_body
    g["vicon_body_vy"] = vy_body
    g["vicon_body_wz"] = wz_body
    g["vicon_yaw_unwrapped"] = yaw

    return g


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def summarize(df):
    metrics = {}

    if {"vicon_vx_original", "vicon_vy_original"}.issubset(df.columns):
        metrics["original_csv_vicon_vs_recomputed_world"] = {
            "vx_rmse": rmse(df["vicon_vx_original"], df["vicon_world_vx_from_pos"]),
            "vy_rmse": rmse(df["vicon_vy_original"], df["vicon_world_vy_from_pos"]),
            "vx_corr": corr(df["vicon_vx_original"], df["vicon_world_vx_from_pos"]),
            "vy_corr": corr(df["vicon_vy_original"], df["vicon_world_vy_from_pos"]),
        }
        metrics["original_csv_vicon_vs_recomputed_body"] = {
            "vx_rmse": rmse(df["vicon_vx_original"], df["vicon_body_vx"]),
            "vy_rmse": rmse(df["vicon_vy_original"], df["vicon_body_vy"]),
            "vx_corr": corr(df["vicon_vx_original"], df["vicon_body_vx"]),
            "vy_corr": corr(df["vicon_vy_original"], df["vicon_body_vy"]),
        }

    if {"odom_vx", "odom_vy"}.issubset(df.columns):
        metrics["odom_vs_recomputed_body_vicon"] = {
            "vx_rmse": rmse(df["odom_vx"], df["vicon_body_vx"]),
            "vy_rmse": rmse(df["odom_vy"], df["vicon_body_vy"]),
            "vx_corr": corr(df["odom_vx"], df["vicon_body_vx"]),
            "vy_corr": corr(df["odom_vy"], df["vicon_body_vy"]),
        }

    return metrics


def plot_check(df, run_name, out_path):
    g = df[df["run_name"] == run_name].copy().sort_values("time_s")
    if len(g) == 0:
        return

    t = g["time_s"].values
    t_rel = t - t[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(t_rel, g["vicon_body_vx"], label="new vicon_body_vx")
    if "vicon_vx_original" in g.columns:
        axes[0].plot(t_rel, g["vicon_vx_original"], label="original CSV vicon_vx")
    if "odom_vx" in g.columns:
        axes[0].plot(t_rel, g["odom_vx"], label="odom_vx")
    axes[0].set_ylabel("vx")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_rel, g["vicon_body_vy"], label="new vicon_body_vy")
    if "vicon_vy_original" in g.columns:
        axes[1].plot(t_rel, g["vicon_vy_original"], label="original CSV vicon_vy")
    if "odom_vy" in g.columns:
        axes[1].plot(t_rel, g["odom_vy"], label="odom_vy")
    axes[1].set_ylabel("vy")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_rel, g["vicon_body_wz"], label="new vicon_body_wz")
    if "vicon_wz_original" in g.columns:
        axes[2].plot(t_rel, g["vicon_wz_original"], label="original CSV vicon_wz")
    if "odom_wz" in g.columns:
        axes[2].plot(t_rel, g["odom_wz"], label="odom_wz")
    axes[2].set_ylabel("wz")
    axes[2].set_xlabel("Time (sec)")
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(f"{run_name} body-frame Vicon velocity check", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="processed_csv_all/all_runs_aligned_50hz.csv")
    parser.add_argument("--out", default="processed_csv_all/all_runs_aligned_50hz_body_vicon.csv")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--overwrite-vicon-velocity-cols", action="store_true")
    parser.add_argument("--plot-run", default="", help="Optional run_name to plot. Default: first run.")
    parser.add_argument("--plot-out", default="", help="Optional output path for diagnostic plot.")
    parser.add_argument("--metrics-out", default="", help="Optional output path for metrics JSON.")
    args = parser.parse_args()

    print("Loading:", args.csv)
    df = pd.read_csv(args.csv)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for c in ["vicon_vx", "vicon_vy", "vicon_wz"]:
        if c in df.columns:
            df[f"{c}_original"] = df[c]

    pieces = []
    for run_name, g in df.groupby("run_name", sort=False):
        pieces.append(compute_body_vicon_for_run(g, args.smooth_window))

    out_df = pd.concat(pieces, ignore_index=True)

    if args.overwrite_vicon_velocity_cols:
        print("Overwriting vicon_vx/vicon_vy/vicon_wz with body-frame Vicon velocities.")
        out_df["vicon_vx"] = out_df["vicon_body_vx"]
        out_df["vicon_vy"] = out_df["vicon_body_vy"]
        out_df["vicon_wz"] = out_df["vicon_body_wz"]
    else:
        print("Keeping original vicon_vx/vicon_vy/vicon_wz unchanged.")
        print("Use --overwrite-vicon-velocity-cols to make the output directly compatible with existing training scripts.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(args.out, index=False)

    metrics = summarize(out_df)

    metrics_out = args.metrics_out
    if not metrics_out:
        base, _ = os.path.splitext(args.out)
        metrics_out = base + "_metrics.json"

    metrics_dir = os.path.dirname(metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSummary metrics:")
    for group, vals in metrics.items():
        print(f"  {group}:")
        for k, v in vals.items():
            print(f"    {k}: {v:.6f}")

    run_names = sorted(out_df["run_name"].dropna().unique().tolist())
    plot_run = args.plot_run.strip() if args.plot_run.strip() else run_names[0]

    plot_out = args.plot_out
    if not plot_out:
        base, _ = os.path.splitext(args.out)
        safe = plot_run.replace("/", "_").replace(" ", "_")
        plot_out = f"{base}_check_{safe}.png"

    plot_dir = os.path.dirname(plot_out)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    plot_check(out_df, plot_run, plot_out)

    print("\nSaved:")
    print("  Output CSV:  ", args.out)
    print("  Metrics JSON:", metrics_out)
    print("  Check plot:  ", plot_out)
    print("\nNext training command example:")
    print(f"python 08_train_1d_cnn_vxy_pose_context.py --csv {args.out} --out-dir models/vxy_body_frame_test --include-runs slippery,water,wet --alpha 0.3 --clip-residual 0.5")


if __name__ == "__main__":
    main()
