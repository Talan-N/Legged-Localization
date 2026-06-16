#!/usr/bin/env python3
"""
Plot every run from a predictions CSV using the same plot style as
08_train_1d_cnn_vxy_pose_context.py.

Input:
  train_predictions.csv or test_predictions.csv from the 08 script.

Outputs per run:
  plot_<run>.png
  velocity_diagnostics_<run>.png
  residual_diagnostics_<run>.png

Example:

python 12_plot_all_runs_like_08.py \
  --pred models/vxy_body_frame_all_alpha06/test_predictions.csv \
  --out-dir models/vxy_body_frame_all_alpha06/all_run_plots

For your uploaded train_predictions.csv:

python 12_plot_all_runs_like_08.py \
  --pred train_predictions.csv \
  --out-dir train_plots
"""

import argparse
import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


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

    # Position plot
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

    # X over time
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

    # Y over time
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


def make_index_html(run_names, out_dir):
    html_path = os.path.join(out_dir, "index.html")
    rows = []
    for run_name in run_names:
        s = safe_name(run_name)
        rows.append(f"""
        <h2>{run_name}</h2>
        <p>
          <a href="plot_{s}.png">Position plot</a> |
          <a href="velocity_diagnostics_{s}.png">Velocity diagnostics</a> |
          <a href="residual_diagnostics_{s}.png">Residual diagnostics</a>
        </p>
        <img src="plot_{s}.png" style="width: 100%; max-width: 1200px;">
        <hr>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Run plots</title>
</head>
<body>
  <h1>Run plots</h1>
  {''.join(rows)}
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    return html_path


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prediction CSV: {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="train_predictions.csv or test_predictions.csv from the 08 script")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--runs", default="", help="Optional comma-separated run_name list. Default plots all runs.")
    parser.add_argument("--position-only", action="store_true", help="Only save the 08-style position/X/Y plot.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading predictions:", args.pred)
    df = pd.read_csv(args.pred)

    require_columns(df, [
        "run_name", "time_s",
        "vicon_x", "vicon_y",
        "odom_x", "odom_y",
        "corrected_x", "corrected_y",
    ])

    all_runs = sorted(df["run_name"].dropna().unique().tolist())

    if args.runs.strip():
        requested = [r.strip() for r in args.runs.split(",") if r.strip()]
        missing = [r for r in requested if r not in all_runs]
        if missing:
            raise ValueError(f"Requested runs not found: {missing}\nAvailable runs: {all_runs}")
        run_names = requested
    else:
        run_names = all_runs

    if not args.position_only:
        require_columns(df, [
            "true_vx", "true_vy", "true_wz",
            "odom_vx", "odom_vy", "odom_wz",
            "corrected_vx", "corrected_vy", "corrected_wz",
            "true_residual_vx", "true_residual_vy",
            "pred_residual_vx_raw", "pred_residual_vy_raw",
            "applied_residual_vx", "applied_residual_vy",
        ])

    print(f"Found {len(all_runs)} runs.")
    print(f"Plotting {len(run_names)} runs into: {args.out_dir}")

    for i, run_name in enumerate(run_names, start=1):
        s = safe_name(run_name)
        print(f"[{i}/{len(run_names)}] {run_name}")

        position_path = os.path.join(args.out_dir, f"plot_{s}.png")
        velocity_path = os.path.join(args.out_dir, f"velocity_diagnostics_{s}.png")
        residual_path = os.path.join(args.out_dir, f"residual_diagnostics_{s}.png")

        plot_run(df, run_name, position_path)

        if not args.position_only:
            plot_velocity_diagnostics(df, run_name, velocity_path)
            plot_residual_diagnostics(df, run_name, residual_path)

    html_path = make_index_html(run_names, args.out_dir)

    print("\nSaved plots to:", args.out_dir)
    print("Index HTML:    ", html_path)
    print("\nOpen with:")
    print(f"xdg-open {html_path}")


if __name__ == "__main__":
    main()
