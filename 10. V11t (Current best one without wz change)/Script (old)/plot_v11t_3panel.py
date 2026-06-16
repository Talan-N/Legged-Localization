#!/usr/bin/env python3
"""
Create the exact 3-panel plots like:
  left: XY trajectory
  top-right: X over time
  bottom-right: Y over time

Usage:
python plot_v11t_3panel.py --out-dir models/v11td_alpha048

Optional:
python plot_v11t_3panel.py --out-dir models/v11td_alpha048 --split test
python plot_v11t_3panel.py --out-dir models/v11td_alpha048 --run misc_concrete_slippery_preslip_rectangle
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def safe_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def plot_3panel(pred_df: pd.DataFrame, run_name: str, out_path: Path):
    g = pred_df[pred_df["run_name"] == run_name].copy().sort_values("time_s")
    if len(g) == 0:
        print(f"Skipping {run_name}: no rows found")
        return

    t = g["time_s"].to_numpy()
    t = t - t[0]

    terrain = g["terrain_name"].iloc[0] if "terrain_name" in g.columns else ""
    maneuver = g["maneuver_name"].iloc[0] if "maneuver_name" in g.columns else ""

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)

    ax_pos = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])

    # XY trajectory
    ax_pos.plot(g["vicon_x"], g["vicon_y"], label="Ground Truth")
    ax_pos.plot(g["odom_x"], g["odom_y"], label="Go2 Odometry")
    ax_pos.plot(g["corrected_x"], g["corrected_y"], label="Corrected CNN")
    ax_pos.set_title("Position", fontsize=30)
    ax_pos.set_xlabel("X (meters)", fontsize=22)
    ax_pos.set_ylabel("Y (meters)", fontsize=22)
    ax_pos.grid(True)
    ax_pos.axis("equal")
    ax_pos.legend(fontsize=10)

    # X over time
    ax_x.plot(t, g["vicon_x"], label="Ground Truth")
    ax_x.plot(t, g["odom_x"], label="Go2 Odometry")
    ax_x.plot(t, g["corrected_x"], label="Corrected CNN")
    ax_x.set_title("X over time", fontsize=30)
    ax_x.set_xlabel("Time (sec)", fontsize=22)
    ax_x.set_ylabel("X (meters)", fontsize=20)
    ax_x.grid(True)
    ax_x.legend(fontsize=10)

    # Y over time
    ax_y.plot(t, g["vicon_y"], label="Ground Truth")
    ax_y.plot(t, g["odom_y"], label="Go2 Odometry")
    ax_y.plot(t, g["corrected_y"], label="Corrected CNN")
    ax_y.set_title("Y over time", fontsize=30)
    ax_y.set_xlabel("Time (sec)", fontsize=22)
    ax_y.set_ylabel("Y (meters)", fontsize=20)
    ax_y.grid(True)
    ax_y.legend(fontsize=10)

    title = f"{run_name} Estimate"
    if terrain or maneuver:
        title += f" ({terrain}, {maneuver})"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Model output dir, e.g. models/v11td_alpha048")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--run", default="", help="Optional single run name")
    parser.add_argument("--plots-subdir", default="three_panel_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_path = out_dir / f"{args.split}_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Could not find {pred_path}")

    pred_df = pd.read_csv(pred_path)
    plots_dir = out_dir / args.plots_subdir

    if args.run:
        run_names = [args.run]
    else:
        run_names = sorted(pred_df["run_name"].dropna().unique())

    for run_name in run_names:
        out_path = plots_dir / f"plot_{safe_name(run_name)}.png"
        plot_3panel(pred_df, run_name, out_path)


if __name__ == "__main__":
    main()
