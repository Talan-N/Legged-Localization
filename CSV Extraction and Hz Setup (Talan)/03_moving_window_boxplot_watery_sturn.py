import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.makedirs("plots", exist_ok=True)

    df = pd.read_csv("processed_csv/watery_sturn_first_motion_aligned_50hz.csv")

    # 50 Hz data, 1-second window
    window_size = 50

    # step=5 means a new window every 0.1 seconds
    step = 5

    rows = []

    for end_idx in range(window_size - 1, len(df), step):
        start_idx = end_idx - window_size + 1
        w = df.iloc[start_idx:end_idx + 1]

        rows.append({
            "time_end_s": df["time_s"].iloc[end_idx],

            # residual score used to split low vs high residual
            "mean_abs_speed_residual": w["speed_residual_mag_abs"].mean(),

            # used to filter only moving windows
            "mean_vicon_speed_mag": w["vicon_speed_mag"].mean(),

            # proprioceptive features
            "mean_total_foot_force": w["total_foot_force"].mean(),
            "std_total_foot_force": w["total_foot_force"].std(),
            "mean_joint_velocity": w["joint_velocity_magnitude_sum_abs"].mean(),
            "mean_horizontal_accel_magnitude": w["horizontal_accel_magnitude"].mean(),
        })

    win = pd.DataFrame(rows)

    # Keep only moving windows
    moving_speed_threshold = 0.20
    moving = win[win["mean_vicon_speed_mag"] > moving_speed_threshold].copy()

    # Split moving windows into low and high residual groups
    low_thresh = moving["mean_abs_speed_residual"].quantile(0.25)
    high_thresh = moving["mean_abs_speed_residual"].quantile(0.75)

    low = moving[moving["mean_abs_speed_residual"] <= low_thresh].copy()
    high = moving[moving["mean_abs_speed_residual"] >= high_thresh].copy()

    # Save window summary
    moving.to_csv(
        "processed_csv/watery_sturn_first_motion_moving_windows.csv",
        index=False
    )

    features = [
        ("mean_total_foot_force", "Mean total foot force"),
        ("std_total_foot_force", "Std total foot force"),
        ("mean_joint_velocity", "Mean joint velocity Σ|dq|"),
        ("mean_horizontal_accel_magnitude", "Mean horizontal accel |a_xy|"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, features):
        ax.boxplot(
            [low[col].dropna(), high[col].dropna()],
          labels=["Low residual\nbottom 25%", "High residual\ntop 25%"],
            showmeans=True
        )

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Watery S-Turn: Low vs High Velocity-Magnitude Residual Windows\n"
        "(first-motion alignment only; moving windows only: mean Vicon speed > 0.20 m/s)",
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = "plots/watery_sturn_first_motion_moving_boxplots.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved plot: {output_path}")
    print("Saved summary: processed_csv/watery_sturn_first_motion_moving_windows.csv")
    print(f"Total windows: {len(win)}")
    print(f"Moving windows kept: {len(moving)}")
    print(f"Low residual threshold: <= {low_thresh:.4f} m/s")
    print(f"High residual threshold: >= {high_thresh:.4f} m/s")
    print(f"Low windows: {len(low)}")
    print(f"High windows: {len(high)}")


if __name__ == "__main__":
    main()
