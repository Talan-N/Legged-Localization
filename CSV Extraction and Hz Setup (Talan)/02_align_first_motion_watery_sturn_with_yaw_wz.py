import os
import numpy as np
import pandas as pd
import torch


def relative_time(df):
    df = df.copy()
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    return df


def smooth(x, window=9):
    return pd.Series(x).rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean().to_numpy()


def interp_to_grid(df, t_grid, cols):
    out = {}
    t = df["time_s"].to_numpy()
    for col in cols:
        out[col] = np.interp(t_grid, t, df[col].to_numpy())
    return pd.DataFrame(out)


def interp_angle_to_grid(df, t_grid, col):
    """
    Interpolate an angle column safely by unwrapping first.
    This prevents jumps at -pi/pi from creating fake high angular velocity.
    """
    t = df["time_s"].to_numpy()
    angle = np.unwrap(df[col].to_numpy())
    return np.interp(t_grid, t, angle)


def compute_velocity_from_uniform_position(x, y, dt):
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vx = smooth(vx, 9)
    vy = smooth(vy, 9)
    return vx, vy


def compute_wz_from_uniform_yaw(yaw, dt):
    """
    Compute yaw rate from yaw angle.
    yaw should already be unwrapped before this function is called.
    """
    wz = np.gradient(yaw, dt)
    wz = smooth(wz, 9)
    return wz


def find_motion_start(t, speed, threshold=0.1, hold_time=0.2):
    dt = np.median(np.diff(t))
    hold_samples = max(1, int(hold_time / dt))
    above = speed > threshold
    for i in range(len(above) - hold_samples):
        if np.all(above[i:i + hold_samples]):
            return t[i]
    return t[0]


def main():
    os.makedirs("processed_csv", exist_ok=True)
    os.makedirs("tensors", exist_ok=True)

    lowstate = pd.read_csv("raw_csv/watery_sturn_lowstate.csv")
    odom = pd.read_csv("raw_csv/watery_sturn_go2_odom.csv")
    vicon = pd.read_csv("raw_csv/watery_sturn_vicon.csv")

    lowstate = relative_time(lowstate)
    odom = relative_time(odom)
    vicon = relative_time(vicon)

    dt = 0.02  # 50 Hz

    odom_t_grid = np.arange(odom["time_s"].iloc[0], odom["time_s"].iloc[-1], dt)
    vicon_t_grid = np.arange(vicon["time_s"].iloc[0], vicon["time_s"].iloc[-1], dt)

    # Used only to find first-motion timing offset
    odom_speed = smooth(
        np.interp(
            odom_t_grid,
            odom["time_s"].to_numpy(),
            np.sqrt(odom["vx"]**2 + odom["vy"]**2)
        ),
        9
    )

    vicon_x_for_start = np.interp(vicon_t_grid, vicon["time_s"].to_numpy(), vicon["x"].to_numpy())
    vicon_y_for_start = np.interp(vicon_t_grid, vicon["time_s"].to_numpy(), vicon["y"].to_numpy())

    vicon_vx_for_start, vicon_vy_for_start = compute_velocity_from_uniform_position(
        vicon_x_for_start,
        vicon_y_for_start,
        dt
    )
    vicon_speed = smooth(np.sqrt(vicon_vx_for_start**2 + vicon_vy_for_start**2), 9)

    offset = find_motion_start(odom_t_grid, odom_speed) - find_motion_start(vicon_t_grid, vicon_speed)
    vicon["time_s"] = vicon["time_s"] + offset

    start = max(lowstate["time_s"].iloc[0], odom["time_s"].iloc[0], vicon["time_s"].iloc[0])
    end = min(lowstate["time_s"].iloc[-1], odom["time_s"].iloc[-1], vicon["time_s"].iloc[-1])
    t_grid = np.arange(start, end, dt)

    # Interpolate lowstate normally. lowstate already has IMU roll/pitch/yaw.
    low_rs = interp_to_grid(lowstate, t_grid, [c for c in lowstate.columns if c != "time_s"])

    # Odometry already has yaw and wz from /odometry/filtered extraction.
    odom_rs = interp_to_grid(odom, t_grid, ["x", "y", "vx", "vy", "yaw", "wz"])
    odom_rs["yaw"] = interp_angle_to_grid(odom, t_grid, "yaw")

    # Vicon pose has x/y/yaw. We compute vx/vy from x/y and wz from yaw.
    vicon_x = interp_to_grid(vicon, t_grid, ["x"])["x"].to_numpy()
    vicon_y = interp_to_grid(vicon, t_grid, ["y"])["y"].to_numpy()
    vicon_yaw = interp_angle_to_grid(vicon, t_grid, "yaw")

    vicon_vx, vicon_vy = compute_velocity_from_uniform_position(vicon_x, vicon_y, dt)
    vicon_wz = compute_wz_from_uniform_yaw(vicon_yaw, dt)

    aligned = pd.DataFrame({"time_s": t_grid - t_grid[0]})

    for c in low_rs.columns:
        aligned[c] = low_rs[c]

    for c in odom_rs.columns:
        aligned[f"odom_{c}"] = odom_rs[c]

    aligned["vicon_x"] = vicon_x
    aligned["vicon_y"] = vicon_y
    aligned["vicon_yaw"] = vicon_yaw
    aligned["vicon_vx"] = vicon_vx
    aligned["vicon_vy"] = vicon_vy
    aligned["vicon_wz"] = vicon_wz

    # Helpful magnitude/error columns for analysis and plots
    aligned["odom_speed_mag"] = np.sqrt(aligned["odom_vx"]**2 + aligned["odom_vy"]**2)
    aligned["vicon_speed_mag"] = np.sqrt(aligned["vicon_vx"]**2 + aligned["vicon_vy"]**2)
    aligned["speed_residual_mag_abs"] = np.abs(aligned["vicon_speed_mag"] - aligned["odom_speed_mag"])
    aligned["total_foot_force"] = sum(aligned[f"foot_force_{i}"] for i in range(4))
    aligned["joint_velocity_magnitude_sum_abs"] = sum(np.abs(aligned[f"dq_{i}"]) for i in range(12))
    aligned["horizontal_accel_magnitude"] = np.sqrt(aligned["accel_x"]**2 + aligned["accel_y"]**2)

    # Export tensors for residual correction: [vx, vy, wz]
    exclude = [
        "time_s",
        "vicon_x", "vicon_y", "vicon_yaw", "vicon_vx", "vicon_vy", "vicon_wz",
        "odom_x", "odom_y", "odom_yaw", "odom_vx", "odom_vy", "odom_wz",
    ]
    feature_cols = [c for c in aligned.columns if c not in exclude]

    input_data = torch.tensor(aligned[feature_cols].values, dtype=torch.float32)

    vicon_targets = aligned[["vicon_vx", "vicon_vy", "vicon_wz"]].values
    odom_baseline = aligned[["odom_vx", "odom_vy", "odom_wz"]].values
    targets = torch.tensor(vicon_targets - odom_baseline, dtype=torch.float32)

    torch.save(input_data, "tensors/features.pt")
    torch.save(targets, "tensors/targets.pt")

    output_path = "processed_csv/watery_sturn_first_motion_aligned_50hz.csv"
    aligned.to_csv(output_path, index=False)

    print(f"Saved aligned CSV with yaw and wz: {output_path}")
    print("Exported training tensors to /tensors directory.")
    print("New useful columns include:")
    print("  odom_yaw, odom_wz")
    print("  vicon_yaw, vicon_wz")
    print("  residual targets can now be vx, vy, wz")


if __name__ == "__main__":
    main()
