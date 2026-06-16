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

def compute_velocity_from_uniform_position(x, y, dt):
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vx = smooth(vx, 9)
    vy = smooth(vy, 9)
    return vx, vy

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

    dt = 0.02
    odom_t_grid = np.arange(odom["time_s"].iloc[0], odom["time_s"].iloc[-1], dt)
    vicon_t_grid = np.arange(vicon["time_s"].iloc[0], vicon["time_s"].iloc[-1], dt)

    odom_speed = smooth(np.interp(odom_t_grid, odom["time_s"].to_numpy(), np.sqrt(odom["vx"]**2 + odom["vy"]**2)), 9)
    vicon_vx, vicon_vy = compute_velocity_from_uniform_position(
        np.interp(vicon_t_grid, vicon["time_s"].to_numpy(), vicon["x"].to_numpy()),
        np.interp(vicon_t_grid, vicon["time_s"].to_numpy(), vicon["y"].to_numpy()),
        dt
    )
    vicon_speed = smooth(np.sqrt(vicon_vx**2 + vicon_vy**2), 9)

    offset = find_motion_start(odom_t_grid, odom_speed) - find_motion_start(vicon_t_grid, vicon_speed)
    vicon["time_s"] = vicon["time_s"] + offset

    start = max(lowstate["time_s"].iloc[0], odom["time_s"].iloc[0], vicon["time_s"].iloc[0])
    end = min(lowstate["time_s"].iloc[-1], odom["time_s"].iloc[-1], vicon["time_s"].iloc[-1])
    t_grid = np.arange(start, end, dt)

    low_rs = interp_to_grid(lowstate, t_grid, [c for c in lowstate.columns if c != "time_s"])
    odom_rs = interp_to_grid(odom, t_grid, ["x", "y", "vx", "vy"])
    vicon_vx, vicon_vy = compute_velocity_from_uniform_position(
        interp_to_grid(vicon, t_grid, ["x"])["x"].to_numpy(),
        interp_to_grid(vicon, t_grid, ["y"])["y"].to_numpy(),
        dt
    )

    aligned = pd.DataFrame({"time_s": t_grid - t_grid[0]})
    for c in low_rs.columns: aligned[c] = low_rs[c]
    for c in odom_rs.columns: aligned[f"odom_{c}"] = odom_rs[c]
    aligned["vicon_vx"], aligned["vicon_vy"] = vicon_vx, vicon_vy

    # Exporting Tensors
    exclude = ["time_s", "vicon_x", "vicon_y", "vicon_vx", "vicon_vy", "odom_x", "odom_y", "odom_vx", "odom_vy"]
    feature_cols = [c for c in aligned.columns if c not in exclude]
    
    input_data = torch.tensor(aligned[feature_cols].values, dtype=torch.float32)
    targets = torch.tensor(aligned[["vicon_vx", "vicon_vy"]].values - aligned[["odom_vx", "odom_vy"]].values, dtype=torch.float32)
    
    torch.save(input_data, "tensors/features.pt")
    torch.save(targets, "tensors/targets.pt")

    aligned.to_csv("processed_csv/watery_sturn_first_motion_aligned_50hz.csv", index=False)
    print("Saved aligned CSV and exported training tensors to /tensors directory.")

if __name__ == "__main__":
    main()