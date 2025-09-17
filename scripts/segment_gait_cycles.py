# %% Import and load GRF data for Subject 1, trial 1

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biosym.utils import read_mot, write_mot

DATA_PATH = Path("~/.biosym/Data/Subject_01").expanduser()
ik_path = DATA_PATH / "IK"

# Ensure the data path exists and the requested files are present. Expanduser
# makes the path independent of the current working directory.
mot_path = DATA_PATH / "Subject1_trial1.mot"
if not mot_path.exists():
    raise FileNotFoundError(f"GRF file not found: {mot_path}")

# read_mot may accept a Path or a string; convert to str for compatibility
grf_trial1 = read_mot(str(mot_path))
right_vgrf = grf_trial1["ground_force_vy"]
left_vgrf = grf_trial1["1_ground_force_vy"]

plt.plot(
    grf_trial1["time"][:1000],
    left_vgrf[:1000],
    "o-",
    markersize=2,
    label="Vertical GRF",
)
plt.show()


# %% Segement gait cycles based on heel strike


def find_heel_strikes(vgrf, force_increase=300, time_points=10):
    """
    Detect heel strikes as upward crossings of local minimum
    Also checks if the force increases by at least force_increase within specified time points)
    right_vgrf: array of vertical GRF
    """
    hs_idx = []
    for i in range(1, len(vgrf) - time_points):
        # check if local minimum
        if vgrf[i - 1] > vgrf[i] < vgrf[i + 1]:
            # check if signal rises enough afterwards
            if (vgrf[i + time_points] - vgrf[i]) >= force_increase:
                hs_idx.append(i)
    return np.array(hs_idx)


start = time.time()
# choose contact threshold, Newtons as minimum force increase and time points to check
right_hs_idx = find_heel_strikes(right_vgrf, force_increase=500, time_points=20)
# left_hs_idx = find_heel_strikes(left_vgrf, force_increase=500, time_points=20)

end = time.time()
print(f"Function took {end - start:.6f} seconds")

# Plot first 1000 samples
plt.figure(figsize=(10, 5))
plt.plot(right_vgrf[:1000], label="right_vGRF")

for idx in right_hs_idx[right_hs_idx < 1000]:
    plt.axvline(
        x=idx,
        color="r",
        linestyle="--",
        alpha=0.7,
        label="Heel Strike" if idx == right_hs_idx[0] else "",
    )

plt.xlabel("Sample")
plt.ylabel("left_vGRF [N]")
plt.legend()
plt.title("Heel strike detection (first 1000 samples)")
plt.show()

# %% A gait cycle can be defined as the period between two consecutive
# heel strikes for the same foot

right_gait_cycles = []
for i in range(len(right_hs_idx) - 1):
    right_gait_cycles.append(right_vgrf[right_hs_idx[i] : right_hs_idx[i + 1]])

print(len(right_gait_cycles), "right foot gait cycles detected.")

# compute duration of each gait cycle and average over them
cycle_durations = [len(cycle) for cycle in right_gait_cycles]
avg_cycle_duration = np.mean(cycle_durations) / 100
print(f"Average gait cycle duration: {avg_cycle_duration:.2f} (s)")
# %% Interpolate and average gait cycles.
# save averaged joint angles to from dataframes to csv


def create_averaged_gait_joint_angles(ik_data, heel_strike_index, foot="right"):
    """
    Extracts gait cycles from inverse kinematics data, interpolates them, calculates mean and variance,
    and creates a pandas DataFrame for averaged joint angles over gait cycles.

    Args:
        ik_trial1 (pd.DataFrame): DataFrame containing raw inverse kinematics data.
        hs_idx (list): List of heel strike indices.
        joint_angles (list): A list of joint names.

    Returns:
        pd.DataFrame: A DataFrame with mean and variance columns for each joint.
    """

    if foot == "left":
        joint_angles = [
            col
            for col in ik_data.columns
            if any(s in col for s in ("angle_l", "flexion_l", "tilt", "tx", "ty"))
        ]
    elif foot == "right":
        joint_angles = [
            col
            for col in ik_data.columns
            if any(s in col for s in ("angle_r", "flexion_r", "tilt", "tx", "ty"))
        ]

    # Extracts individual gait cycles from raw joint angle data
    joint_angle_cycles = {col: [] for col in joint_angles}
    for col in joint_angles:
        data = ik_data[col].values
        for i in range(len(heel_strike_index) - 1):
            joint_angle_cycles[col].append(
                data[heel_strike_index[i] : heel_strike_index[i + 1]]
            )

    # Interpolates and averages each gait cycle
    n_points = 100  # Number of samples per normalized gait cycle
    mean_cycles = {}
    var_cycles = {}
    for col in joint_angles:
        cycles = joint_angle_cycles[col]
        interpolated = []
        for cycle in cycles:
            x_old = np.linspace(0, 1, len(cycle))
            x_new = np.linspace(0, 1, n_points)
            interpolated_cycle = np.interp(x_new, x_old, cycle)
            interpolated.append(interpolated_cycle)
        interpolated = np.array(interpolated)
        mean_cycles[col] = np.mean(interpolated, axis=0)
        var_cycles[col] = np.var(interpolated, axis=0)

    # Creates the DataFrame
    data = {}
    for col in joint_angles:
        data[f"{col}_mean"] = mean_cycles[col]
        data[f"{col}_var"] = var_cycles[col]

    return pd.DataFrame(data)


# Read IK data for trial 1
ik_trial1 = read_mot(ik_path / "Subject1_trial1_ik.mot")

# Create a DataFrame from the mean and variance data for the right joint angles
right_avg_joint_angles = create_averaged_gait_joint_angles(
    ik_trial1, right_hs_idx, foot="right"
)

left_avg_joint_angles = create_averaged_gait_joint_angles(
    ik_trial1, right_hs_idx, foot="left"
)

# Reorder columns to match model state vector
column_order = [
    "pelvis_tx_mean",
    "pelvis_tx_var",
    "pelvis_ty_mean",
    "pelvis_ty_var",
    "pelvis_tilt_mean",
    "pelvis_tilt_var",
    "hip_flexion_r_mean",
    "hip_flexion_r_var",
    "knee_angle_r_mean",
    "knee_angle_r_var",
    "ankle_angle_r_mean",
    "ankle_angle_r_var",
    "hip_flexion_l_mean",
    "hip_flexion_l_var",
    "knee_angle_l_mean",
    "knee_angle_l_var",
    "ankle_angle_l_mean",
    "ankle_angle_l_var",
]
# Filter only valid columns
gait_avg_joint_angles = right_avg_joint_angles.combine_first(left_avg_joint_angles)
valid_order = [col for col in column_order if col in gait_avg_joint_angles.columns]
gait_avg_joint_angles = gait_avg_joint_angles[valid_order]

# Save to MOT and CSV
gait_avg_joint_angles.to_csv(DATA_PATH / "gait_avg_joint_angles.csv", index=False)

write_mot(
    gait_avg_joint_angles,
    DATA_PATH / "gait_avg_joint_angles.mot",
    name="Coordinates",
    in_degrees=False,
)

# %% Plot averaged joint angles with variance
mean = left_avg_joint_angles["knee_angle_l_mean"]
std = np.sqrt(left_avg_joint_angles["knee_angle_l_var"])
x = np.linspace(0, 100, len(mean))

plt.figure(figsize=(10, 6))
plt.plot(x, mean, label="Mean")
plt.fill_between(x, mean - std, mean + std, alpha=0.3, label="Standard Deviation")
plt.title("Hip Flexion: Mean and Standard Deviation over Gait Cycle")
plt.xlabel("Gait Cycle (%)")
plt.ylabel("Hip Flexion (degrees)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
