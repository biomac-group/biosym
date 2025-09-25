from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import re
import warnings

from biosym.utils import read_mot, read_trc


def read_tracking_objective_files(
    yaml_path: str | Path = "tests/collocation/walking2d.yaml",
):
    """
    Read IK, GRF and TRC files referenced in the collocation YAML.
    Returns (ik_df|None, grf_df|None, trc_df|None). Missing entries return None.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Collocation YAML not found: {p}")

    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    coll = cfg.get("collocation", cfg)
    objectives = coll.get("objectives", [])

    ik_path = None
    grf_path = None
    trc_path = None
    for obj in objectives:
        name = obj.get("name", "")
        args = obj.get("args", {}) or {}
        fileval = args.get("file", None)
        if not fileval:
            continue
        path = Path(fileval)
        if name == "track_angles":
            ik_path = path
        elif name == "track_grf":
            grf_path = path
        elif name == "track_markers":
            trc_path = path

    ik_df = read_mot(str(ik_path)) if (ik_path and ik_path.exists()) else None
    grf_df = read_mot(str(grf_path)) if (grf_path and grf_path.exists()) else None
    trc_df = read_trc(str(trc_path)) if (trc_path and trc_path.exists()) else None
    return ik_df, grf_df, trc_df

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


def detect_heel_strikes_from_grf(
    grf_df: pd.DataFrame,
    channel: str = "ground_force_vy",
    force_increase: float = 500.0,
    time_points: int = 20,
):
    """Return heel strike indices detected on grf_df[channel]."""
    if channel not in grf_df.columns:
        raise KeyError(f"GRF channel '{channel}' not found in GRF dataframe.")
    vgrf = grf_df[channel].values
    return find_heel_strikes(
        vgrf, force_increase=force_increase, time_points=time_points
    )


# %% Interpolate and average gait cycles.
# save averaged joint angles to from dataframes to csv


def create_averaged_gait_joint_angles(
    ik_data: pd.DataFrame,
    heel_strike_index,
    n_points: int = 100,
    interpolate_missing: bool = True,
) -> pd.DataFrame:
    """
    Extract gait cycles from inverse kinematics data, (optionally) fill NaNs, interpolate
    each cycle to a fixed length and return per-sample mean/variance for every joint.

    Parameters
    ----------
    ik_data : pd.DataFrame
        Raw inverse kinematics dataframe (columns are joint angle signals).
    heel_strike_index : array-like
        Sample indices of heel strikes delimiting cycles.
    n_points : int, default 100
        Number of samples in the normalized output gait cycle.
    interpolate_missing : bool, default True
        If True, linearly interpolate NaNs in each joint angle column (extending to
        both ends). Entirely NaN columns are zero-filled with a warning.

    Returns
    -------
    pd.DataFrame
        Columns '<joint>_mean' and '<joint>_var' (length n_points) in a canonical order
        subset matching available joints.
    """

    joint_angles = [
        col
        for col in ik_data.columns
        if any(
            s in col
            for s in (
                "angle_l",
                "flexion_l",
                "angle_r",
                "flexion_r",
                "tilt",
                "tx",
                "ty",
            )
        )
    ]

    # Optionally interpolate NaNs in joint angle columns before cycle extraction
    if interpolate_missing:
        for col in joint_angles:
            if ik_data[col].isna().any():
                filled = ik_data[col].interpolate(method="linear", limit_direction="both")
                if filled.isna().any():
                    warnings.warn(
                        f"Joint angle column '{col}' all-NaN; filling with zeros.",
                        RuntimeWarning,
                    )
                    filled = filled.fillna(0.0)
                ik_data[col] = filled

    # Extracts individual gait cycles from raw joint angle data
    joint_angle_cycles = {col: [] for col in joint_angles}
    for col in joint_angles:
        data = ik_data[col].values
        for i in range(len(heel_strike_index) - 1):
            joint_angle_cycles[col].append(
                data[heel_strike_index[i] : heel_strike_index[i + 1]]
            )

    # Interpolates and averages each gait cycle
    n_points = n_points  # Number of samples per normalized gait cycle
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

    data = pd.DataFrame(data)

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

    valid_order = [col for col in column_order if col in data.columns]
    df = data[valid_order]

    return df


# ...existing code...
def create_averaged_gait_forces(
    grf_data: pd.DataFrame,
    heel_strike_index,
    channels=None,
    n_points: int = 100,
    interpolate_missing: bool = True,
) -> pd.DataFrame:
    """
    Create averaged gait-cycle mean/variance for selected GRF channels.

    Parameters
    ----------
    grf_data : pd.DataFrame
        Raw ground reaction force dataframe.
    heel_strike_index : array-like
        Sample indices of heel strikes delimiting cycles.
    channels : list[str] | None, default None
        Channels to include; if None a canonical subset is used (present columns only).
    n_points : int, default 100
        Samples per normalized gait cycle.
    interpolate_missing : bool, default True
        If True, linearly interpolate NaNs in each channel (extending ends). Entirely
        NaN channels are zero-filled with a warning.

    Returns
    -------
    pd.DataFrame
        Columns '<channel>_mean' and '<channel>_var' ordered canonically.
    """
    # Forces order for output columns. First are left foot, then right foot.
    desired_order = [
        "1_ground_force_vx",
        "1_ground_force_vy",
        "1_ground_force_vz",
        "ground_force_vx",
        "ground_force_vy",
        "ground_force_vz",
    ]

    # If user didn't pass channels, use desired_order but keep only existing columns
    if channels is None:
        channels = [c for c in desired_order if c in grf_data.columns]
    else:
        # keep only columns that exist
        channels = [c for c in channels if c in grf_data.columns]
        # preserve canonical ordering
        channels = [c for c in desired_order if c in channels]

    # Optionally interpolate NaNs in force channels before cycle extraction
    if interpolate_missing:
        for ch in channels:
            if grf_data[ch].isna().any():
                filled = grf_data[ch].interpolate(method="linear", limit_direction="both")
                if filled.isna().any():
                    warnings.warn(
                        f"GRF channel '{ch}' all-NaN; filling with zeros.",
                        RuntimeWarning,
                    )
                    filled = filled.fillna(0.0)
                grf_data[ch] = filled

    # Extract cycles
    force_cycles = {ch: [] for ch in channels}
    for ch in channels:
        data = grf_data[ch].values
        for i in range(len(heel_strike_index) - 1):
            start = heel_strike_index[i]
            end = heel_strike_index[i + 1]
            if end - start <= 1:
                continue
            force_cycles[ch].append(data[start:end])

    # Interpolate and compute mean/var
    mean_cycles = {}
    var_cycles = {}
    x_new = np.linspace(0, 1, n_points)
    for ch in channels:
        cycles = force_cycles[ch]
        if len(cycles) == 0:
            mean_cycles[ch] = np.zeros(n_points)
            var_cycles[ch] = np.zeros(n_points)
            continue
        interpolated = []
        for cycle in cycles:
            x_old = np.linspace(0, 1, len(cycle))
            interp_cycle = np.interp(x_new, x_old, cycle)
            interpolated.append(interp_cycle)
        interpolated = np.array(interpolated)
        mean_cycles[ch] = np.mean(interpolated, axis=0)
        var_cycles[ch] = np.var(interpolated, axis=0)

    # Build DataFrame with deterministic ordering: <channel>_mean, <channel>_var
    data = {}
    for ch in channels:
        data[f"{ch}_mean"] = mean_cycles[ch]
        data[f"{ch}_var"] = var_cycles[ch]

    df = pd.DataFrame(data)

    ordered_cols = []
    for ch in channels:
        mean_col = f"{ch}_mean"
        var_col = f"{ch}_var"
        if mean_col in df.columns:
            ordered_cols.append(mean_col)
        if var_col in df.columns:
            ordered_cols.append(var_col)
    if ordered_cols:
        df = df[ordered_cols]

    return df

# ...existing code...

def create_averaged_markers(
    trc_data: pd.DataFrame,
    heel_strike_index,
    n_points: int = 100,
    treadmill_speed: float | None = None,
    forward_axis: str = "X",
    time_column: str = "Time",
    interpolate_missing: bool = True,
) -> pd.DataFrame:
    """
    Average all markers' Cartesian coordinates over gait cycles.

    Detects markers via columns named '<Marker>_X', '<Marker>_Y', '<Marker>_Z'.
    Returns columns '<Marker>_<Axis>_mean' and '<Marker>_<Axis>_var' for each marker/axis.

    Treadmill adjustment (optional):
    If ``treadmill_speed`` (m/s) is provided, shift the forward-axis marker
    coordinate within each gait cycle by a linear ramp from 0 to
    ``treadmill_speed * cycle_duration``. This converts treadmill-relative
    trajectories into overground-like forward progression.

    Parameters
    ----------
    trc_data : pd.DataFrame
        TRC dataframe with a 'Time' column (seconds) and marker columns named
        '<Marker>_X', '<Marker>_Y', '<Marker>_Z'.
    heel_strike_index : array-like
        Sample indices of heel strikes that delimit gait cycles.
    n_points : int, default 100
        Number of samples per normalized gait cycle output.
    treadmill_speed : float | None, default None
        Belt speed in m/s. If provided, a forward translation of
        speed × (cycle duration) is added across the cycle.
    forward_axis : {"X","Y","Z"}, default "X"
        Axis considered forward for the translation.
    time_column : str, default "Time"
        Name of the time column (seconds). Required if treadmill_speed is set.
    interpolate_missing : bool, default True
        If True, linearly interpolate NaNs in each marker axis column before
        cycle segmentation. Leading/trailing NaNs are extended (both
        directions). Entirely NaN columns are filled with zeros and a warning.
    """
    axes = ("X", "Y", "Z")

    # Detect markers present as <name>_(X|Y|Z); preserve first-seen order
    marker_axes = {}
    marker_order = []
    for col in trc_data.columns:
        m = re.match(r"^(?P<name>.+)_(?P<ax>[XYZ])$", col)
        if not m:
            continue
        name = m.group("name")
        ax = m.group("ax")
        if name not in marker_axes:
            marker_axes[name] = set()
            marker_order.append(name)
        marker_axes[name].add(ax)
    markers = [m for m in marker_order if set(axes).issubset(marker_axes[m])]
    if not markers:
        raise KeyError("No markers with X/Y/Z columns found in TRC data.")

    # Optionally interpolate missing marker samples (occlusions) BEFORE segmentation
    if interpolate_missing:
        for name in markers:
            for ax in axes:
                col = f"{name}_{ax}"
                series = trc_data[col]
                if series.isna().any():
                    filled = series.interpolate(method="linear", limit_direction="both")
                    if filled.isna().any():
                        warnings.warn(
                            f"Column '{col}' contains only NaNs after interpolation; filling with zeros.",
                            RuntimeWarning,
                        )
                        filled = filled.fillna(0.0)
                    trc_data[col] = filled

    # Collect segments per marker/axis between heel strikes
    cycles = {name: {ax: [] for ax in axes} for name in markers}
    hs = np.asarray(heel_strike_index).astype(int)
    has_time = time_column in trc_data.columns
    for i in range(len(hs) - 1):
        start = hs[i]
        end = hs[i + 1]
        if end - start <= 1:
            continue
        # Determine cycle duration if treadmill translation is requested
        if treadmill_speed is not None:
            if not has_time:
                # Without explicit time, we cannot construct a metrically correct ramp.
                # Fall back to skipping the treadmill adjustment for this cycle.
                cycle_duration = None
            else:
                tseg = trc_data[time_column].values[start:end]
                cycle_duration = float(tseg[-1] - tseg[0]) if len(tseg) > 1 else 0.0
        for name in markers:
            for ax in axes:
                series = trc_data[f"{name}_{ax}"].values
                seg = series[start:end]
                # Apply treadmill forward translation as a linear ramp over this segment
                if treadmill_speed is not None and cycle_duration is not None and ax == forward_axis:
                    # Build a per-sample ramp from 0 to v*T over the segment length
                    if len(seg) > 1:
                        ramp = np.linspace(0.0, treadmill_speed * cycle_duration, len(seg))
                        seg = seg + ramp
                cycles[name][ax].append(seg)

    # Interpolate to n_points and compute mean/var
    x_new = np.linspace(0, 1, n_points)
    out = {}
    for name in markers:
        for ax in axes:
            segs = cycles[name][ax]
            if len(segs) == 0:
                out[f"{name}_{ax}_mean"] = np.zeros(n_points)
                out[f"{name}_{ax}_var"] = np.zeros(n_points)
                continue
            interp_segs = []
            for seg in segs:
                x_old = np.linspace(0, 1, len(seg))
                interp = np.interp(x_new, x_old, seg)
                # If treadmill translation requested but time was missing for this cycle, we cannot add a
                # metrically correct ramp post-hoc. We've already added ramps when cycles had time.
                interp_segs.append(interp)
            arr = np.vstack(interp_segs)
            out[f"{name}_{ax}_mean"] = np.mean(arr, axis=0)
            out[f"{name}_{ax}_var"] = np.var(arr, axis=0)

    # Deterministic column order: by marker order, then X,Y,Z, each mean then var
    ordered_cols = []
    for name in markers:
        for ax in axes:
            mean_c = f"{name}_{ax}_mean"
            var_c = f"{name}_{ax}_var"
            if mean_c in out:
                ordered_cols.append(mean_c)
            if var_c in out:
                ordered_cols.append(var_c)
    return pd.DataFrame({c: out[c] for c in ordered_cols})


def segment_gait_averages(
    n_points: int = 100,
    grf_channel: str = "ground_force_vy",
    treadmill_speed: float | None = None,
    forward_axis: str = "X",
    time_column: str = "Time",
    interpolate_missing: bool = True,
    interpolate_missing_angles: bool | None = None,
    interpolate_missing_forces: bool | None = None,
    interpolate_missing_markers: bool | None = None,
):
    """
    Compute averaged joint angles, GRFs and markers (mean/variance across cycles).

    Parameters
    ----------
    n_points : int, default 100
        Samples per normalized gait cycle for each averaged signal.
    grf_channel : str, default 'ground_force_vy'
        Vertical GRF channel used for heel strike detection.
    treadmill_speed : float | None
        If provided, forward translation (belt * time) added to marker forward axis.
    forward_axis : {"X","Y","Z"}, default "X"
        Axis considered forward when applying treadmill translation.
    time_column : str, default 'Time'
        Name of TRC time column.
    interpolate_missing : bool, default True
        Global default for filling NaNs in angles/forces/markers.
    interpolate_missing_angles / forces / markers : bool | None
        Override flags (if None, fall back to global default).

    Returns
    -------
    tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]
        (averaged_joint_angles, averaged_grfs, averaged_markers)
    """
    ik_df, grf_df, trc_df = read_tracking_objective_files(
        yaml_path="tests/collocation/walking2d.yaml"
    )


    # Heel strikes (only if GRF available)
    if grf_df is not None:
        hs_idx = detect_heel_strikes_from_grf(
            grf_df, channel=grf_channel, force_increase=500.0, time_points=20
        )
    else:
        hs_idx = np.array([0, len(trc_df)]) if trc_df is not None else np.array([0, 1])


    gait_avg_joint_angles = None
    gait_avg_grfs = None
    gait_avg_markers = None

    if interpolate_missing_angles is None:
        interpolate_missing_angles = interpolate_missing
    if interpolate_missing_forces is None:
        interpolate_missing_forces = interpolate_missing
    if interpolate_missing_markers is None:
        interpolate_missing_markers = interpolate_missing

    if ik_df is not None:
        gait_avg_joint_angles = create_averaged_gait_joint_angles(
            ik_df,
            hs_idx,
            n_points=n_points,
            interpolate_missing=interpolate_missing_angles,
        )

    if grf_df is not None:
        gait_avg_grfs = create_averaged_gait_forces(
            grf_df,
            hs_idx,
            n_points=n_points,
            interpolate_missing=interpolate_missing_forces,
        )

    if trc_df is not None:
        gait_avg_markers = create_averaged_markers(
            trc_df,
            hs_idx,
            n_points=n_points,
            treadmill_speed=treadmill_speed,
            forward_axis=forward_axis,
            time_column=time_column,
            interpolate_missing=interpolate_missing_markers,
        )

    return gait_avg_joint_angles, gait_avg_grfs, gait_avg_markers


# Plot averaged joint angles with variance
if __name__ == "__main__":
    
    ik_df, grf_df, trc_df = read_tracking_objective_files(
        yaml_path="tests/collocation/walking2d.yaml"
    )

    # find right certical grf forces
    right_vgrf = grf_df["ground_force_vy"].values

    right_hs_idx = find_heel_strikes(right_vgrf, force_increase=500, time_points=20)

    gait_avg_joint_angles, gait_avg_grfs, gait_avg_markers = segment_gait_averages(n_points=100, treadmill_speed=1.3)
    gait_avg_markers.to_csv("gait_averaged_markers.csv", index=False)


    # A gait cycle can be defined as the period between two consecutive# heel strikes for the same foot

    right_gait_cycles = []
    for i in range(len(right_hs_idx) - 1):
        right_gait_cycles.append(right_vgrf[right_hs_idx[i] : right_hs_idx[i + 1]])

    # compute duration of each gait cycle and average over them
    cycle_durations = [len(cycle) for cycle in right_gait_cycles]
    avg_cycle_duration = np.mean(cycle_durations) / 100

    # print how many gait cycles were detected
    print(len(right_gait_cycles), "right foot gait cycles detected.")

    # print average durationof gait cycles
    print(f"Average gait cycle duration: {avg_cycle_duration:.2f} (s)")

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

    # plot selected joint
    mean = gait_avg_joint_angles["knee_angle_l_mean"]
    std = np.sqrt(gait_avg_joint_angles["knee_angle_l_var"])
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


