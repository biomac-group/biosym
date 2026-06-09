import contextlib
import csv
import re

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp

def rotation_matrix_xyz(angles, force_2d=False):
    """Create rotation matrix from body-fixed XYZ Euler angles (in radians).
    
    Parameters
    ----------
    angles : array-like
        [rx, ry, rz] Euler angles in radians
    force_2d : bool, optional
        If True, only apply X rotation for planar motion. Default is False.
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    rx, ry, rz = angles
    
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    if force_2d:
        # For 2D planar model in XY plane, only use Z rotation
        return Rz
    
    # Body-fixed XYZ: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def rotate_xy(arr, *, angle_deg=None, angle_rad=None, x_col=0, y_col=1, copy=True):
    """Rotate an array in the XY-plane around the origin.

    Works with both NumPy ndarrays and JAX arrays. Rotation is applied to the
    columns (or last-dimension indices) specified by `x_col` and `y_col`.

    Parameters
    ----------
    arr : array-like
        Array with at least max(x_col, y_col)+1 entries on the last axis.
    angle_deg : float, optional
        Rotation angle in degrees (counter-clockwise).
    angle_rad : float, optional
        Rotation angle in radians (counter-clockwise). If provided, takes
        precedence over `angle_deg`.
    x_col, y_col : int
        Indices along the last axis corresponding to X and Y.
    copy : bool
        For NumPy inputs, whether to copy before modifying. Ignored for JAX
        (always returns a new array).

    Returns
    -------
    ndarray or jax.Array
        Rotated array of the same shape.
    """
    if arr is None:
        return None

    if angle_rad is None:
        if angle_deg is None:
            raise ValueError("Provide either angle_deg or angle_rad")
        angle_rad = np.deg2rad(angle_deg)

    is_jax_array = isinstance(arr, getattr(jax, "Array", ())) or (
        hasattr(arr, "__array_priority__") and type(arr).__module__.startswith("jax")
    )
    xp = jnp if is_jax_array else np

    a = xp.asarray(arr) if is_jax_array else (np.asarray(arr).copy() if copy else np.asarray(arr))

    c = xp.cos(angle_rad)
    s = xp.sin(angle_rad)

    x = a[..., x_col]
    y = a[..., y_col]
    x_new = x * c - y * s
    y_new = x * s + y * c

    if is_jax_array:
        a = a.at[..., x_col].set(x_new)
        a = a.at[..., y_col].set(y_new)
        return a

    a[..., x_col] = x_new
    a[..., y_col] = y_new
    return a


def resample_dataframe(df: pd.DataFrame, n_points: int, time_column: str | None = None) -> pd.DataFrame:
    """Resample all columns in a DataFrame to a fixed number of points.

    If time_column is provided and exists, it is used as the interpolation axis.
    Otherwise, a normalized [0, 1] axis is used.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    if len(df) == 0:
        raise ValueError("Cannot resample an empty DataFrame")

    if len(df) == n_points:
        return df.copy()

    if time_column is not None and time_column in df.columns:
        x_old = df[time_column].to_numpy()
        if len(x_old) > 1:
            x_new = np.linspace(float(x_old[0]), float(x_old[-1]), n_points)
        else:
            x_new = np.zeros(n_points)
    else:
        x_old = np.linspace(0.0, 1.0, len(df))
        x_new = np.linspace(0.0, 1.0, n_points)

    out = {}
    for col in df.columns:
        series = df[col].to_numpy()
        if len(series) == 1:
            out[col] = np.full(n_points, series[0])
        else:
            out[col] = np.interp(x_new, x_old, series)

    return pd.DataFrame(out)


def read_trc(filepath: str) -> pd.DataFrame:
    """
    Read a TRC file and return a DataFrame with columns:
    Frame, Time, and {Marker}_X, {Marker}_Y, {Marker}_Z for each marker.
    """
    # read all lines to detect header rows
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # find the header line with Frame and Time (the marker names row)
    hdr_idx = None
    for i, L in enumerate(lines):
        if re.search(r"\bFrame#?\b", L) and re.search(r"\bTime\b", L):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("TRC parse error: could not find header row with 'Frame' and 'Time'")

    # find the axis row (next line with X/Y/Z or X1/Y1/Z1 tokens)
    axis_idx = None
    axis_pat = re.compile(r"^[XYZxyz]\d*$")
    for j in range(hdr_idx + 1, min(hdr_idx + 6, len(lines))):
        toks = [t for t in re.split(r"[\t, ]+", lines[j].strip()) if t]
        if sum(1 for t in toks if axis_pat.match(t)) >= 3:
            axis_idx = j
            break
    if axis_idx is None:
        axis_idx = hdr_idx + 1

    # tokenize header and axis rows
    top = [t for t in re.split(r"[\t, ]+", lines[hdr_idx].strip()) if t]
    bot = [t for t in re.split(r"[\t, ]+", lines[axis_idx].strip()) if t]

    # Frame/Time + markers
    frame_name = "Frame" if top[0].lower().startswith("frame") else top[0]
    time_name = "Time"
    markers = top[2:]
    n_markers = len(markers)

    # axes (3 per marker) from axis row; fallback to X/Y/Z
    axes_triplets = []
    for i_m in range(n_markers):
        triple = bot[3 * i_m : 3 * i_m + 3]
        axes = []
        for k, default_ax in enumerate(("X", "Y", "Z")):
            try:
                ax = re.match(r"^([XYZxyz])", triple[k]).group(1).upper()
            except Exception:
                ax = default_ax
            axes.append(ax)
        axes_triplets.append(axes)

    # build final column names
    colnames = [frame_name, time_name]
    for m, axes in zip(markers, axes_triplets):
        for ax in axes:
            colnames.append(f"{m}_{ax}")

    # data starts after axis row (skip blank lines)
    data_start = axis_idx + 1
    while data_start < len(lines) and not re.search(r"\d", lines[data_start]):
        data_start += 1

    # read numeric data
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        engine="python",
        header=None,
        skiprows=data_start,
        comment="#",
        quoting=csv.QUOTE_NONE,
    )

    if df.shape[1] != len(colnames):
        raise ValueError(f"TRC parse error: found {df.shape[1]} columns, expected {len(colnames)}")

    df.columns = colnames

    # cast Frame/Time if present
    if "Frame" in df.columns:
        with pd.option_context("mode.chained_assignment", None), contextlib.suppress(Exception):
            df["Frame"] = df["Frame"].astype(int)
    if "Time" in df.columns:
        with pd.option_context("mode.chained_assignment", None), contextlib.suppress(Exception):
            df["Time"] = df["Time"].astype(float)

    return df


def read_mot(filepath: str) -> pd.DataFrame:
    """
    Reads a .mot file with a header ending in 'endheader'.
    Returns a DataFrame.
    If header contains 'inDegrees=yes' (case-insensitive), numeric data
    columns (except 'time' or 'frame') are converted from degrees to radians.
    """
    header = {}
    header_end_line = None

    # Read header and store key/value pairs (keys lower-cased)
    with open(filepath) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if s.lower() == "endheader":
                header_end_line = i
                break
            if "=" in s:
                k, v = s.split("=", 1)
                header[k.strip().lower()] = v.strip()
            else:
                # keep non key=value header lines if you want:
                # header.setdefault("_lines", []).append(s)
                pass

    if header_end_line is None:
        raise ValueError("'endheader' not found in the file.")

    # Read the data section
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=header_end_line + 1)

    # Check inDegrees flag (accept 'yes','true','1','y' as true)
    in_degrees_val = header.get("inDegrees")
    if in_degrees_val and in_degrees_val.strip().lower() in ("yes", "true", "1", "y"):
        # Convert numeric columns except typical time/frame columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_convert = [
            c for c in numeric_cols if c.lower() not in ("time", "frame")
        ]
        if cols_to_convert:
            df[cols_to_convert] = np.deg2rad(df[cols_to_convert])

    return df


def write_mot(df: pd.DataFrame, filepath: str, name: str = "Coordinates", in_degrees: bool = False) -> None:
    """
    Save a pandas DataFrame as a .mot file with OpenSim-compatible header.

    Args:
        df (pd.DataFrame): DataFrame containing time-series data.
        filepath (str or Path): Output file path with .mot extension.
        name (str): Name for the motion file (default: 'Coordinates').
        in_degrees (bool): Whether rotational values are in degrees.
    """
    n_rows, n_cols = df.shape

    with open(filepath, "w") as f:
        # OpenSim MOT header
        f.write(f"{name}\n")
        f.write("version=1\n")
        f.write(f"nRows={n_rows}\n")
        f.write(f"nColumns={n_cols}\n")
        f.write(f"inDegrees={'yes' if in_degrees else 'no'}\n")
        f.write("\n")
        f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
        f.write(
            "If the header above contains a line with 'inDegrees', this indicates "
            "whether rotational values are in degrees (yes) or radians (no).\n"
        )
        f.write("\n")
        f.write("endheader\n")

        # Column labels
        f.write("\t".join(df.columns) + "\n")

        # Data rows
        for _, row in df.iterrows():
            f.write("\t".join(f"{val:.6f}" for val in row.values) + "\n")

    print(f"Saved MOT file to: {filepath}")


def write_trc(
    df: pd.DataFrame,
    filepath: str,
    data_rate: int = 100,
    camera_rate: int = 100,
    units: str = "m",
) -> None:
    """
    Save a pandas DataFrame as a .trc file with OpenSim-compatible header.

    Args:
        df (pd.DataFrame): DataFrame with marker columns named '<Marker>_X', '<Marker>_Y', '<Marker>_Z'.
        filepath (str or Path): Output file path with .trc extension.
        data_rate (int): Data sampling rate in Hz (default: 100).
        camera_rate (int): Camera sampling rate in Hz (default: 100).
        units (str): Units for marker positions (default: 'm' for meters).
    """
    # Extract marker names from columns (assumes format: <Marker>_X, <Marker>_Y, <Marker>_Z)
    marker_names = []
    for col in df.columns:
        if col.endswith("_X"):
            marker_names.append(col[:-2])  # Remove '_X' suffix

    num_markers = len(marker_names)
    num_frames = len(df)

    with open(filepath, "w") as f:
        # TRC header line 1
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")

        # TRC header line 2
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")

        # TRC header line 3
        f.write(f"{data_rate}\t{camera_rate}\t{num_frames}\t{num_markers}\t{units}\t{data_rate}\t0\t{num_frames}\n")

        # Marker names row
        f.write("Frame#\tTime\t")
        f.write("\t".join([f"{name}\t\t" for name in marker_names]))
        f.write("\n")

        # Axis labels row
        f.write("\t\t")
        for _ in marker_names:
            f.write("X\tY\tZ\t")
        f.write("\n")

        # Data rows (Frame, Time, then X/Y/Z for each marker)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            # Frame number (1-indexed) and time
            time = (i - 1) / data_rate
            f.write(f"{i}\t{time:.2f}\t")

            # Marker coordinates
            coords = []
            for marker in marker_names:
                coords.append(f"{row[f'{marker}_X']:.6f}")
                coords.append(f"{row[f'{marker}_Y']:.6f}")
                coords.append(f"{row[f'{marker}_Z']:.6f}")
            f.write("\t".join(coords) + "\n")

    print(f"Saved TRC file to: {filepath}")
