# %%
import re
import csv
import pandas as pd
import numpy as np
# ...existing code...

def read_trc(filepath: str) -> pd.DataFrame:
    """
    Read a TRC file and return a DataFrame with columns:
    Frame, Time, and {Marker}_X, {Marker}_Y, {Marker}_Z for each marker.
    """
    # read all lines to detect header rows
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
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
        with pd.option_context("mode.chained_assignment", None):
            try:
                df["Frame"] = df["Frame"].astype(int)
            except Exception:
                pass
    if "Time" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            try:
                df["Time"] = df["Time"].astype(float)
            except Exception:
                pass

    return df

# %%

def read_mot(filepath):
    """
    Reads a .mot file with a header ending in 'endheader'.
    Returns a tuple (df, header_dict).
    If header contains 'inDegrees=yes' (case-insensitive), numeric data
    columns (except 'time' or 'frame') are converted from degrees to radians.
    """
    header = {}
    header_end_line = None

    # Read header and store key/value pairs (keys lower-cased)
    with open(filepath, "r") as f:
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
    in_degrees_val = header.get("inDegrees", None)
    if in_degrees_val and in_degrees_val.strip().lower() in ("yes", "true", "1", "y"):
        # Convert numeric columns except typical time/frame columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_convert = [
            c for c in numeric_cols if c.lower() not in ("time", "frame")
        ]
        if cols_to_convert:
            df[cols_to_convert] = np.deg2rad(df[cols_to_convert])

    return df


def write_mot(df, filepath, name="Coordinates", in_degrees=False):
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
