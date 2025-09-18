import numpy as np
import pandas as pd


def read_trc(filepath):
    """
    Reads a .trc file and returns a pandas DataFrame with the data and a dict with metadata.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # The 3rd line (index 2) contains the column headers
    header_line_idx = 2

    # The 5th line (index 4) is where the data starts
    data_start_idx = 2

    # Read the data into a DataFrame
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=header_line_idx,
        skiprows=range(data_start_idx),
        engine="python",
    )

    # Optionally, extract metadata from the header lines
    metadata = {"path": filepath, "header_lines": lines[:data_start_idx]}

    return df, metadata


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
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=header_end_line + 1)

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
