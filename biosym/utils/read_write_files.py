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
    Reads a .mot file with a header ending in 'endheader' and returns a pandas DataFrame.
    """
    # First, find the line number where 'endheader' appears
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if line.strip().lower() == "endheader":
                header_end_line = i
                break
        else:
            raise ValueError("'endheader' not found in the file.")

    # The data starts after the 'endheader' line
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=header_end_line + 1)
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
