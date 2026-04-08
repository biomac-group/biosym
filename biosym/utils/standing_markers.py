from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import pandas as pd
from biosym.utils.read_write_files import read_trc
import matplotlib.pyplot as plt


def create_standing_markers(
    trc_path: str | Path,
    tracked_markers: List[str] | None = None,
    n_samples: int = 10,
    interpolate_missing: bool = False,
    two_d: bool = True
) -> pd.DataFrame:
    """
    Compute the mean of the first ``n_samples`` for each marker position column.

    The returned DataFrame contains a single row. Columns are the original TRC
    marker coordinate names (``<Marker>_X``, ``<Marker>_Y``, ``<Marker>_Z``).

    Parameters
    ----------
    trc_path : str | Path
        Path to the TRC file. Parsed using ``biosym.utils.read_write_files.read_trc``.
    n_samples : int, default 100
        Number of initial samples to average. Clamped to the available rows.
    interpolate_missing : bool, default False
        If True, linearly interpolate NaNs in each marker axis column
        (extending to both ends) before averaging.
    two_d : bool, default False
        If True, set all Z-axis outputs to zero in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame of means with columns equal to the original
        TRC marker position column names ("<Marker>_X", "<Marker>_Y", "<Marker>_Z").
    """
    df = read_trc(str(trc_path))

    # Optionally interpolate NaNs prior to slicing
    # Select only marker position columns (suffix _X/_Y/_Z)
    pat = re.compile(r"^.+_[XYZ]$")
    marker_cols: List[str] = [c for c in df.columns if pat.match(c)]
    if not marker_cols:
        raise KeyError("No marker position columns (*_X/_Y/_Z) found in TRC data.")

    if interpolate_missing:
        for col in marker_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().any():
                s = s.interpolate(method="linear", limit_direction="both")
                df[col] = s


    def _expand(names: List[str]) -> List[str]:
        """Expand base marker names to X/Y/Z; keep full names if already axis-specific."""
        expanded = []
        seen = set()
        for name in names:
            if re.match(r".+_[XYZ]$", name):
                # Already coordinate-specific
                if name not in seen:
                    expanded.append(name)
                    seen.add(name)
            else:
                for axis in ("X", "Y", "Z"):
                    full = f"{name}_{axis}"
                    if full not in seen:
                        expanded.append(full)
                        seen.add(full)
        return expanded

    if tracked_markers is None:
        columns_to_iterate = marker_cols
    else:
        wanted = set(_expand(tracked_markers))
        # Preserve original TRC order
        columns_to_iterate = [c for c in marker_cols if c in wanted]
        missing = sorted(wanted.difference(columns_to_iterate))
        if missing:
            # Optional: could raise; for now just warn silently via print
            print(f"[create_standing_markers] Missing columns (skipped): {missing}")

    out = {}
    for col in columns_to_iterate:
        vals = pd.to_numeric(df[col].iloc[:n_samples], errors="coerce").to_numpy()
        out[col] = float(np.nanmean(vals))

    result = pd.DataFrame([out])
    if two_d:
        # Zero all Z columns in the output
        for col in list(result.columns):
            if col.endswith("_Z"):
                result[col] = 0.0
    return result


def plot_standing_markers_xy(standing_df: pd.DataFrame, tracked_markers: List[str] | None = None):
    """Scatter plot the X-Y locations of standing markers from a single-row DataFrame.

    Parameters
    ----------
    standing_df : pd.DataFrame
        Output of ``create_standing_markers`` (single row with marker columns).
    tracked_markers : list[str] | None, optional
        Optional subset of base marker names to plot. If provided, only
        markers whose base name is in this list are shown. If None, all
        markers in the dataframe are used.
    """
    if standing_df.shape[0] != 1:
        raise ValueError("standing_df must have exactly one row (output of create_standing_markers).")

    # Build mapping from base name to X/Y
    base_to_xy: Dict[str, tuple[float, float]] = {}
    for col in standing_df.columns:
        if col.endswith('_X'):
            base = col[:-2]
            x = float(standing_df[col].iloc[0])
            y_col = f"{base}_Y"
            if y_col in standing_df.columns:
                y = float(standing_df[y_col].iloc[0])
            else:
                y = 0.0
            base_to_xy[base] = (x, y)

    if tracked_markers is not None:
        base_to_xy = {k: v for k, v in base_to_xy.items() if k in tracked_markers}

    if not base_to_xy:
        raise ValueError("No markers available to plot after filtering.")

    xs = [v[0] for v in base_to_xy.values()]
    ys = [v[1] for v in base_to_xy.values()]
    labels = list(base_to_xy.keys())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, c='k')
    for x, y, lbl in zip(xs, ys, labels):
        ax.text(x, y, lbl, fontsize=8, ha='center', va='bottom')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Standing Marker Positions (X-Y)')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig, ax
