"""
Reporting utilities for biosym optimization results.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from biosym.utils.segment_gait_cycles import (
    segment_gait_averages,
)


def _normalize_solution_inputs(solution_states, globals_dict):
    """Accept either (StatesDict, Globals) or a tuple (StatesDict, Globals)."""
    if isinstance(solution_states, tuple) and len(solution_states) == 2:
        states_dict, tuple_globals = solution_states
        # Prefer the tuple's globals unless an explicit override is provided.
        return states_dict, (globals_dict if globals_dict is not None else tuple_globals)
    return solution_states, globals_dict


def _resolve_from_yaml_path(path_value: str, yaml_path: str | None) -> str:
    """Resolve paths similarly to tracking objectives.

    Order:
    1) Absolute paths are returned as-is (after ~ expansion).
    2) If a YAML path is known and the YAML-relative resolution exists, use it.
    3) Otherwise, resolve relative paths against the repo's `example_data/` directory.

    This matches `biosym.utils.segment_gait_cycles.read_tracking_objective_files()`
    behavior, so `IK_file`/`grf_file` in reports accept the same relative paths
    as objectives.
    """
    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded):
        return expanded

    # Prefer YAML-relative paths *if* they exist.
    if yaml_path:
        yaml_dir = Path(os.path.expanduser(yaml_path)).resolve().parent
        candidate = (yaml_dir / expanded).resolve()
        if candidate.exists():
            return str(candidate)

    # Default: resolve relative to repo_root/example_data
    data_dir = (Path(__file__).resolve().parents[2] / "example_data").resolve()
    return str((data_dir / expanded).resolve())


def get_trial_files_from_settings(settings: dict | None) -> tuple[str | None, str | None, str | None]:
    """Return (ik_file, grf_file, trc_file) from YAML settings.

    - IK_file is the mot results with kinematics and ground reaction forces from addbiomechanics: from collocation.settings
    - grf_file, trc_file: from first objectives args that contain them
    """

    # IK file from collocation.settings
    ik_file = None
    collocation_settings = settings.get("settings")
    if isinstance(collocation_settings, dict):
        ik_file = collocation_settings.get("IK_file")

    # GRF and TRC files from objectives args
    grf_file = None
    trc_file = None
    objectives = settings.get("objectives")
    if isinstance(objectives, list):
        for obj in objectives:
            if not isinstance(obj, dict):
                continue
            args = obj.get("args") or {}
            if not isinstance(args, dict):
                continue
            if not grf_file:
                grf_file = args.get("grf_file")
            if not trc_file:
                trc_file = args.get("trc_file")

    return ik_file, grf_file, trc_file


def _select_ik_column(df_columns: list[str], candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _extract_sim_joint_angle(model, solution_states, joint_name: str) -> np.ndarray | None:
    solution_states, _ = _normalize_solution_inputs(solution_states, None)
    q_name = f"q_{joint_name}"
    try:
        q_local_idx = model.coordinates["names"].index(q_name)
    except ValueError:
        return None

    col = model.coordinates["idx"] + q_local_idx
    angles = np.asarray(solution_states.states.model[:, col]).astype(float)
    
    # Flip sign for knee joints
    if "knee" in joint_name.lower():
        angles = -angles
    
    return angles


def _get_sim_percent_axis(solution_states, globals_dict) -> np.ndarray:
    solution_states, globals_dict = _normalize_solution_inputs(solution_states, globals_dict)
    n = int(solution_states.states.model.shape[0])
    dur = None
    if globals_dict is not None and hasattr(globals_dict, "dur"):
        try:
            dur = float(globals_dict.dur)
        except Exception:
            dur = None
    if dur and dur > 0:
        t = np.linspace(0.0, dur, n)
        return 100.0 * t / dur
    return np.linspace(0.0, 100.0, n)


def _to_2d_float_array(arr) -> np.ndarray:
    """Convert array-like input to 2D float ndarray with shape (n_nodes, n_vars)."""
    out = np.asarray(arr, dtype=float)
    if out.ndim == 0:
        return out.reshape(1, 1)
    if out.ndim == 1:
        return out[:, np.newaxis]
    return out


def _get_model_mass_kg(model) -> float | None:
    """Return the model total mass in kg, if available."""
    masses = getattr(model, "masses", None)
    default_values = getattr(model, "default_values", None)
    if isinstance(masses, dict) and default_values is not None:
        try:
            idx = int(masses.get("idx", 0))
            n = int(masses.get("n", 0))
            if n > 0:
                mass_vals = np.asarray(default_values[idx : idx + n], dtype=float)
                total = float(np.sum(mass_vals))
                return total if total > 0.0 else None
        except Exception:
            pass

    bodies = getattr(model, "dicts", {}).get("bodies", [])
    if bodies:
        try:
            total = float(sum(float(b.get("mass", 0.0)) for b in bodies))
            return total if total > 0.0 else None
        except Exception:
            return None
    return None


def _collect_solved_state_variables(model, solution_states, globals_dict=None) -> tuple[dict[str, np.ndarray], list[str]]:
    """Collect solved variables from model, actuator model, and contact model.

    Returns
    -------
    tuple[dict[str, np.ndarray], list[str]]
        A dictionary of node-wise time series and a list of warnings.
    """
    warnings: list[str] = []
    series: dict[str, np.ndarray] = {}

    solution_states, globals_dict = _normalize_solution_inputs(solution_states, globals_dict)

    model_states = _to_2d_float_array(solution_states.states.model)
    n_nodes = int(model_states.shape[0])

    series["node"] = np.arange(n_nodes, dtype=float)
    series["gait_cycle_percent"] = _get_sim_percent_axis(solution_states, globals_dict).astype(float)

    # Core model state variables.
    model_state_names = list(getattr(model, "state_vector", []))
    n_model_cols = min(len(model_state_names), model_states.shape[1])
    for i in range(n_model_cols):
        series[str(model_state_names[i])] = model_states[:, i]

    # Explicit joint accelerations and moments (aliases for easier reporting/filtering).
    for joint_name in [j[2:] for j in getattr(model, "accs", {}).get("names", [])]:
        key = f"qdd_{joint_name}"
        if key in series:
            series[f"joint_acc_{joint_name}"] = np.asarray(series[key], dtype=float)

    for moment_name in getattr(model, "forces", {}).get("names", []):
        if moment_name in series:
            joint_name = str(moment_name).replace("M_", "", 1)
            series[f"joint_moment_{joint_name}"] = np.asarray(series[moment_name], dtype=float)

    # Contact model states.
    gc_states = getattr(solution_states.states, "gc_model", None)
    if gc_states is not None:
        gc_arr = _to_2d_float_array(gc_states)
        if gc_arr.size > 0 and gc_arr.shape[1] > 0:
            gc_names = []
            if hasattr(model, "gc_model"):
                if hasattr(model.gc_model, "state_vector"):
                    gc_names = list(getattr(model.gc_model, "state_vector", []))
                elif hasattr(model.gc_model, "get_states"):
                    try:
                        gc_names = list(model.gc_model.get_states())
                    except Exception:
                        gc_names = []
            if not gc_names or len(gc_names) != gc_arr.shape[1]:
                gc_names = [f"gc_state_{i}" for i in range(gc_arr.shape[1])]
            for i, name in enumerate(gc_names):
                key = str(name)
                if key in series:
                    key = f"gc_{key}"
                series[key] = gc_arr[:, i]

    # Actuator model states.
    actuator_obj = getattr(model, "actuator_model", None)
    actuator_states = getattr(solution_states.states, "actuator_model", None)
    act_arr = None
    if actuator_states is not None:
        act_arr = _to_2d_float_array(actuator_states)
        if act_arr.size > 0 and act_arr.shape[1] > 0:
            act_names = []
            if actuator_obj is not None and hasattr(actuator_obj, "state_vector"):
                act_names = list(getattr(actuator_obj, "state_vector", []))
            if not act_names or len(act_names) != act_arr.shape[1]:
                act_names = [f"actuator_state_{i}" for i in range(act_arr.shape[1])]
            for i, name in enumerate(act_names):
                key = str(name)
                if key in series:
                    key = f"act_{key}"
                series[key] = act_arr[:, i]

    # Muscle-specific derived variables from Hill2d muscle equations.
    if actuator_obj is not None and hasattr(actuator_obj, "muscle_equations"):
        try:
            f_ce, f_see, _f_pee = actuator_obj.muscle_equations(solution_states.states, solution_states.constants, model)
            f_ce = np.asarray(f_ce, dtype=float)
            f_see = np.asarray(f_see, dtype=float)

            if f_ce.ndim == 2:
                f_ce = f_ce.T
            if f_see.ndim == 2:
                f_see = f_see.T

            muscle_names = list(getattr(actuator_obj, "names", [f"muscle_{i}" for i in range(f_ce.shape[1])]))
            if len(muscle_names) != f_ce.shape[1]:
                muscle_names = [f"muscle_{i}" for i in range(f_ce.shape[1])]

            mass_kg = _get_model_mass_kg(model)
            force_scale = 1.0 / mass_kg if mass_kg and mass_kg > 0.0 else 1.0

            for i, muscle_name in enumerate(muscle_names):
                safe_name = str(muscle_name)
                series[f"muscle_force_{safe_name}"] = f_ce[:, i] * force_scale
                series[f"tendon_force_{safe_name}"] = f_see[:, i]
        except Exception as e:
            warnings.append(f"Failed to compute Hill muscle/tendon force series: {e}")

    # Muscle activations, if available.
    if actuator_obj is not None and act_arr is not None and hasattr(actuator_obj, "idx"):
        try:
            idx_a = actuator_obj.idx.get("a", None)
            if idx_a is not None:
                idx_a = np.asarray(idx_a, dtype=int).reshape(-1)
                if idx_a.size > 0 and np.max(idx_a) < act_arr.shape[1] and np.min(idx_a) >= 0:
                    act_names = list(getattr(actuator_obj, "names", [f"actuator_{i}" for i in range(idx_a.size)]))
                    if len(act_names) != idx_a.size:
                        act_names = [f"actuator_{i}" for i in range(idx_a.size)]
                    for i, idx in enumerate(idx_a):
                        series[f"activation_{act_names[i]}"] = act_arr[:, idx]
        except Exception as e:
            warnings.append(f"Failed to extract activation series: {e}")

    # Generic torque actuator states, if available.
    if actuator_obj is not None and act_arr is not None and hasattr(actuator_obj, "idx"):
        try:
            idx_tau = actuator_obj.idx.get("torque", None)
            if idx_tau is not None:
                idx_tau = np.asarray(idx_tau, dtype=int).reshape(-1)
                if idx_tau.size > 0 and np.max(idx_tau) < act_arr.shape[1] and np.min(idx_tau) >= 0:
                    torque_names = []
                    if hasattr(actuator_obj, "actuators") and isinstance(actuator_obj.actuators, dict):
                        torque_names = list(actuator_obj.actuators.keys())
                    if len(torque_names) != idx_tau.size:
                        torque_names = [f"actuator_{i}" for i in range(idx_tau.size)]
                    for i, idx in enumerate(idx_tau):
                        series[f"actuator_torque_{torque_names[i]}"] = act_arr[:, idx]
        except Exception as e:
            warnings.append(f"Failed to extract generic actuator torque series: {e}")

    return series, warnings


def _save_series_group_line_plots(
    report_path: str,
    series: dict[str, np.ndarray],
    keys: list[str],
    filename_prefix: str,
    y_label: str,
    title_prefix: str,
    output_subdir: str | None = None,
    trials_config: list[dict] | None = None,
) -> tuple[list[str], list[str]]:
    """Save one line plot per key in `keys` and return (paths, warnings)."""
    warnings: list[str] = []
    paths: list[str] = []
    if not keys:
        return paths, warnings

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()
    if output_subdir:
        out_dir = os.path.join(out_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    def _build_segments(n_nodes: int) -> list[dict]:
        segments: list[dict] = []
        if not trials_config:
            segments.append(
                {
                    "start_idx": 0,
                    "end_idx": n_nodes,
                    "x_vals": np.asarray(
                        series.get("gait_cycle_percent", np.linspace(0.0, 100.0, n_nodes)),
                        dtype=float,
                    ),
                    "separator_x": None,
                }
            )
            return segments

        current_idx = 0
        x_offset = 0.0
        for i, trial_cfg in enumerate(trials_config):
            n_frames = trial_cfg.get("n_frames")
            try:
                n_frames = int(n_frames)
            except Exception:
                warnings.append(f"Trial {i}: invalid n_frames='{n_frames}' for {title_prefix} plots; skipping.")
                continue

            if n_frames <= 0:
                warnings.append(f"Trial {i}: n_frames must be > 0 for {title_prefix} plots; skipping.")
                continue

            if current_idx >= n_nodes:
                break

            end_idx = min(current_idx + n_frames, n_nodes)
            seg_len = end_idx - current_idx
            if seg_len <= 0:
                continue

            segments.append(
                {
                    "start_idx": current_idx,
                    "end_idx": end_idx,
                    "x_vals": np.linspace(x_offset, x_offset + 100.0, seg_len),
                    "separator_x": x_offset + 100.0,
                }
            )
            current_idx = end_idx
            x_offset += 100.0

        if not segments:
            segments.append(
                {
                    "start_idx": 0,
                    "end_idx": n_nodes,
                    "x_vals": np.asarray(
                        series.get("gait_cycle_percent", np.linspace(0.0, 100.0, n_nodes)),
                        dtype=float,
                    ),
                    "separator_x": None,
                }
            )
            return segments

        if current_idx < n_nodes:
            seg_len = n_nodes - current_idx
            warnings.append(
                f"Trial frame counts for {title_prefix} plots do not cover all nodes; appending final segment."
            )
            segments.append(
                {
                    "start_idx": current_idx,
                    "end_idx": n_nodes,
                    "x_vals": np.linspace(x_offset, x_offset + 100.0, seg_len),
                    "separator_x": None,
                }
            )

        return segments

    def _sanitize(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)

    for key in keys:
        y = np.asarray(series.get(key, []), dtype=float)
        if y.size == 0:
            continue
        n_nodes = int(y.shape[0])
        segments = _build_segments(n_nodes)

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        has_data = False
        for seg in segments:
            s_idx = seg["start_idx"]
            e_idx = seg["end_idx"]
            x_vals = np.asarray(seg["x_vals"], dtype=float)
            y_vals = y[s_idx:e_idx]
            if y_vals.size == 0 or y_vals.size != x_vals.size:
                continue
            ax.plot(x_vals, y_vals, linewidth=2)
            has_data = True

            sep_x = seg.get("separator_x")
            if trials_config and sep_x is not None and e_idx < n_nodes:
                ax.axvline(x=sep_x, color="k", linestyle=":", alpha=0.3)

        if not has_data:
            plt.close(fig)
            warnings.append(f"Skipping '{key}' plot due to empty segment data.")
            continue

        if trials_config:
            ax.set_xlabel("Cumulative gait cycle (%)")
        else:
            ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix}: {key}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        stem = str(key)
        if stem.startswith(f"{filename_prefix}_"):
            stem = stem[len(filename_prefix) + 1 :]
        out_path = os.path.join(out_dir, f"{filename_prefix}_{_sanitize(stem)}.png")
        fig.savefig(out_path)
        plt.close(fig)
        paths.append(out_path)

    return paths, warnings


def _save_report_series_csv(
    report_path: str,
    series: dict[str, np.ndarray],
    filename: str = "report_variables.csv",
) -> tuple[str | None, str | None]:
    """Save all report series into a single node-wise CSV file."""
    if not series:
        return None, "No solved series available; skipping CSV export."

    lengths = {k: int(np.asarray(v).shape[0]) for k, v in series.items() if np.asarray(v).ndim >= 1}
    if not lengths:
        return None, "Series are empty; skipping CSV export."

    n_rows = max(lengths.values())
    data = {}
    for k, v in series.items():
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.shape[0] < n_rows:
            padded = np.full((n_rows,), np.nan, dtype=float)
            padded[: arr.shape[0]] = arr
            arr = padded
        elif arr.shape[0] > n_rows:
            arr = arr[:n_rows]
        data[k] = arr

    out_dir = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    pd.DataFrame(data).to_csv(out_path, index=False)
    return out_path, None


def _save_angle_plots(
    report_path: str,
    model,
    solution_states,
    globals_dict,
    settings: dict | None,
    yaml_path: str | None,
    joint_names: list[str],
    filename_prefix: str = "plot",
    trials_config: list[dict] | None = None,
) -> tuple[list[str], list[str]]:
    """Return (plot_paths, warnings)."""
    warnings: list[str] = []
    plot_paths: list[str] = []

    solution_states, globals_dict = _normalize_solution_inputs(solution_states, globals_dict)

    # Use a non-interactive backend
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()
    out_dir = os.path.join(out_dir, "angles")
    os.makedirs(out_dir, exist_ok=True)
    
    # Multi-trial plotting: prepare configuration for each trial
    # When solving multiple trials together, solution_states.states.model contains all frames
    # concatenated sequentially: [Trial 1 frames | Trial 2 frames | Trial 3 frames | ...]
    # We record start/end indices for each trial so we can slice the correct frames during plotting.
    prepared_trials = []
    
    if trials_config:
        # Multi-trial mode: iterate through each trial's configuration
        current_idx = 0
        x_offset_pct = 0.0
        
        for i, trial_cfg in enumerate(trials_config):
            ik_file = trial_cfg.get("ik_file")
            grf_file = trial_cfg.get("grf_file")
            n_frames = trial_cfg.get("n_frames")
            
            if not ik_file or not grf_file or n_frames is None:
                warnings.append(f"Trial {i}: Missing IK/GRF/frames info; skipping.")
                continue
                
            ik_path = _resolve_from_yaml_path(ik_file, yaml_path)
            grf_path = _resolve_from_yaml_path(grf_file, yaml_path)
            
            # Record indices to slice solution_states.states.model for this trial
            x_vals = np.linspace(x_offset_pct, x_offset_pct + 100.0, int(n_frames))
            
            prepared_trials.append({
                "ik_path": ik_path,
                "grf_path": grf_path,
                "n_frames": int(n_frames),
                "start_idx": current_idx,
                "end_idx": current_idx + int(n_frames),
                "x_vals": x_vals,
                "label_suffix": f" (Trial {i+1})",
                "x_offset": x_offset_pct
            })
            
            current_idx += int(n_frames)
            x_offset_pct += 100.0
            
    else:
        # Single trial mode
        ik_file, grf_file, _trc_file = get_trial_files_from_settings(settings)
        if not ik_file or not grf_file:
            warnings.append(
                "YAML/settings must specify both IK and GRF trial files to generate segmented angle plots. "
                "Provide them either under collocation.settings (e.g. 'IK_file' and 'grf_file') or under an objective's args as 'ik_file'/'grf_file'."
            )
            return plot_paths, warnings

        ik_path = _resolve_from_yaml_path(ik_file, yaml_path)
        grf_path = _resolve_from_yaml_path(grf_file, yaml_path)
        if not os.path.exists(ik_path):
             warnings.append(f"IK file not found at '{ik_path}'; skipping angle plots.")
             return plot_paths, warnings
        if not os.path.exists(grf_path):
             warnings.append(f"GRF file not found at '{grf_path}'; skipping angle plots.")
             return plot_paths, warnings
        
        n_points = int(solution_states.states.model.shape[0])
        pct_sim = _get_sim_percent_axis(solution_states, globals_dict)
        
        prepared_trials.append({
            "ik_path": ik_path,
            "grf_path": grf_path,
            "n_frames": n_points,
            "start_idx": 0,
            "end_idx": n_points,
            "x_vals": pct_sim,
            "label_suffix": "",
            "x_offset": 0.0
        })

    # Now iterate joints and plot concatenated validation
    
    for joint in joint_names:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        
        has_data = False
        
        # We will collect arrays to plot continuous lines if desired,
        # or plot separate segments if we want gaps. Concatenated usually implies continuous if possible,
        # but 0-100 resets mean discontinuous X or discontinuous data?
        # If we map x to 0..100..200, it's continuous.
        
        for trial in prepared_trials:
            start = trial["start_idx"]
            end = trial["end_idx"]
            
            # Extract Sim for this joint
            sim_full = _extract_sim_joint_angle(model, solution_states, joint)
                
            sim_segment = sim_full[start:end]
            
            # Get IK data
            ik_path = trial["ik_path"]
            grf_path = trial["grf_path"]
            n_frames = trial["n_frames"]
            
            if not os.path.exists(ik_path) or not os.path.exists(grf_path):
                 warnings.append(f"Missing files for plotting {joint}: {ik_path}")
                 continue

            try:
                gait_avg_joint_angles, _, _ = segment_gait_averages(
                    grf_file=grf_path,
                    ik_file=ik_path,
                    trc_file=None,
                    n_points=n_frames,
                )
            except Exception as e:
                warnings.append(f"Failed to segment/resample IK for {joint}: {e}")
                continue

            if gait_avg_joint_angles is None:
                continue

            # Identify column
            # Averaged gait-cycle joint-angle columns (from segment_gait_cycles.py)
            if joint == "hip_r": ik_col = "hip_flexion_r_mean"
            elif joint == "knee_r": ik_col = "knee_angle_r_mean"
            elif joint == "ankle_r": ik_col = "ankle_angle_r_mean"
            elif joint == "hip_l": ik_col = "hip_flexion_l_mean"
            elif joint == "knee_l": ik_col = "knee_angle_l_mean"
            elif joint == "ankle_l": ik_col = "ankle_angle_l_mean"
            elif joint == "pelvis_tx": ik_col = "pelvis_tx_mean"
            elif joint == "pelvis_ty": ik_col = "pelvis_ty_mean"
            elif joint == "pelvis_tilt": ik_col = "pelvis_tilt_mean"
            else: ik_col = f"{joint}_mean"

            if ik_col not in gait_avg_joint_angles.columns:
                 warnings.append(f"Column {ik_col} not found in {ik_path}")
                 continue

            ik_segment = np.asarray(gait_avg_joint_angles[ik_col]).astype(float)
            
            if ik_segment.shape[0] != sim_segment.shape[0]:
                 # Try to coerce? segment_gait_averages should handle n_points
                 warnings.append(f"Shape mismatch {joint}: IK {ik_segment.shape[0]} vs Sim {sim_segment.shape[0]}")
                 continue
                 
            # Plot specific trial segment
            # Use color cycle or fixed colors? Sim usually dashed/solid vs IK
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color = color_cycle[0] # IK
            color_sim = color_cycle[1] # Sim
            
            ax.plot(trial["x_vals"], ik_segment, color=color, alpha=0.6, label="IK" if not has_data else "")
            ax.plot(trial["x_vals"], sim_segment, color=color_sim, linestyle="--", label="Sim" if not has_data else "")
            
            # Add vertical separator if multiple trials and not last
            if trials_config and trial["end_idx"] < solution_states.states.model.shape[0]:
                ax.axvline(x=trial["x_offset"] + 100.0, color='k', linestyle=':', alpha=0.3)
            
            has_data = True

        if not has_data:
            plt.close(fig)
            continue
            
        ax.set_xlabel("Cumulative Gait Cycle (%)")
        ax.set_ylabel("Angle (rad)")
        ax.set_title(f"{joint}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(out_dir, f"{filename_prefix}_{joint}.png")
        plt.savefig(fig_path)
        plt.close(fig)
        plot_paths.append(fig_path)

    if plot_paths:
        if trials_config:
             warnings.insert(0, "Angle plots saved (Multi-trial concatenated).")
        else:
               warnings.insert(0, f"Angle plots saved in 'angles/' (segmented IK source: '{ik_path}', GRF source: '{grf_path}').")

    return plot_paths, warnings

    ik_path = _resolve_from_yaml_path(ik_file, yaml_path)
    grf_path = _resolve_from_yaml_path(grf_file, yaml_path)
    if not os.path.exists(ik_path):
        warnings.append(f"IK file not found at '{ik_path}'; skipping angle plots.")
        return plot_paths, warnings
    if not os.path.exists(grf_path):
        warnings.append(f"GRF file not found at '{grf_path}'; skipping angle plots.")
        return plot_paths, warnings

    n_points = int(solution_states.states.model.shape[0])
    try:
        gait_avg_joint_angles, _, _ = segment_gait_averages(
            grf_file=grf_path,
            ik_file=ik_path,
            trc_file=None,
            n_points=n_points,
        )
    except Exception as e:
        warnings.append(f"Failed to segment trial into averaged gait cycle: {e}")
        return plot_paths, warnings

    if gait_avg_joint_angles is None:
        warnings.append("Segmentation returned no averaged joint angles; skipping angle plots.")
        return plot_paths, warnings

    pct_sim = _get_sim_percent_axis(solution_states, globals_dict)
    out_dir = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()
    out_dir = os.path.join(out_dir, "grf")

    # Use a non-interactive backend
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for joint in joint_names:
        sim = _extract_sim_joint_angle(model, solution_states, joint)
        if sim is None:
            warnings.append(f"Joint '{joint}' not found in model coordinates; skipping plot.")
            continue

        # Averaged gait-cycle joint-angle columns (from segment_gait_cycles.py)
        if joint == "hip_r":
            ik_col = "hip_flexion_r_mean"
        elif joint == "knee_r":
            ik_col = "knee_angle_r_mean"
        elif joint == "ankle_r":
            ik_col = "ankle_angle_r_mean"
        elif joint == "hip_l":
            ik_col = "hip_flexion_l_mean"
        elif joint == "knee_l":
            ik_col = "knee_angle_l_mean"
        elif joint == "ankle_l":
            ik_col = "ankle_angle_l_mean"
        elif joint == "pelvis_tx":
            ik_col = "pelvis_tx_mean"
        elif joint == "pelvis_ty":
            ik_col = "pelvis_ty_mean"
        elif joint == "pelvis_tilt":
            ik_col = "pelvis_tilt_mean"
        else:
            ik_col = f"{joint}_mean"

        if ik_col not in gait_avg_joint_angles.columns:
            warnings.append(f"Averaged IK column '{ik_col}' not found for joint '{joint}'; skipping plot.")
            continue

        ik_on_sim = np.asarray(gait_avg_joint_angles[ik_col]).astype(float)
        if ik_on_sim.shape[0] != pct_sim.shape[0]:
            warnings.append(f"Averaged IK column '{ik_col}' length mismatch; skipping joint '{joint}'.")
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax.plot(pct_sim, ik_on_sim, label=f"IK avg ({os.path.basename(ik_path)}:{ik_col})")
        ax.plot(pct_sim, sim, label="Simulated")
        ax.set_title(f"{joint} angle")
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        out_path = os.path.join(out_dir, f"{joint}.png")
        try:
            fig.tight_layout()
            fig.savefig(out_path)
            plot_paths.append(out_path)
        finally:
            plt.close(fig)

    if plot_paths:
        warnings.insert(0, f"Angle plots saved next to report (segmented IK source: '{ik_path}', GRF source: '{grf_path}').")

    return plot_paths, warnings


def _save_grf_plots(
    report_path: str,
    model,
    solution_states,
    globals_dict,
    settings: dict | None,
    yaml_path: str | None,
    trials_config: list[dict] | None = None,
) -> tuple[list[str], list[str]]:
    """Save GRF comparison plots (Fx, Fy) as PNG.

    Plots simulated vs experimental GRFs for left/right feet.
    Simulated GRFs are taken from the state vector slice `model.ext_forces`.
    Experimental GRFs are segmented/averaged from the GRF trial file.
    """
    warnings: list[str] = []
    plot_paths: list[str] = []

    solution_states, globals_dict = _normalize_solution_inputs(solution_states, globals_dict)
    
    # 1. Prepare segments configuration
    prepared_segments = []
    
    if trials_config:
        current_idx = 0
        x_offset_pct = 0.0
        for i, trial_cfg in enumerate(trials_config):
            grf_file = trial_cfg.get("grf_file")
            n_frames = trial_cfg.get("n_frames")
            
            if not grf_file or n_frames is None:
                warnings.append(f"Trial {i}: Missing GRF file or n_frames; skipping GRF segment.")
                continue

            grf_path = _resolve_from_yaml_path(grf_file, yaml_path)
            if not os.path.exists(grf_path):
                warnings.append(f"GRF file not found at '{grf_path}'; skipping GRF segment.")
                continue
                
            x_vals_exp = np.linspace(x_offset_pct, x_offset_pct + 100.0, int(n_frames))
            # Sim x-axis for this segment (could use time if available, but consistent % is easier for concatenation)
            x_vals_sim = np.linspace(x_offset_pct, x_offset_pct + 100.0, int(n_frames))
            
            prepared_segments.append({
                "grf_path": grf_path,
                "n_frames": int(n_frames),
                "start_idx": current_idx,
                "end_idx": current_idx + int(n_frames),
                "x_vals_exp": x_vals_exp,
                "x_vals_sim": x_vals_sim,
                "x_offset": x_offset_pct
            })
            
            current_idx += int(n_frames)
            x_offset_pct += 100.0
            
    else:
        # Single trial fallback
        _ik_file, grf_file, _trc_file = get_trial_files_from_settings(settings)
        if not grf_file:
            warnings.append(
                "YAML/settings must specify a GRF trial file to generate GRF plots. "
                "Provide it either under collocation.settings (e.g. 'grf_file') or under an objective's args as 'grf_file'."
            )
            return plot_paths, warnings

        grf_path = _resolve_from_yaml_path(grf_file, yaml_path)
        if not os.path.exists(grf_path):
            warnings.append(f"GRF file not found at '{grf_path}'; skipping GRF plots.")
            return plot_paths, warnings
            
        n_points = int(solution_states.states.model.shape[0])
        # For single trial, use real time scaling if available, else 0-100
        pct_sim = _get_sim_percent_axis(solution_states, globals_dict)
        pct_exp = np.linspace(0.0, 100.0, n_points)
        
        prepared_segments.append({
            "grf_path": grf_path,
            "n_frames": n_points,
            "start_idx": 0,
            "end_idx": n_points,
            "x_vals_exp": pct_exp,
            "x_vals_sim": pct_sim,
            "x_offset": 0.0
        })

    # 2. Extract Simulated GRFs (full concatenated array)
    idx0 = int(getattr(model, "ext_forces", {}).get("idx", 0))
    n_grfs_sim_model = int(getattr(model, "ext_forces", {}).get("n", 0))
    if n_grfs_sim_model < 4:
        warnings.append(f"Model ext_forces has n={n_grfs_sim_model}; expected at least 4 (Fx/Fy for two feet). Skipping GRF plots.")
        return plot_paths, warnings

    grf_sim_full = np.asarray(solution_states.states.model[:, idx0 : idx0 + n_grfs_sim_model]).astype(float)

    def _safe_col(arr: np.ndarray, i: int) -> np.ndarray:
        if i < 0 or i >= arr.shape[1]:
            return np.full((arr.shape[0],), np.nan)
        return arr[:, i]

    # Sim indices: [Lx, Ly, Lz, Rx, Ry, Rz, ...]
    sim_l_fx_full = _safe_col(grf_sim_full, 0)
    sim_l_fy_full = _safe_col(grf_sim_full, 1)
    sim_r_fx_full = _safe_col(grf_sim_full, 3)
    sim_r_fy_full = _safe_col(grf_sim_full, 4)

    # 3. Plotting setup
    out_dir = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()
    out_dir = os.path.join(out_dir, "grf")
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # We will accumulate data for plotting
    # Helper to plot a component (Fx or Fy)
    def _plot_concatenated(component_name: str, sim_l_full, sim_r_full, out_name_suffix: str) -> str | None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        has_data = False

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        c_exp = color_cycle[0]
        c_sim = color_cycle[1]

        # Iterate segments
        for seg in prepared_segments:
            s_idx = seg["start_idx"]
            e_idx = seg["end_idx"]
            
            # Load Exp Data for this segment
            # We must load/segment each time because files differ
            grf_path = seg["grf_path"]
            n_p = seg["n_frames"]
            
            try:
                _, gait_avg_grfs, _ = segment_gait_averages(
                    grf_file=grf_path, ik_file=None, trc_file=None, n_points=n_p
                )
            except Exception:
                continue

            if gait_avg_grfs is None:
                continue
                
            # Mapping
            # Mapping logic based on component_name
            if component_name == "Fx":
                col_l = "1_ground_force_vx_mean"
                col_r = "ground_force_vx_mean"
            else: # Fy
                col_l = "1_ground_force_vy_mean"
                col_r = "ground_force_vy_mean"
            
            def _get_exp_data(df, c):
                if c in df.columns: return df[c].to_numpy(dtype=float)
                return np.full((n_p,), np.nan)

            exp_l = _get_exp_data(gait_avg_grfs, col_l)
            exp_r = _get_exp_data(gait_avg_grfs, col_r)
            
            # Extract Sim slice
            if e_idx > len(sim_l_full):
                 # Safety clip
                 e_idx = len(sim_l_full)
            if s_idx >= e_idx:
                 continue

            sim_l = sim_l_full[s_idx:e_idx]
            sim_r = sim_r_full[s_idx:e_idx]
            
            x_exp = seg["x_vals_exp"]
            x_sim = seg["x_vals_sim"]

            # Plot
            # Only label first segment
            lbl_exp = "exp" if not has_data else ""
            lbl_sim = "sim" if not has_data else ""
            
            axes[0].plot(x_exp, exp_l, color=c_exp, label=lbl_exp, linewidth=2, alpha=0.7)
            axes[0].plot(x_sim, sim_l, color=c_sim, label=lbl_sim, linewidth=2, linestyle="--")
            
            axes[1].plot(x_exp, exp_r, color=c_exp, label=lbl_exp, linewidth=2, alpha=0.7)
            axes[1].plot(x_sim, sim_r, color=c_sim, label=lbl_sim, linewidth=2, linestyle="--")
            
            # Separator
            if trials_config and seg["end_idx"] < len(sim_l_full):
                 sep = seg["x_offset"] + 100.0
                 axes[0].axvline(x=sep, color='k', linestyle=':', alpha=0.3)
                 axes[1].axvline(x=sep, color='k', linestyle=':', alpha=0.3)
            
            has_data = True

        if not has_data:
            plt.close(fig)
            return None

        axes[0].set_title(f"Left {component_name}")
        axes[0].set_xlabel("Gait cycle (%)")
        axes[0].set_ylabel("Force [N]")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title(f"Right {component_name}")
        axes[1].set_xlabel("Gait cycle (%)")
        axes[1].grid(True, alpha=0.3)
        
        handles, labels = axes[1].get_legend_handles_labels()
        if labels:
            # unique labels
            by_label = dict(zip(labels, handles, strict=False))
            fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)

        fig.suptitle(f"GRF {component_name}: simulated vs experimental")
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        
        out_path = os.path.join(out_dir, f"grf_{out_name_suffix}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    try:
        fx_path = _plot_concatenated("Fx", sim_l_fx_full, sim_r_fx_full, "fx")
        if fx_path: plot_paths.append(fx_path)
        
        fy_path = _plot_concatenated("Fy", sim_l_fy_full, sim_r_fy_full, "fy")
        if fy_path: plot_paths.append(fy_path)
        
    except Exception as e:
        warnings.append(f"Failed to generate GRF plots: {e}")

    return plot_paths, warnings


def _save_activation_end_plot(
    report_path: str,
    model,
    solution_states,
) -> tuple[list[str], list[str]]:
    """Save a heatmap of actuator activations for all nodes as PNG."""
    warnings: list[str] = []
    plot_paths: list[str] = []

    solution_states, _ = _normalize_solution_inputs(solution_states, None)

    # Verify actuator state matrix exists and has content
    actuator_states = getattr(solution_states.states, "actuator_model", None)
    if actuator_states is None:
        warnings.append("No actuator_model states found; skipping activation plot.")
        return plot_paths, warnings

    actuator_states = np.asarray(actuator_states)
    if actuator_states.ndim != 2 or actuator_states.shape[0] == 0 or actuator_states.shape[1] == 0:
        warnings.append("Empty actuator_model states; skipping activation plot.")
        return plot_paths, warnings

    # Resolve activation indices in actuator state vector
    idx_a = None
    if hasattr(model, "actuators") and hasattr(model.actuators, "idx"):
        idx_a = model.actuators.idx.get("a", None)
    if idx_a is None:
        warnings.append("Actuator activation indices not available; skipping activation plot.")
        return plot_paths, warnings

    idx_a = np.asarray(idx_a, dtype=int).reshape(-1)
    if idx_a.size == 0:
        warnings.append("No activation indices found; skipping activation plot.")
        return plot_paths, warnings

    if np.max(idx_a) >= actuator_states.shape[1] or np.min(idx_a) < 0:
        warnings.append("Activation indices are out of bounds for actuator_model states; skipping activation plot.")
        return plot_paths, warnings

    # Activation values for all nodes and all actuators: [n_nodes, n_actuators]
    activation_matrix = actuator_states[:, idx_a].astype(float)

    # Best-effort labels
    labels = []
    if hasattr(model.actuators, "names") and len(getattr(model.actuators, "names", [])) == activation_matrix.shape[1]:
        labels = [str(n) for n in model.actuators.names]
    else:
        labels = [f"a_{i+1}" for i in range(activation_matrix.shape[1])]

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()
    out_dir = os.path.join(out_dir, "activations")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "activations_all_nodes.png")

    n_nodes, n_actuators = activation_matrix.shape
    fig_width = max(10.0, min(24.0, 0.02 * n_nodes + 8.0))
    fig_height = max(6.0, min(24.0, 0.20 * n_actuators + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    im = ax.imshow(
        activation_matrix.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("Actuator activations across all nodes")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Actuator")

    y_ticks = np.arange(n_actuators)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Activation")

    try:
        fig.tight_layout()
        fig.savefig(out_path)
        plot_paths.append(out_path)
        warnings.append("Activation plot saved in 'activations/' (all nodes, all actuators).")
    finally:
        plt.close(fig)

    return plot_paths, warnings


def _save_ipopt_iteration_log(
    report_path: str,
    iteration_history: list[dict] | None,
) -> tuple[str | None, str | None]:
    """Save IPOPT per-iteration metrics into a sibling text file.

    Returns
    -------
    tuple[str | None, str | None]
        (log_path, warning_message). warning_message is None on success.
    """
    if not iteration_history:
        return None, "No IPOPT iteration history available; skipping iteration log file."

    report_abs = os.path.abspath(report_path)
    report_dir = os.path.dirname(report_abs) or os.getcwd()
    report_stem = os.path.splitext(os.path.basename(report_abs))[0]
    log_path = os.path.join(report_dir, f"{report_stem}_iterations.txt")

    headers = [
        "iter_count",
        "alg_mod",
        "obj_value",
        "inf_pr",
        "inf_du",
        "mu",
        "d_norm",
        "regularization_size",
        "alpha_du",
        "alpha_pr",
        "ls_trials",
    ]

    with open(log_path, "w") as fh:
        fh.write("# IPOPT Iteration Log\n")
        fh.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"# Rows: {len(iteration_history)}\n")
        fh.write("\t".join(headers) + "\n")

        integer_columns = {"iter_count", "alg_mod", "ls_trials"}

        for row in iteration_history:
            if not isinstance(row, dict):
                continue

            values = []
            for key in headers:
                value = row.get(key, "")

                if value == "" or value is None:
                    values.append("")
                    continue

                try:
                    numeric_value = float(value)
                except Exception:
                    values.append(str(value))
                    continue

                if key == "obj_value":
                    values.append(f"{numeric_value:.10e}")
                elif key == "mu":
                    if numeric_value > 0:
                        values.append(f"{np.log10(numeric_value):.1f}")
                    else:
                        values.append("nan")
                elif key in integer_columns:
                    values.append(str(int(round(numeric_value))))
                else:
                    values.append(f"{numeric_value:.2e}")
            fh.write("\t".join(values) + "\n")

    return log_path, None


def generate_optimization_report(
    report_path,
    info,
    model,
    tunable_const_indices,
    initial_constants=None,
    final_constants=None,
    n_trials=1,
    objectives=None,
    constraints=None,
    final_iter_count=None,
    final_inf_pr=None,
    final_inf_du=None,
    final_obj_value=None,
    solution_states=None,
    globals_dict=None,
    settings=None,
    yaml_path=None,
    save_plots=True,
    plot_joints=("hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"),
    objective_manager=None,
    constraint_manager=None,
    precomputed_objectives: dict[str, float] | None = None,
    precomputed_constraints: dict[str, float] | None = None,
    trials_config: list[dict] | None = None,
    iteration_history: list[dict] | None = None,
):
    """
    Generate a text report of optimization results.
    
    Parameters
    ----------
    report_path : str
        Path to save the report file
    info : dict
        IPOPT solver information dictionary
    model : BiosymModel
        The model containing variable information
    tunable_const_indices : array-like
        Indices of tunable constants in the model
    initial_constants : array-like, optional
        Initial values of shared constants before optimization
    final_constants : array-like, optional
        Final values of shared constants after optimization
    n_trials : int, optional
        Number of trials in the optimization (default=1)
    objectives : list, optional
        List of objective names used in the optimization
    constraints : list, optional
        List of constraint names used in the optimization
    final_iter_count : int, optional
        Final iteration count from IPOPT intermediate callback
    final_inf_pr : float, optional
        Final primal infeasibility from IPOPT intermediate callback
    final_inf_du : float, optional
        Final dual infeasibility from IPOPT intermediate callback
    final_obj_value : float, optional
        Final objective value from IPOPT intermediate callback
    solution_states : StatesDict, optional
        Solved trajectory, used for generating comparison plots.
    globals_dict : Globals, optional
        Globals (e.g., duration) associated with solution_states.
    settings : dict, optional
        Collocation settings dict (ideally parsed YAML) to locate the IK file.
    yaml_path : str, optional
        Path to the YAML file (used to resolve relative paths).
    save_plots : bool
        If True and solution_states are provided, saves angle plots as PNG.
    plot_joints : tuple[str, ...]
        Joint names to plot (model joint names without 'q_' prefix).
    objective_manager : object, optional
        Objective manager to re-evaluate objective function values.
    constraint_manager : object, optional
        Constraint manager to re-evaluate constraint values.
    precomputed_objectives : dict, optional
        Dictionary of precalculated objective values.
    precomputed_constraints : dict, optional
        Dictionary of precalculated constraint values.
    trials_config : list[dict], optional
         List of dictionaries containing configuration for each trial in a multi-trial problem.
         Each dict should have: 'ik_file', 'grf_file', 'n_frames'.
    iteration_history : list[dict], optional
        Full IPOPT intermediate callback history (one entry per iteration).
        If provided, a sibling file `<report_stem>_iterations.txt` is written.
    """
    # Ensure parent directory exists
    report_dir = os.path.dirname(report_path)
    if report_dir:  # Only create if there's a directory component
        os.makedirs(report_dir, exist_ok=True)

    # Helper: map block name -> configured weight (as provided in YAML/settings)
    def _get_configured_weights(settings_dict: dict | None, key: str) -> dict[str, object]:
        weights_by_name: dict[str, object] = {}
        if not isinstance(settings_dict, dict):
            return weights_by_name
        blocks = settings_dict.get(key, [])
        if not isinstance(blocks, list):
            return weights_by_name
        for block in blocks:
            if not isinstance(block, dict):
                continue
            name = block.get("name")
            if not name:
                continue
            weights_by_name[str(name)] = block.get("weight", None)
        return weights_by_name

    def _resolve_numeric_weight(configured_weight: object, runtime_weight: float | None, term_name: str, term_kind: str) -> float:
        """Return a numeric weight for reporting; raise on unresolved weights (strict mode)."""
        if runtime_weight is not None:
            try:
                return float(runtime_weight)
            except Exception:
                raise ValueError(
                    f"Invalid runtime weight for {term_kind} '{term_name}': {runtime_weight!r}."
                )

        if configured_weight is None:
            raise ValueError(
                f"Missing weight for {term_kind} '{term_name}'. Provide a numeric weight or a supported symbolic weight."
            )

        if isinstance(configured_weight, (int, float)):
            return float(configured_weight)

        if isinstance(configured_weight, str):
            w_str = configured_weight.strip()
            try:
                return float(w_str)
            except Exception:
                pass

            if w_str == "1/BW":
                try:
                    masses = [body["mass"] for body in model.dicts["bodies"]]
                    bw = float(np.sum(np.asarray(masses, dtype=float)))
                    if bw > 0:
                        return 1.0 / bw
                except Exception:
                    raise ValueError(
                        f"Failed to evaluate symbolic weight '1/BW' for {term_kind} '{term_name}'."
                    )

                raise ValueError(
                    f"Invalid bodyweight sum while evaluating '1/BW' for {term_kind} '{term_name}'."
                )

        raise ValueError(
            f"Unsupported configured weight for {term_kind} '{term_name}': {configured_weight!r}."
        )
    
    plot_paths: list[str] = []
    plot_warnings: list[str] = []
    solution_states, globals_dict = _normalize_solution_inputs(solution_states, globals_dict)

    report_series: dict[str, np.ndarray] = {}
    csv_export_path: str | None = None
    if solution_states is not None:
        report_series, series_warnings = _collect_solved_state_variables(
            model=model,
            solution_states=solution_states,
            globals_dict=globals_dict,
        )
        plot_warnings.extend(series_warnings)

        csv_export_path, csv_warning = _save_report_series_csv(
            report_path=report_path,
            series=report_series,
        )
        if csv_export_path:
            plot_warnings.append(f"Report variables CSV saved: {csv_export_path}")
        if csv_warning:
            plot_warnings.append(csv_warning)

    if save_plots and solution_states is not None:
        try:
            angle_paths, angle_warnings = _save_angle_plots(
                report_path=report_path,
                model=model,
                solution_states=solution_states,
                globals_dict=globals_dict,
                settings=settings,
                yaml_path=yaml_path,
                joint_names=list(plot_joints),
                trials_config=trials_config
            )
            plot_paths.extend(angle_paths)
            plot_warnings.extend(angle_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate angle plots: {e}")
            import traceback
            traceback.print_exc()

        # GRF plots (Fx, Fy)
        try:
            grf_paths, grf_warnings = _save_grf_plots(
                report_path=report_path,
                model=model,
                solution_states=solution_states,
                globals_dict=globals_dict,
                settings=settings,
                yaml_path=yaml_path,
                trials_config=trials_config
            )
            plot_paths.extend(grf_paths)
            plot_warnings.extend(grf_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate GRF plots: {e}")

        # Additional requested line plots from solved variables/derived series.
        try:
            acc_keys = [k for k in report_series if k.startswith("qdd_")]
            acc_paths, acc_warnings = _save_series_group_line_plots(
                report_path=report_path,
                series=report_series,
                keys=acc_keys,
                filename_prefix="joint_acceleration",
                y_label="Acceleration",
                title_prefix="Joint acceleration",
                output_subdir="joint_accelerations",
                trials_config=trials_config,
            )
            plot_paths.extend(acc_paths)
            plot_warnings.extend(acc_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate joint acceleration plots: {e}")

        try:
            moment_keys = [k for k in report_series if k.startswith("M_")]
            moment_paths, moment_warnings = _save_series_group_line_plots(
                report_path=report_path,
                series=report_series,
                keys=moment_keys,
                filename_prefix="joint_moment",
                y_label="Moment (Nm)",
                title_prefix="Joint moment",
                output_subdir="joint_moments",
                trials_config=trials_config,
            )
            plot_paths.extend(moment_paths)
            plot_warnings.extend(moment_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate joint moment plots: {e}")

        try:
            activation_keys = [k for k in report_series if k.startswith("activation_")]
            activation_paths, activation_warnings = _save_series_group_line_plots(
                report_path=report_path,
                series=report_series,
                keys=activation_keys,
                filename_prefix="activation",
                y_label="Activation",
                title_prefix="Muscle activation",
                output_subdir="muscle_activations",
                trials_config=trials_config,
            )
            plot_paths.extend(activation_paths)
            plot_warnings.extend(activation_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate muscle activation line plots: {e}")

        try:
            muscle_force_keys = [k for k in report_series if k.startswith("muscle_force_")]
            muscle_force_paths, muscle_force_warnings = _save_series_group_line_plots(
                report_path=report_path,
                series=report_series,
                keys=muscle_force_keys,
                filename_prefix="muscle_force",
                y_label="Force (N)",
                title_prefix="Muscle force",
                output_subdir="muscle_forces",
                trials_config=trials_config,
            )
            plot_paths.extend(muscle_force_paths)
            plot_warnings.extend(muscle_force_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate muscle force plots: {e}")

        try:
            tendon_force_keys = [k for k in report_series if k.startswith("tendon_force_")]
            tendon_force_paths, tendon_force_warnings = _save_series_group_line_plots(
                report_path=report_path,
                series=report_series,
                keys=tendon_force_keys,
                filename_prefix="tendon_force",
                y_label="Force (N)",
                title_prefix="Tendon force",
                output_subdir="tendon_forces",
                trials_config=trials_config,
            )
            plot_paths.extend(tendon_force_paths)
            plot_warnings.extend(tendon_force_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate tendon force plots: {e}")

        try:
            actuator_torque_keys = [k for k in report_series if k.startswith("actuator_torque_")]
            actuator_torque_paths, actuator_torque_warnings = _save_series_group_line_plots(
                report_path=report_path,
                series=report_series,
                keys=actuator_torque_keys,
                filename_prefix="actuator_torque",
                y_label="Torque",
                title_prefix="Generic actuator torque",
                output_subdir="actuator_torques",
                trials_config=trials_config,
            )
            plot_paths.extend(actuator_torque_paths)
            plot_warnings.extend(actuator_torque_warnings)
        except Exception as e:
            plot_warnings.append(f"Failed to generate generic actuator torque plots: {e}")

    iteration_log_path = None
    iteration_log_warning = None
    try:
        iteration_log_path, iteration_log_warning = _save_ipopt_iteration_log(
            report_path=report_path,
            iteration_history=iteration_history,
        )
    except Exception as e:
        iteration_log_warning = f"Failed to save IPOPT iteration log: {e}"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        if n_trials > 1:
            f.write("BIOSYM MULTI-TRIAL OPTIMIZATION REPORT\n")
        else:
            f.write("BIOSYM OPTIMIZATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if n_trials > 1:
            f.write(f"Number of Trials: {n_trials}\n\n")

        # Iteration log
        if iteration_log_path or iteration_log_warning:
            f.write("-"*80 + "\n")
            f.write("IPOPT ITERATION LOG\n")
            f.write("-"*80 + "\n")
            if iteration_log_path:
                f.write(f"  - Saved: {iteration_log_path}\n")
            if iteration_log_warning:
                f.write(f"  - {iteration_log_warning}\n")
            f.write("\n")

        # Plots
        if plot_paths or plot_warnings:
            f.write("-"*80 + "\n")
            f.write("PLOTS\n")
            f.write("-"*80 + "\n")
            # Include the IK/GRF source line if present, plus any other warnings.
            source_line = None
            for w in plot_warnings:
                if isinstance(w, str) and (
                    "segmented IK source" in w
                    or "Angle plots saved next to report" in w
                    or "Multi-trial concatenated" in w
                ):
                    source_line = w
                    break
            if source_line:
                f.write(f"  - {source_line}\n")
            for w in plot_warnings:
                if not isinstance(w, str) or not w.strip():
                    continue
                if source_line and w == source_line:
                    continue
                f.write(f"  - {w}\n")
            f.write("\n")
        
        # Compute final per-term values (if managers + solution are available)
        obj_values: dict[str, float] = {}
        con_values: dict[str, float] = {}
        obj_weights: dict[str, float] = {}
        con_weights: dict[str, float] = {}

        if precomputed_objectives:
            obj_values.update(precomputed_objectives)
        
        if precomputed_constraints:
            con_values.update(precomputed_constraints)

        # Configured weights from settings (may be numeric or strings like "1/BW")
        configured_obj_weights = _get_configured_weights(settings, "objectives")
        configured_con_weights = _get_configured_weights(settings, "constraints")
        if solution_states is not None:
            # Objectives
            if objective_manager is not None and hasattr(objective_manager, "_objectives"):
                try:
                    weights = list(getattr(objective_manager, "weights", []))
                    for i, obj in enumerate(getattr(objective_manager, "_objectives", [])):
                        try:
                            # Use obj_info to avoid shadowing the main 'info' argument
                            obj_info = obj._get_info() if hasattr(obj, "_get_info") else {}
                            name = obj_info.get("name") or getattr(obj, "name", None) or f"objective_{i}"
                            val = obj.get_objfun()(solution_states, globals_dict)
                            obj_values[str(name)] = float(np.asarray(val))
                            if i < len(weights):
                                obj_weights[str(name)] = float(weights[i])
                        except Exception:
                            continue
                except Exception:
                    obj_values = {}

            # Constraints: report max abs residual per constraint term
            if constraint_manager is not None and hasattr(constraint_manager, "_constraints"):
                try:
                    weights = list(getattr(constraint_manager, "weights", []))
                    for i, con in enumerate(getattr(constraint_manager, "_constraints", [])):
                        try:
                            # Use obj_info to avoid shadowing the main 'info' argument
                            obj_info = con._get_info() if hasattr(con, "_get_info") else {}
                            name = obj_info.get("name") or getattr(con, "name", None) or f"constraint_{i}"
                            cval = con.get_confun()(solution_states, globals_dict)
                            cval_np = np.asarray(cval)
                            con_values[str(name)] = float(np.max(np.abs(cval_np))) if cval_np.size else 0.0
                            if i < len(weights):
                                con_weights[str(name)] = float(weights[i])
                        except Exception:
                            continue
                except Exception:
                    con_values = {}

        # Objectives and Constraints
        if objectives is not None and len(objectives) > 0:
            f.write("-"*80 + "\n")
            f.write("OBJECTIVES\n")
            f.write("-"*80 + "\n")
            for obj_name in objectives:
                val = obj_values.get(obj_name)
                cfg_w = configured_obj_weights.get(obj_name, None)
                
                # If weight not found by full name, try stripping prefixes like "Trial 1: "
                if cfg_w is None and ": " in obj_name:
                    suffix = obj_name.split(": ", 1)[1]
                    cfg_w = configured_obj_weights.get(suffix, None)

                w = obj_weights.get(obj_name)

                if val is None and cfg_w is None and w is None:
                    f.write(f"  - {obj_name}\n")
                    continue

                # Prefer configured (YAML) weight for display; fall back to runtime numeric weight
                if cfg_w is not None:
                    w_str = f"{float(cfg_w):.2f}" if isinstance(cfg_w, (int, float)) else str(cfg_w)
                elif w is not None:
                    w_str = f"{float(w):.2f}"
                else:
                    w_str = "N/A"

                if val is not None:
                    numeric_weight = _resolve_numeric_weight(cfg_w, w, obj_name, "objective")
                    weighted_val = float(val) * numeric_weight
                    f.write(f"  - {obj_name}: weight = {w_str}, weighted_value = {weighted_val:.10e}\n")
                else:
                    f.write(f"  - {obj_name}: weight = {w_str}\n")
            f.write("\n")
        
        if constraints is not None and len(constraints) > 0:
            f.write("-"*80 + "\n")
            f.write("CONSTRAINTS\n")
            f.write("-"*80 + "\n")
            for const_name in constraints:
                val = con_values.get(const_name)
                cfg_w = configured_con_weights.get(const_name, None)
                w = con_weights.get(const_name)

                if val is None and cfg_w is None and w is None:
                    f.write(f"  - {const_name}\n")
                    continue

                # Prefer configured (YAML) weight for display; include numeric if available
                if cfg_w is not None:
                    if isinstance(cfg_w, (int, float)):
                        w_str = f"{float(cfg_w):.2f}"
                    else:
                        # If it is a symbolic setting (e.g. 1/BW), also show numeric weight if we have it.
                        w_str = f"{str(cfg_w)} ({float(w):.2f})" if w is not None else str(cfg_w)
                elif w is not None:
                    w_str = f"{float(w):.2f}"
                else:
                    w_str = "N/A"

                if val is not None:
                    numeric_weight = _resolve_numeric_weight(cfg_w, w, const_name, "constraint")
                    weighted_val = float(val) * numeric_weight
                    f.write(f"  - {const_name}: weight = {w_str}, weighted_value = {weighted_val:.10e}\n")
                else:
                    f.write(f"  - {const_name}: weight = {w_str}\n")
            f.write("\n")
        
        # Convergence status
        f.write("-"*80 + "\n")
        f.write("CONVERGENCE STATUS\n")
        f.write("-"*80 + "\n")
        
        # Robust status extraction
        status = info.get('status')
        if status is None:
            status = info.get(b'status')  # Try bytes key
        
        status_msg = info.get('status_msg')
        if status_msg is None:
            status_msg = info.get(b'status_msg')
        
        # Decode bytes message if needed
        if isinstance(status_msg, bytes):
            status_msg = status_msg.decode('utf-8', errors='replace')
            
        # Fallbacks
        if status is None:
            # If status missing, use -1 but also log keys to message for diagnosis
            status = -1
            if status_msg is None:
                keys_list = [str(k) for k in info]
                status_msg = f"Unknown (Info keys: {', '.join(keys_list)})"
        elif status_msg is None:
             status_msg = 'Unknown'

        f.write(f"Status Code: {status}\n")
        f.write(f"Status Message: {status_msg}\n")
        
        # Converged if status is 0 or 1 (Success/Acceptable)
        try:
            status_int = int(status)
            is_converged = status_int in [0, 1]
        except (ValueError, TypeError):
            is_converged = False

        f.write(f"Converged: {'Yes' if is_converged else 'No'}\n")
        # Use final_iter_count from intermediate callback if available, otherwise fall back to info
        iter_count = final_iter_count if final_iter_count is not None else None
        f.write(f"Iterations: {iter_count}\n\n")
        
        # IPOPT statistics
        f.write("-"*80 + "\n")
        f.write("IPOPT STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<40} {'Scaled':<20} {'Unscaled':<20}\n")
        f.write("-"*80 + "\n")
        
        # Objective - use final_obj_value from intermediate if available
        obj = final_obj_value if final_obj_value is not None else info.get('obj_val', None)
        if obj is not None:
            f.write(f"{'Objective':<40} {obj:<20.10e} {obj:<20.10e}\n")
        
        # Dual infeasibility - use final_inf_du from intermediate if available
        dual_inf = final_inf_du if final_inf_du is not None else None
        if dual_inf is not None:
            f.write(f"{'Dual infeasibility':<40} {dual_inf:<20.10e} {dual_inf:<20.10e}\n")
        
        # Constraint violation - use final_inf_pr from intermediate if available
        constr_viol = final_inf_pr if final_inf_pr is not None else None
        if constr_viol is not None:
            f.write(f"{'Constraint violation':<40} {constr_viol:<20.10e} {constr_viol:<20.10e}\n")
        
        # Complementarity
        compl = info.get('inf_compl', None)
        if compl is not None:
            f.write(f"{'Complementarity':<40} {compl:<20.10e} {compl:<20.10e}\n")
        
        # Overall NLP error (max of primal/dual infeasibility)
        if constr_viol is not None and dual_inf is not None:
            nlp_error = max(constr_viol, dual_inf)
            f.write(f"{'Overall NLP error':<40} {nlp_error:<20.10e} {nlp_error:<20.10e}\n")
        
        f.write("\n")
        
        # Shared constants comparison
        if initial_constants is not None and final_constants is not None:
            f.write("-"*80 + "\n")
            if n_trials > 1:
                f.write("SHARED TUNABLE CONSTANTS\n")
            else:
                f.write("TUNABLE CONSTANTS\n")
            f.write("-"*80 + "\n")
            
            # Get constant names from the model
            if tunable_const_indices is not None:
                const_vars = model.variables[model.variables['type'] == 'constant']
                tunable_names = const_vars.iloc[tunable_const_indices]['name'].values
                
                f.write(f"{'Name':<40} {'Initial':<20} {'Final':<20} {'Change (%)':<20}\n")
                f.write("-"*80 + "\n")
                
                for name, init_val, final_val in zip(tunable_names, initial_constants, final_constants, strict=False):
                    change_pct = ((final_val - init_val) / init_val * 100) if init_val != 0 else 0.0
                    f.write(f"{name:<40} {init_val:<20.6f} {final_val:<20.6f} {change_pct:<20.2f}\n")
                
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\nReport saved to: {report_path}")
