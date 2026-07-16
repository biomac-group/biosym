"""
create_initial_guess.py
=======================
Two-step pipeline to produce the warm-start pickle file that the
IMU tracking script (script_imu_tracking.py) reads as its initial guess.

Step 1 — Standing equilibrium (1 node)
    Solves a single-node static equilibrium to find a balanced upright
    posture.  Result saved to  dorschky_demo/result/standing2d.pkl

Step 2 — Walking trajectory (100 nodes, 1.3 m/s)
    Tracks population-average joint angles from the biosym example data
    (example_data/gait_avg_joint_angles.mot) over a half gait cycle at
    1.3 m/s.  Warm-started from the standing solution.
    Result saved to  dorschky_demo/result/walking2d.pkl

Usage
-----
    uv run python dorschky_demo/create_initial_guess.py

Optional flags
--------------
    --skip-standing      Skip Step 1 (use existing dorschky_demo/result/standing2d.pkl)
    --no-solve           Set up both OCPs but skip the actual IPOPT solve
    --max-iter-standing  Max IPOPT iterations for the standing solve (default 3000)
    --max-iter-walking   Max IPOPT iterations for the walking solve  (default 6000)
    --visualize          Show stick-figure animation after each solve
"""

import argparse
import os
import sys
import time

# ── Path setup ───────────────────────────────────────────────────────────────
scripts_dir = os.path.dirname(os.path.abspath(__file__))
demo_dir = os.path.dirname(scripts_dir)
repo_root = os.path.dirname(demo_dir)
src_dir = os.path.join(demo_dir, "src")

for p in (src_dir, demo_dir, repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)

# Set CWD to repo root — required so relative paths in YAML configs
# (e.g. example_data/, dorschky_demo/models/) resolve correctly
os.chdir(repo_root)

# Ensure the result folder exists
RESULT_DIR = os.path.join(demo_dir, "result")
os.makedirs(RESULT_DIR, exist_ok=True)



# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import jax.numpy as jnp
import yaml
from biosym.ocp import collocation
from biosym.utils import read_mot
from biosym.model.model import load_model


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cfg(filename: str) -> str:
    """Return the absolute path to a config file in dorschky_demo/configs/."""
    return os.path.join(demo_dir, "configs", filename)


def _print_section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_result(label: str, info: dict) -> None:
    print(f"\n  {label} result:")
    print(f"    Status   : {info.get('status_msg', 'n/a')}")
    print(f"    Objective: {info.get('obj_val', float('nan')):.6g}")
    print(f"    Iters    : {info.get('iter', 'n/a')}")


# ── Step 1: Standing ──────────────────────────────────────────────────────────

def run_standing(args) -> None:
    """Solve the 1-node standing equilibrium OCP."""
    _print_section("Step 1 · Standing Equilibrium  (1 node)")

    standing_yaml = _cfg("standing2d.yaml")
    print(f"  Config : {standing_yaml}")

    with open(standing_yaml, "r") as f:
        config = yaml.safe_load(f)["collocation"]

    config["settings"]["model"] = os.path.abspath(
        os.path.join(repo_root, config["settings"]["model"])
    )
    if "output" in config["settings"]:
        config["settings"]["output"]["file"] = os.path.abspath(
            os.path.join(RESULT_DIR, "standing2d.pkl")
        )

    model = load_model(config["settings"]["model"], force_rebuild=True)
    ocp = collocation.Collocation(model, config)

    # ── Set a physically correct standing initial guess ──
    total_mass = sum(
        float(body["mass"][0]) if isinstance(body["mass"], (list, np.ndarray)) else float(body["mass"])
        for body in ocp.model.dicts["bodies"]
    )
    total_weight = total_mass * 9.81
    half_weight = total_weight / 4.0

    # 1. Update pelvis translation and joint angles to a symmetric standing posture with slightly bent knees
    ig_states = ocp.initial_guess_states
    q_ig = np.zeros((1, 9))
    q_ig[0, 0] = 0.0       # pelvis_tx
    q_ig[0, 1] = 0.048     # pelvis_ty (places feet on ground given default pos=0.9m)
    q_ig[0, 2] = 0.0       # pelvis_tilt
    q_ig[0, 3] = 0.05      # hip_r
    q_ig[0, 4] = -0.1      # knee_r
    q_ig[0, 5] = 0.05      # ankle_r
    q_ig[0, 6] = 0.05      # hip_l
    q_ig[0, 7] = -0.1      # knee_l
    q_ig[0, 8] = 0.05      # ankle_l
    ig_states = ig_states.replace(q=jnp.asarray(q_ig))

    # Place contact points physically on ground under feet and seed them with balanced forces (1/4 body weight each)
    if hasattr(ocp.model, "gc_model") and ocp.model.gc_model is not None:
        gc_seed = np.zeros((1, ocp.model.gc_model.get_n_states()))
        # heel_r: x=-0.06, y=0.0, fx=0, fy=0.25 BW
        gc_seed[0, 0] = -0.06
        gc_seed[0, 1] = 0.0
        gc_seed[0, 2] = 0.0
        gc_seed[0, 3] = 0.25
        # toe_r: x=0.1636, y=0.0, fx=0, fy=0.25 BW
        gc_seed[0, 4] = 0.1636
        gc_seed[0, 5] = 0.0
        gc_seed[0, 6] = 0.0
        gc_seed[0, 7] = 0.25
        # heel_l: x=-0.06, y=0.0, fx=0, fy=0.25 BW
        gc_seed[0, 8] = -0.06
        gc_seed[0, 9] = 0.0
        gc_seed[0, 10] = 0.0
        gc_seed[0, 11] = 0.25
        # toe_l: x=0.1636, y=0.0, fx=0, fy=0.25 BW
        gc_seed[0, 12] = 0.1636
        gc_seed[0, 13] = 0.0
        gc_seed[0, 14] = 0.0
        gc_seed[0, 15] = 0.25
        ig_states = ig_states.replace(gc_model=jnp.asarray(gc_seed))
    
    # 2. Update joint moments tau to match passive/active actuator forces if tau is an active state
    if ig_states.tau is not None:
        forces_act = ocp.model.run["actuator_model"](ig_states, ocp.model.default_constants).flatten()
        tau_ig = jnp.asarray(ig_states.tau).at[0].set(forces_act)
        ig_states = ig_states.replace(tau=tau_ig)
    from biosym.ocp.utils.settings import filter_active_states
    ocp.initial_guess_states = filter_active_states(ig_states, ocp.settings["active_states"])
    ocp.x0 = collocation.utils.states_dict_to_x(ocp.initial_guess_states, None)

    # Override max_iter if requested on command line
    ocp.nlp.add_option("max_iter", args.max_iter_standing)
    if args.derivative_test:
        ocp.nlp.add_option("derivative_test", "first-order")
        ocp.nlp.add_option("derivative_test_tol", 1e-3)
        ocp.nlp.add_option("max_iter", 5)  # short run just to test derivatives
        print("  [--derivative-test] Running IPOPT derivative test on standing...")

    if args.no_solve:
        print("  [--no-solve] Skipping IPOPT solve for standing.")
        return

    t0 = time.time()
    solution = ocp.solve(visualize=args.visualize)
    info = solution.info
    elapsed = time.time() - t0

    _print_result("Standing", info)
    print(f"    Time     : {elapsed:.1f} s")


# ── Step 2: Walking ───────────────────────────────────────────────────────────

def run_walking(args) -> None:
    """Solve the 100-node walking trajectory OCP."""
    _print_section("Step 2 · Walking Trajectory  (100 nodes, 1.3 m/s)")

    walking_yaml = _cfg("walking2d_initial_guess.yaml")
    print(f"  Config : {walking_yaml}")

    with open(walking_yaml, "r") as f:
        config = yaml.safe_load(f)["collocation"]

    config["settings"]["model"] = os.path.abspath(
        os.path.join(repo_root, config["settings"]["model"])
    )
    if "output" in config["settings"]:
        config["settings"]["output"]["file"] = os.path.abspath(
            os.path.join(RESULT_DIR, "walking2d.pkl")
        )
    if "initial_guess" in config and config["initial_guess"].get("type") == "from_file":
        config["initial_guess"]["file"] = os.path.abspath(
            os.path.join(RESULT_DIR, "standing2d.pkl")
        )

    model = load_model(config["settings"]["model"], force_rebuild=True)
    ocp = collocation.Collocation(model, config)

    # Build a gait-like warm start instead of tiling a static standing node.
    nnodes = ocp.settings["nnodes_dur"]
    coord_names = list(ocp.model.coordinates.names)
    q_seed = np.array(ocp.initial_guess_states.q[:nnodes])

    angle_df = read_mot("example_data/gait_avg_joint_angles.mot")
    grid_target = np.linspace(0.0, 1.0, nnodes)

    # Name-based mapping to match coordinates in the model to columns in gait average data
    mapping = {
        "hip_r": "hip_flexion_r",
        "hip_l": "hip_flexion_l",
        "knee_r": "knee_angle_r",
        "knee_l": "knee_angle_l",
        "ankle_r": "ankle_angle_r",
        "ankle_l": "ankle_angle_l",
    }

    for i, name in enumerate(coord_names):
        name_clean = name[2:] if name.startswith("q_") else name
        mapped_name = mapping.get(name_clean, name_clean)
        col = f"{mapped_name}_mean"
        if col not in angle_df.columns:
            print(f"  Warning: joint average column '{col}' not found for coordinate '{name}'. Skipping.")
            continue
        vals = angle_df[col].to_numpy(dtype=float)
        if vals.shape[0] != nnodes:
            vals = np.interp(grid_target, np.linspace(0.0, 1.0, vals.shape[0]), vals)
        if name_clean == "pelvis_ty":
            vals = vals - 0.9
        q_seed[:, i] = vals

    pelvis_tx_name = next((name for name in coord_names if name.endswith("pelvis_tx")), None)
    dx = np.zeros(nnodes)
    if pelvis_tx_name:
        pelvis_tx_idx = coord_names.index(pelvis_tx_name)
        dur_guess = float(np.asarray(ocp.initial_guess_globals.dur).reshape(-1)[0])
        speed_target = float(np.mean(np.asarray(ocp.settings["bounds"]["speed"], dtype=float)))
        dx = np.linspace(0.0, dur_guess * speed_target, nnodes)
        q_seed[:, pelvis_tx_idx] = dx

    dt = max(float(np.asarray(ocp.initial_guess_globals.dur).reshape(-1)[0]) / max(nnodes - 1, 1), 1e-6)
    qd_seed = np.gradient(q_seed, dt, axis=0)

    ig_states = ocp.initial_guess_states.replace(q=jnp.asarray(q_seed))
    ig_states = ig_states.replace(qd=jnp.asarray(qd_seed))

    # Also shift the contact point x-positions in gc_model by dx
    if pelvis_tx_name and hasattr(ocp.model, "contact_model") and ocp.model.contact_model is not None:
        gc_seed = np.array(ocp.initial_guess_states.gc_model[:nnodes])
        for i in range(len(ocp.model.contact_model.cps)):
            gc_seed[:, 4 * i + 0] = gc_seed[:, 4 * i + 0] + dx
        ig_states = ig_states.replace(gc_model=jnp.asarray(gc_seed))

    tau_seed = ocp.model.run["actuator_model"](ig_states, ocp.model.default_constants)[:nnodes]
    ig_states = ig_states.replace(tau=tau_seed)
    from biosym.ocp.utils.settings import filter_active_states
    ocp.initial_guess_states = filter_active_states(ig_states, ocp.settings["active_states"])
    ocp.x0 = collocation.utils.states_dict_to_x(ocp.initial_guess_states, ocp.initial_guess_globals)

    if args.no_solve:
        print("  [--no-solve] Skipping IPOPT solve for walking.")
        return

    # Override max_iter if requested on command line
    ocp.nlp.add_option("max_iter", args.max_iter_walking)

    t0 = time.time()
    solution = ocp.solve(visualize=args.visualize)
    info = solution.info
    elapsed = time.time() - t0

    _print_result("Walking", info)
    print(f"    Time     : {elapsed:.1f} s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-step initial-guess pipeline: standing → walking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-standing",
        action="store_true",
        help="Skip Step 1 and reuse an existing dorschky_demo/result/standing2d.pkl.",
    )
    parser.add_argument(
        "--no-solve",
        action="store_true",
        help="Set up the OCPs but do NOT call the IPOPT solver (dry-run).",
    )
    parser.add_argument(
        "--max-iter-standing",
        type=int,
        default=3000,
        help="IPOPT iteration limit for the standing solve.",
    )
    parser.add_argument(
        "--max-iter-walking",
        type=int,
        default=400,
        help="IPOPT iteration limit for the walking solve.",
    )
    parser.add_argument(
        "--skip-walking",
        action="store_true",
        help="Skip Step 2 (walking trajectory solve).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show stick-figure animation after each solve.",
    )
    parser.add_argument(
        "--derivative-test",
        action="store_true",
        help="Run IPOPT derivative test on standing problem (verifies Jacobian correctness).",
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   BioSYM  ·  Initial Guess Generator  ·  Dorschky Demo              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if args.skip_standing:
        standing_pkl = os.path.join(RESULT_DIR, "standing2d.pkl")
        _print_section("Step 1 · Standing  [SKIPPED]")
        if os.path.exists(standing_pkl):
            print(f"  Using existing file: {standing_pkl}")
        else:
            print(
                "  WARNING: --skip-standing was set, but no standing2d.pkl was "
                f"found at {standing_pkl}.  Step 2 will likely fail."
            )
    else:
        run_standing(args)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if args.skip_walking:
        _print_section("Step 2 · Walking  [SKIPPED]")
    else:
        run_walking(args)

    _print_section("Done")
    walking_pkl = os.path.join(RESULT_DIR, "walking2d.pkl")
    if os.path.exists(walking_pkl):
        print(f"  Initial guess saved to: {walking_pkl}")
        print()
        print("  You can now run the IMU tracking script:")
        print(
            "    uv run python dorschky_demo/scripts/script_imu_tracking.py "
            "--subject P02 --trial-type normwalking"
        )
    print()


if __name__ == "__main__":
    main()
