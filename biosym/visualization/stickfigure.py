import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def _extract_fk_vis_data(model, state_sequence, markers_exp=None):
    """Extract joint, site, and expected marker (experimental) positions.

    Parameters
    ----------
    model : BiosymModel
        Model providing `run["FK_vis"]` returning stacked (bodies [+ sites]) xyz rows.
    state_sequence : list
        Sequence of per-node state wrappers (as used in collocation) each
        exposing `.states` and `.constants` attributes.
    markers_exp : array-like, optional
        Experimental marker expectations shaped (N, 3*n_sites) with ordering
        [X | Y | Z] per site in model site order. If provided, will be split
        into (N, n_sites, 3).

    Returns
    -------
    dict
        {
          'joint0': (n_bodies, 3) first frame,
          'site0': (n_sites, 3) or None,
          'anim_joints': (N, n_bodies, 3),
          'anim_sites': (N, n_sites, 3) or None,
          'anim_exp': (N, n_sites, 3) or None,
          'n_bodies': int,
          'n_sites': int,
          'n_frames': int
        }
    """
    n_frames = len(state_sequence)
    n_bodies = len(model.dicts.get("bodies", []))
    n_sites = len(model.dicts.get("sites", [])) if model.dicts.get("sites") is not None else 0

    fk_vis = model.run["FK_vis"]

    joint_frames = []
    site_frames = []
    for i in range(n_frames):
        jp_all = fk_vis(state_sequence[i].states, state_sequence[i].constants)
        # Split
        joints = jp_all[:n_bodies, :] if n_bodies > 0 else jp_all
        sites = jp_all[n_bodies:n_bodies + n_sites, :] if n_sites > 0 else None
        joint_frames.append(joints)
        if sites is not None:
            site_frames.append(sites)

    anim_joint_positions = np.asarray(joint_frames)
    anim_site_positions = np.asarray(site_frames) if site_frames else None

    anim_exp_sites = None
    if markers_exp is not None and n_sites > 0:
        me = np.asarray(markers_exp)
        # Clamp length if mismatch
        if me.shape[0] != n_frames:
            nmin = min(me.shape[0], n_frames)
            anim_joint_positions = anim_joint_positions[:nmin]
            if anim_site_positions is not None:
                anim_site_positions = anim_site_positions[:nmin]
            me = me[:nmin]
            n_frames = nmin  # local shadow but we don't reuse outside extraction
        exp_X = me[:, 0:n_sites]
        exp_Y = me[:, n_sites:2 * n_sites] if me.shape[1] >= 2 * n_sites else np.zeros_like(exp_X)
        exp_Z = me[:, 2 * n_sites:3 * n_sites] if me.shape[1] >= 3 * n_sites else np.zeros_like(exp_X)
        anim_exp_sites = np.stack([exp_X, exp_Y, exp_Z], axis=2)

    return {
        "joint0": anim_joint_positions[0],
        "site0": anim_site_positions[0] if anim_site_positions is not None else None,
        "anim_joints": anim_joint_positions,
        "anim_sites": anim_site_positions,
        "anim_exp": anim_exp_sites,
        "n_bodies": n_bodies,
        "n_sites": n_sites,
        "n_frames": anim_joint_positions.shape[0],
    }


def _detect_case(joint0):
    """Determine whether to plot in 2D or 3D based on joint coordinates."""
    if np.any(np.all(joint0 == 0, axis=0)):
        non_zero_axes = np.where(np.any(joint0 != 0, axis=0))[0]
        if len(non_zero_axes) == 1:
            non_zero_axes = [0, 1] if non_zero_axes in [0, 1] else [2, 1]
        return "2D", non_zero_axes
    return "3D", [0, 1, 2]


def _setup_axes(joint0,exp0, r, n_frames):
    """Create figure/axes and set axis limits for standing vs animated cases."""
    case_, non_zero_axes = _detect_case(joint0)
    if case_ == "2D":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

    if n_frames > 1:
        pos_list = [joint0[None, ...]]  # placeholder; limits recomputed later
    # Limits computed later in main to include all frames (need full arrays) -> done outside
    return fig, ax, case_, non_zero_axes


def _compute_limits(case_, non_zero_axes, anim_joints, anim_sites, anim_exp, joint0, site0, exp0, pad_ratio: float = 0.05):
    """Compute axis limits given available position arrays.

    Parameters
    ----------
    case_ : str
        '2D' or '3D'.
    non_zero_axes : list[int]
        Axes retained in 2D mode.
    anim_joints, anim_sites, anim_exp : np.ndarray or None
        Full trajectory arrays (N, items, 3) when animated; may be None.
    joint0, site0, exp0 : np.ndarray or None
        Single-frame arrays for standing case.
    pad_ratio : float, default 0.05
        Fractional padding applied to each axis range. (For standing we may
        pass a larger value from the caller.)
    """
    if anim_joints.shape[0] > 1:  # animation
        pos_list = [anim_joints]
        if anim_sites is not None:
            pos_list.append(anim_sites)
        if anim_exp is not None:
            pos_list.append(anim_exp)
        all_pos = np.concatenate(pos_list, axis=1)  # (N, M, 3)
        min_ = np.min(all_pos, axis=(0, 1))
        max_ = np.max(all_pos, axis=(0, 1))
    else:  # standing
        stack = [joint0]
        if site0 is not None:
            stack.append(site0)
        if exp0 is not None:
            stack.append(exp0)
        all_pos = np.vstack(stack)
        min_ = np.min(all_pos, axis=0)
        max_ = np.max(all_pos, axis=0)
    d = np.abs((max_ - min_) * pad_ratio)
    min_ -= d
    max_ += d
    if case_ == "2D":
        return (min_[non_zero_axes[0]], max_[non_zero_axes[0]], min_[non_zero_axes[1]], max_[non_zero_axes[1]])
    return (min_[0], max_[0], min_[1], max_[1], min_[2], max_[2])


def _draw_initial_frame(
    ax,
    case_,
    non_zero_axes,
    joint0,
    site0,
    exp0,
    plot_sites,
    plot_expected,
    joint_markersize: float = 6.0,
    site_markersize: float = 4.0,
    expected_markersize: float = 4.0,
):
    """Draw joints, segments, optional sites & expected markers for first frame.

    Parameters
    ----------
    joint_markersize : float
        Marker size (points) for joint centers.
    site_markersize : float
        Marker size (points) for simulated site (marker) positions (red).
    expected_markersize : float
        Marker size (points) for experimental expected markers (green).
    """
    joints_art = []
    for i in range(joint0.shape[0]):
        if case_ == "2D":
            (l,) = ax.plot(
                joint0[i, non_zero_axes[0]],
                joint0[i, non_zero_axes[1]],
                c="k",
                marker="o",
                markersize=joint_markersize,
            )
        else:
            (l,) = ax.plot(
                joint0[i, 0],
                joint0[i, 1],
                joint0[i, 2],
                c="k",
                marker="o",
                markersize=joint_markersize,
            )
        joints_art.append(l)

    site_artists = []
    if plot_sites and site0 is not None:
        for i in range(site0.shape[0]):
            if case_ == "2D":
                (s,) = ax.plot(
                    site0[i, non_zero_axes[0]],
                    site0[i, non_zero_axes[1]],
                    c="r",
                    marker="o",
                    linestyle="None",
                    markersize=site_markersize,
                )
            else:
                (s,) = ax.plot(
                    [site0[i, 0]],
                    [site0[i, 1]],
                    [site0[i, 2]],
                    c="r",
                    marker="o",
                    linestyle="None",
                    markersize=site_markersize,
                )
            site_artists.append(s)

    exp_artists = []
    if plot_expected and exp0 is not None:
        for i in range(exp0.shape[0]):
            if case_ == "2D":
                (g,) = ax.plot(
                    exp0[i, non_zero_axes[0]],
                    exp0[i, non_zero_axes[1]],
                    c="g",
                    marker="o",
                    linestyle="None",
                    markersize=expected_markersize,
                )
            else:
                (g,) = ax.plot(
                    [exp0[i, 0]],
                    [exp0[i, 1]],
                    [exp0[i, 2]],
                    c="g",
                    marker="o",
                    linestyle="None",
                    markersize=expected_markersize,
                )
            exp_artists.append(g)
    return joints_art, site_artists, exp_artists


def _gather_connections(model, n_bodies):
    """Return body-to-body connection index pairs from topology tree."""
    connections = []
    def _walk(topology):
        for node in topology:
            body_name = node["name"]
            try:
                body_idx = [b["name"] for b in model.dicts["bodies"]].index(body_name)
            except ValueError:
                continue
            for child in node["children"]:
                child_name = child["name"]
                try:
                    child_idx = [b["name"] for b in model.dicts["bodies"]].index(child_name)
                except ValueError:
                    continue
                connections.append((body_idx, child_idx))
                _walk([child])
    _walk(model.topology_tree)
    return np.array([c for c in connections if c[0] < n_bodies and c[1] < n_bodies])


def _add_segments(ax, case_, non_zero_axes, joint0, connections):
    """Draw initial body segments and return their artists."""
    segs = []
    for c in connections:
        if case_ == "2D":
            (l,) = ax.plot(joint0[c][:, non_zero_axes[0]], joint0[c][:, non_zero_axes[1]], c="k")
        else:
            (l,) = ax.plot(joint0[c][:, 0], joint0[c][:, 1], joint0[c][:, 2], c="k")
        segs.append(l)
    return segs


def _create_update_func(anim_joints, anim_sites, anim_exp, joints_art, site_artists, exp_artists, segments, connections, case_, non_zero_axes, dt, ax, model, hascontact, contact_plot_objects):
    """Return a Matplotlib FuncAnimation update callback."""
    ispaused = False
    speed_multiplier = 1.0
    n_frames = anim_joints.shape[0]

    speed_text = ax.figure.text(
        0.5, 0.98, f"Speed: {speed_multiplier:.1f}x | Controls: Space=Pause, ↑↓=Speed", ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray")
    ) if n_frames > 1 else None

    def on_key_press(event):
        nonlocal ispaused, speed_multiplier
        if event.key == " ":
            ispaused = not ispaused
        elif event.key == "up":
            speed_multiplier = min(speed_multiplier * 1.2, 10.0)
        elif event.key == "down":
            speed_multiplier = max(speed_multiplier / 1.2, 0.1)
        if speed_text is not None:
            speed_text.set_text(f"Speed: {speed_multiplier:.1f}x | Controls: Space=Pause, ↑↓=Speed")
            ax.figure.canvas.draw_idle()

    if n_frames > 1:
        ax.figure.canvas.mpl_connect("key_press_event", on_key_press)

    last_update_time = 0.0
    current_frame = 0

    def update(_):
        nonlocal last_update_time, current_frame
        if n_frames == 1:
            return []
        import time
        now = time.time()
        if last_update_time == 0:
            last_update_time = now
        # Pause logic
        # (We just skip advancement if paused)
        # Compute frames to advance
        elapsed = now - last_update_time
        frames_to_advance = int(elapsed * speed_multiplier / dt)
        if frames_to_advance >= 1:
            current_frame = (current_frame + frames_to_advance) % n_frames
            last_update_time = now
        f = current_frame
        # Joints
        for i in range(anim_joints.shape[1]):
            if case_ == "2D":
                joints_art[i].set_data([[anim_joints[f][i, non_zero_axes[0]]], [anim_joints[f][i, non_zero_axes[1]]]])
            else:
                joints_art[i].set_data([[anim_joints[f][i, 0]], [anim_joints[f][i, 1]]])
                joints_art[i].set_3d_properties(anim_joints[f][i, 2])
        # Segments
        for i, c in enumerate(connections):
            if case_ == "2D":
                segments[i].set_data([
                    [anim_joints[f][c][0, non_zero_axes[0]], anim_joints[f][c][1, non_zero_axes[0]]],
                    [anim_joints[f][c][0, non_zero_axes[1]], anim_joints[f][c][1, non_zero_axes[1]]],
                ])
            else:
                a = anim_joints[f][c][0]
                b = anim_joints[f][c][1]
                segments[i].set_data([a[0], b[0]], [a[1], b[1]])
                segments[i].set_3d_properties([a[2], b[2]])
        # Sites (sim)
        if anim_sites is not None and len(site_artists) == anim_sites.shape[1]:
            for i in range(anim_sites.shape[1]):
                if case_ == "2D":
                    site_artists[i].set_data([[anim_sites[f][i, non_zero_axes[0]]], [anim_sites[f][i, non_zero_axes[1]]]])
                else:
                    site_artists[i].set_data([[anim_sites[f][i, 0]]], [[anim_sites[f][i, 1]]])
                    site_artists[i].set_3d_properties([anim_sites[f][i, 2]])
        # Expected markers
        if anim_exp is not None and len(exp_artists) == anim_exp.shape[1]:
            for i in range(anim_exp.shape[1]):
                if case_ == "2D":
                    exp_artists[i].set_data([[anim_exp[f][i, 0]], [anim_exp[f][i, 1]]])
                else:
                    exp_artists[i].set_data([[anim_exp[f][i, 0]]], [[anim_exp[f][i, 1]]])
                    exp_artists[i].set_3d_properties([anim_exp[f][i, 2]])
        ax.set_title(f"T = {f * dt:.2f} s")
        if hascontact and contact_plot_objects is not None:
            # Ensure tuple of 4 lists; skip if malformed to avoid IndexError.
            if (
                isinstance(contact_plot_objects, tuple)
                and len(contact_plot_objects) == 4
                and all(obj is not None for obj in contact_plot_objects)
            ):
                model.gc_model.plot(
                    False,
                    model,
                    mode="update",
                    ax=ax,
                    case=case_,
                    non_zero_axes=non_zero_axes,
                    frame=f,
                    plot_objects=contact_plot_objects,
                )
        return []

    return update


def plot_stick_figure(
    model,
    states,
    dt=0.01,
    frame=None,
    plot_sites=True,
    plot_expected=True,
    joint_markersize: float = 6.0,
    site_markersize: float = 4.0,
    expected_markersize: float = 4.0,
    **kwargs,
):
    """Plot (or animate) a stick-figure of the model with optional sites & markers.

    This refactored implementation splits the original monolithic logic into
    modular helpers to clarify the distinct phases:
    1) FK data extraction (bodies, sites, expected markers)
    2) Figure/axes setup (2D vs 3D)
    3) Axis limit computation (standing vs animation)
    4) Initial frame drawing
    5) Optional animation update loop (multi-node trajectories)

    Parameters
    ----------
    model : BiosymModel
        The biomechanical model with compiled FK functions.
    states : tuple
        Tuple (state_sequence, globals) where state_sequence is a list-like
        of per-node state wrappers (collocation solution) and globals may
        provide duration (dur) for timestep derivation.
    dt : float, default 0.01
        Base timestep; if globals supplied, overridden by dur / nnodes.
    frame : int or None
        Reserved for future single-frame extraction; currently unused.
    plot_sites : bool, default True
        Whether to display simulated site positions (red).
    plot_expected : bool, default True
        Whether to display experimental expected markers (green) when
        `markers_exp` kwarg is provided.
    markers_exp : array-like, optional (via **kwargs)
        Experimental markers shaped (N, 3*n_sites).

    Notes
    -----
    - Standing (n_nodes == 1): still plots expected markers if provided.
    - Walking (n_nodes > 1): creates an interactive animation with pause &
      speed controls (space / up / down keys).
    - Sites are not states; they are derived via FK_vis appended rows.
    - Function preserves previous external signature except for new optional
      flags enabling selective plotting.

    Returns
    -------
    matplotlib.animation.FuncAnimation or None
        Animation object when multiple frames; otherwise None after showing.
    """
    # Backward compatibility: previously this function accepted just a state_sequence
    # (e.g., `plot_stick_figure(model, self.x, ...)`) where `self.x` was a StatesDict-like
    # object. New API prefers `(state_sequence, globals)` tuple. Detect form here.
    if isinstance(states, tuple) and len(states) == 2:
        state_sequence, globals_obj = states
    else:
        state_sequence = states
        globals_obj = None

    if globals_obj is not None:
        try:
            dt = globals_obj.dur / len(state_sequence)
        except AttributeError:
            pass  # keep provided dt if globals lacks `dur`
    n_frames = len(state_sequence)

    # Contact model? (Used in updates)
    hascontact = hasattr(model, "gc_model") and model.gc_model is not None

    markers_exp = kwargs.get("markers_exp", None)
    data = _extract_fk_vis_data(model, state_sequence, markers_exp if plot_expected else None)
    joint_positions = data["joint0"]
    site_positions = data["site0"] if plot_sites else None
    anim_joint_positions = data["anim_joints"]
    anim_site_positions = data["anim_sites"] if plot_sites else None
    anim_exp_sites = data["anim_exp"] if plot_expected else None
    n_sites = data["n_sites"]
    n_bodies = data["n_bodies"]
    n_frames = data["n_frames"]

    fig, ax, case_, non_zero_axes = _setup_axes(joint_positions, site_positions, None, n_frames)

    # Prepare expected markers first frame (standing) for limit computation
    exp0 = None
    if anim_exp_sites is not None:
        exp0 = anim_exp_sites[0]

    # Compute and set limits
    # Allow user override of padding; enlarge default for standing (single frame)
    user_pad = kwargs.get("pad_ratio", 0.25 if n_frames == 1 else 0.05)
    limits = _compute_limits(case_, non_zero_axes, anim_joint_positions, anim_site_positions, anim_exp_sites,
                              joint_positions, site_positions, exp0, pad_ratio=user_pad)
    if case_ == "2D":
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    else:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_zlim(limits[4], limits[5])

    connections = _gather_connections(model, n_bodies)

    # Plot all joint centers for the first frame
    exp0_for_draw = exp0 if plot_expected else None
    joints, site_artists, exp_site_artists = _draw_initial_frame(
        ax,
        case_,
        non_zero_axes,
        joint_positions,
        site_positions,
        exp0_for_draw,
        plot_sites,
        plot_expected,
        joint_markersize=joint_markersize,
        site_markersize=site_markersize,
        expected_markersize=expected_markersize,
    )
    # Plot all connections
    segments = _add_segments(ax, case_, non_zero_axes, joint_positions, connections)

    plot_objects = None
    if hascontact:
        # Initialize and retain plot objects for contact model so update phase can reuse them.
        plot_objects = model.gc_model.plot(
            state_sequence, model, mode="init", ax=ax, case=case_, non_zero_axes=non_zero_axes
        )

    # Set axis labels
    ax.set_xlabel("X-axis [m]")
    ax.set_ylabel("Y-axis [m]")
    if case_ == "3D":
        ax.set_zlabel("Z-axis [m]")
        ax.set_title("Stick Figure (3D)")
        ax.view_init(elev=20, azim=30)

    if n_frames == 1:  # standing
        ax.set_title("Stick Figure")
        plt.tight_layout()
        plt.show()
        return None

    update = _create_update_func(
        anim_joint_positions,
        anim_site_positions,
        anim_exp_sites,
        joints,
        site_artists,
        exp_site_artists,
        segments,
        connections,
        case_,
        non_zero_axes,
        dt,
        ax,
        model,
        hascontact,
        plot_objects,
    )
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(n_frames), interval=50, blit=False
    )
    plt.show()
    # Attach contact plot objects for use inside update via closure if needed
    if hascontact:
        # Monkey-patch attribute onto animation for potential external use/debugging
        ani._contact_plot_objects = plot_objects  # noqa: SLF001 (intentional internal attribute)
        ani._contact_state_sequence = state_sequence
    return ani
