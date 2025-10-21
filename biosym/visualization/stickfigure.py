import matplotlib.pyplot as plt
import matplotlib
import warnings
import numpy as np
from matplotlib import animation


def _extract_fk_marker_data(model, state_sequence, markers_exp=None):
    """Extract joint, site, and expected marker (experimental) positions.

    Parameters
    ----------
    model : BiosymModel
        Model providing `run["FK_marker"]` returning stacked (bodies [+ markers]) xyz rows.
    state_sequence : list
        Sequence of per-node state wrappers (as used in collocation) each
        exposing `.states` and `.constants` attributes.
    markers_exp : array-like, optional
        Experimental marker expectations shaped (N, 3*n_markers) with ordering
        [X | Y | Z] per site in model site order. If provided, will be split
        into (N, n_markers, 3).

    Returns
    -------
    dict
        {
          'joint0': (n_bodies, 3) first frame,
          'site0': (n_markers, 3) or None,
          'anim_joints': (N, n_bodies, 3),
          'anim_markers': (N, n_markers, 3) or None,
          'anim_exp': (N, n_markers, 3) or None,
          'n_bodies': int,
          'n_markers': int,
          'n_frames': int
        }
    """
    n_frames = len(state_sequence)
    n_bodies = len(model.dicts.get("bodies", []))
    n_markers = len(model.dicts.get("markers", [])) if model.dicts.get("markers") is not None else 0

    fk_marker = model.run["FK_marker"]
    # Check if the model has a contact model

    joint_frames = []
    site_frames = []
    for i in range(n_frames):
        jp_all = fk_marker(state_sequence[i].states, state_sequence[i].constants)
        # Split
        joints = jp_all[:n_bodies, :] if n_bodies > 0 else jp_all
        markers = jp_all[n_bodies:n_bodies + n_markers, :] if n_markers > 0 else None
        joint_frames.append(joints)
        if markers is not None:
            site_frames.append(markers)

    anim_joint_positions = np.asarray(joint_frames)
    anim_site_positions = np.asarray(site_frames) if site_frames else None

    anim_exp_markers = None
    if markers_exp is not None and n_markers > 0:
        me = np.asarray(markers_exp)
        # Clamp length if mismatch
        if me.shape[0] != n_frames:
            nmin = min(me.shape[0], n_frames)
            anim_joint_positions = anim_joint_positions[:nmin]
            if anim_site_positions is not None:
                anim_site_positions = anim_site_positions[:nmin]
            me = me[:nmin]
            n_frames = nmin  # local shadow but we don't reuse outside extraction
        exp_X = me[:, 0:n_markers]
        exp_Y = me[:, n_markers:2 * n_markers] if me.shape[1] >= 2 * n_markers else np.zeros_like(exp_X)
        exp_Z = me[:, 2 * n_markers:3 * n_markers] if me.shape[1] >= 3 * n_markers else np.zeros_like(exp_X)
        anim_exp_markers = np.stack([exp_X, exp_Y, exp_Z], axis=2)

    return {
        "joint0": anim_joint_positions[0],
        "site0": anim_site_positions[0] if anim_site_positions is not None else None,
        "anim_joints": anim_joint_positions,
        "anim_markers": anim_site_positions,
        "anim_exp": anim_exp_markers,
        "n_bodies": n_bodies,
        "n_markers": n_markers,
        "n_frames": anim_joint_positions.shape[0],
    }


def _auto_markers_from_objective(obj):
    """Try to extract experimental markers from a tracking objective.

    Looks through an ObjectiveFunction instance's sub-objectives for a
    `obj_settings` dict containing a `markers_exp` array with shape (N, 3*n_markers).

    Parameters
    ----------
    obj : ObjectiveFunction or None
        The objective manager instance (e.g., `prob.objective`).

    Returns
    -------
    np.ndarray or None
        Experimental markers array if found, else None.
    """
    if obj is None:
        return None
    try:
        subs = getattr(obj, "_objectives", [])
        # Prefer objectives with name indicating marker tracking
        ordered = []
        for sub in subs:
            name = None
            if hasattr(sub, "_get_info"):
                try:
                    info = sub._get_info()
                    name = info.get("name")
                except Exception:
                    name = None
            ordered.append((name or "", sub))
        # Sort so entries containing 'marker' bubble first
        ordered.sort(key=lambda t: ("marker" not in (t[0] or "").lower(), t[0]))
        for _, sub in ordered:
            settings = getattr(sub, "obj_settings", None)
            if isinstance(settings, dict) and ("markers_exp" in settings):
                try:
                    return settings["markers_exp"] 
                except Exception:
                    return None
    except Exception:
        return None
    return None


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


def _compute_limits(case_, non_zero_axes, anim_joints, anim_markers, anim_exp, joint0, site0, exp0, pad_ratio: float = 0.05):
    """Compute axis limits given available position arrays.

    Parameters
    ----------
    case_ : str
        '2D' or '3D'.
    non_zero_axes : list[int]
        Axes retained in 2D mode.
    anim_joints, anim_markers, anim_exp : np.ndarray or None
        Full trajectory arrays (N, items, 3) when animated; may be None.
    joint0, site0, exp0 : np.ndarray or None
        Single-frame arrays for standing case.
    pad_ratio : float, default 0.05
        Fractional padding applied to each axis range. (For standing we may
        pass a larger value from the caller.)
    """
    if anim_joints.shape[0] > 1:  # animation
        pos_list = [anim_joints]
        if anim_markers is not None:
            pos_list.append(anim_markers)
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

        # Build per-axis padding: only X gets padding in standing mode
        d = np.zeros_like(min_)
        # X axis index: first displayed axis in 2D, or 0 in 3D
        x_idx = non_zero_axes[0] if case_ == "2D" else 0
        span_x = max_[x_idx] - min_[x_idx]
        # If degenerate, give a small default span to make limits markerible
        if not np.isfinite(span_x) or span_x == 0:
            span_x = 0.1
        d[x_idx] = np.abs(span_x * pad_ratio)
        min_[x_idx] -= d[x_idx]
        max_[x_idx] += d[x_idx]
        # Y (and Z) unchanged
  
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
    plot_markers,
    plot_expected,
    joint_markersize: float = 6.0,
    site_markersize: float = 4.0,
    expected_markersize: float = 4.0,
    label_markers: bool = False,
):
    """Draw joints, segments, optional markers & expected markers for first frame.

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
    if plot_markers and site0 is not None:
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
            if label_markers and i == 0:
                s.set_label("sim markers")
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
            if label_markers and i == 0:
                g.set_label("exp markers")
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
        for idx, site in enumerate(model.dicts["sites"]):
            parent_name = site["parent"]
            site_idx = len(model.dicts["bodies"]) + idx
            parent_idx = [body["name"] for body in model.dicts["bodies"]].index(
                parent_name
            )
            connections.append((parent_idx, site_idx))

    _walk(model.topology_tree)
    return np.array([c for c in connections])


def _add_segments(ax, case_, non_zero_axes, joint0, site0, connections):
    """Draw initial body/site segments and return their artists and mapped connections.

    Supports connections where the child index may refer to a site index
    (offset by n_bodies). Returns only the connections that could be drawn
    given available data (bodies + optional site0), preserving order.
    """
    segs = []
    drawn_connections = []
    n_bodies = joint0.shape[0]
    for parent_idx, child_idx in connections:
        # Resolve endpoints: parent is always a body
        if parent_idx < 0 or parent_idx >= n_bodies:
            continue
        a = joint0[parent_idx]
        # Child may be a body or a site
        if child_idx < n_bodies:
            b = joint0[child_idx]
        else:
            if site0 is None:
                continue
            sidx = child_idx - n_bodies
            if sidx < 0 or sidx >= site0.shape[0]:
                continue
            b = site0[sidx]
        if case_ == "2D":
            (l,) = ax.plot([a[non_zero_axes[0]], b[non_zero_axes[0]]], [a[non_zero_axes[1]], b[non_zero_axes[1]]], c="k")
        else:
            (l,) = ax.plot([a[0], b[0]], [a[1], b[1]], c="k")
            l.set_3d_properties([a[2], b[2]])
        segs.append(l)
        drawn_connections.append((parent_idx, child_idx))
    return segs, np.array(drawn_connections)


def _create_update_func(anim_joints, anim_markers, anim_exp, joints_art, site_artists, exp_artists, segments, connections, case_, non_zero_axes, dt, ax, model, hascontact, contact_plot_objects):
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
        n_bodies_local = anim_joints.shape[1]
        for i, (pidx, cidx) in enumerate(connections):
            # Resolve endpoints (parent is a body, child may be body or site)
            a = anim_joints[f][pidx]
            if cidx < n_bodies_local:
                b = anim_joints[f][cidx]
            else:
                if anim_markers is None:
                    continue
                sidx = cidx - n_bodies_local
                if sidx < 0 or sidx >= anim_markers.shape[1]:
                    continue
                b = anim_markers[f][sidx]
            if case_ == "2D":
                segments[i].set_data([[a[non_zero_axes[0]], b[non_zero_axes[0]]], [a[non_zero_axes[1]], b[non_zero_axes[1]]]])
            else:
                segments[i].set_data([a[0], b[0]], [a[1], b[1]])
                segments[i].set_3d_properties([a[2], b[2]])
        # markers (sim)
        if anim_markers is not None and len(site_artists) == anim_markers.shape[1]:
            for i in range(anim_markers.shape[1]):
                if case_ == "2D":
                    site_artists[i].set_data([[anim_markers[f][i, non_zero_axes[0]]], [anim_markers[f][i, non_zero_axes[1]]]])
                else:
                    site_artists[i].set_data([[anim_markers[f][i, 0]]], [[anim_markers[f][i, 1]]])
                    site_artists[i].set_3d_properties([anim_markers[f][i, 2]])
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
    plot_markers=True,
    plot_expected=True,
    joint_markersize: float = 6.0,
    site_markersize: float = 4.0,
    expected_markersize: float = 4.0,
    **kwargs,
):
    """Plot (or animate) a stick-figure of the model with optional markers & markers.

    This refactored implementation splits the original monolithic logic into
    modular helpers to clarify the distinct phases:
    1) FK data extraction (bodies, markers, expected markers)
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
    plot_markers : bool, default True
        Whether to display simulated site positions (red).
    plot_expected : bool, default True
        Whether to display experimental expected markers (green) when
        `markers_exp` kwarg is provided.
    markers_exp : array-like, optional (via **kwargs)
        Experimental markers shaped (N, 3*n_markers). If not provided, this
        function will attempt to auto-discover them from a tracking markers
        objective if you pass either `objective=prob.objective` or
        `problem=prob`/`prob=prob` in **kwargs.

    Notes
    -----
    - Standing (n_nodes == 1): still plots expected markers if provided.
    - Walking (n_nodes > 1): creates an interactive animation with pause &
      speed controls (space / up / down keys).
    - markers are not states; they are derived via FK_marker appended rows.
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

    fk_marker = model.run["FK_marker"]
    # Check if the model has a contact model
    hascontact = hasattr(model, "gc_model") and model.gc_model is not None
    
    # Check if the model has an actuator model (muscles)
    hasmuscles = hasattr(model, "actuator_model") and model.actuator_model is not None and hasattr(model.actuator_model, "plot")

    # Auto-discover experimental markers from objectives if not provided
    markers_exp = kwargs.get("markers_exp", None)
    if markers_exp is None and plot_expected:
        # 1) If the model carries an attached objective (set by ObjectiveFunction), use it.
        objective = getattr(model, "_biosym_objective", None)
        # 2) Otherwise, try any provided problem handle fallback
        if objective is None:
            prob = kwargs.get("problem") or kwargs.get("prob")
            if prob is not None and hasattr(prob, "objective"):
                objective = prob.objective
        markers_exp = _auto_markers_from_objective(objective)
    data = _extract_fk_marker_data(model, state_sequence, markers_exp if plot_expected else None)
    joint_positions = data["joint0"]
    site_positions = data["site0"] if plot_markers else None
    anim_joint_positions = data["anim_joints"]
    anim_site_positions = data["anim_markers"] if plot_markers else None
    anim_exp_markers = data["anim_exp"] if plot_expected else None
    n_markers = data["n_markers"]
    n_bodies = data["n_bodies"]
    n_frames = data["n_frames"]

    fig, ax, case_, non_zero_axes = _setup_axes(joint_positions, site_positions, None, n_frames)

    # Prepare expected markers first frame (standing) for limit computation
    exp0 = None
    if anim_exp_markers is not None:
        exp0 = anim_exp_markers[0]

    # Compute and set limits
    # Allow user override of padding; enlarge default for standing (single frame)
    user_pad = kwargs.get("pad_ratio", 1 if n_frames == 1 else 0.05)
    limits = _compute_limits(case_, non_zero_axes, anim_joint_positions, anim_site_positions, anim_exp_markers,
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
        plot_markers,
        plot_expected,
        joint_markersize=joint_markersize,
        site_markersize=site_markersize,
        expected_markersize=expected_markersize,
        label_markers=(n_frames > 1),
    )
    # Plot all connections
    segments, connections = _add_segments(ax, case_, non_zero_axes, joint_positions, site_positions, connections)

    plot_objects = None
    if hascontact:
        # Initialize and retain plot objects for contact model so update phase can reuse them.
        contact_plot_objects = model.gc_model.plot(
            state_sequence, model, mode="init", ax=ax, case=case_, non_zero_axes=non_zero_axes
        )
    else:
        contact_plot_objects = None
    
    if hasmuscles:
        muscle_plot_objects = model.actuator_model.plot(
            state_sequence, model, mode="init", ax=ax, case=case_, non_zero_axes=non_zero_axes
        )
    else:
        muscle_plot_objects = None

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

    update = _create_update_func(
        anim_joint_positions,
        anim_site_positions,
        anim_exp_markers,
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
    # Only add legend for multi-frame animations to avoid clutter in standing
    if n_frames > 1:
        handles, labels = ax.get_legend_handles_labels()
        ispaused, speed_multiplier = False, 1.0
        if labels:
            ax.legend(loc="best")



        global pauseframes_total
        pauseframes_total = 0
        current_frame = 0
        last_update_time = 0

        def update(frame_input):
            nonlocal current_frame, last_update_time
            global pauseframes_total

            import time

            current_time = time.time()

            if ispaused:
                last_update_time = current_time
                return []  # No update if paused

            # Calculate frame based on speed multiplier
            if last_update_time == 0:
                last_update_time = current_time

            time_elapsed = current_time - last_update_time
            frames_to_advance = int(time_elapsed * speed_multiplier / dt)

            if frames_to_advance >= 1:
                current_frame = (current_frame + frames_to_advance) % n_frames
                last_update_time = current_time

            frame = current_frame

            for i, joint in enumerate(model.positions):
                if case_ == "2D":
                    joints[i].set_data(
                        [
                            [anim_joint_positions[frame][i, non_zero_axes[0]]],
                            [anim_joint_positions[frame][i, non_zero_axes[1]]],
                        ]
                    )
                else:
                    joints[i].set_data(
                        [
                            [anim_joint_positions[frame][i, 0]],
                            [anim_joint_positions[frame][i, 1]],
                        ]
                    )
                    joints[i].set_3d_properties(anim_joint_positions[frame][i, 2])
            for i, connection in enumerate(connections):
                if case_ == "2D":
                    segments[i].set_data(
                        [
                            [
                                anim_joint_positions[frame][connection][
                                    :, non_zero_axes[0]
                                ]
                            ],
                            [
                                anim_joint_positions[frame][connection][
                                    :, non_zero_axes[1]
                                ]
                            ],
                        ]
                    )
                else:
                    pos_a = anim_joint_positions[frame][connection][0]
                    pos_b = anim_joint_positions[frame][connection][1]
                    segments[i].set_data([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]])
                    segments[i].set_3d_properties([pos_a[2], pos_b[2]])
            ax.set_title(f"T = {frame * dt:.2f} s")
            if hascontact:
                model.gc_model.plot(
                    states,
                    model,
                    mode="update",
                    ax=ax,
                    case=case_,
                    non_zero_axes=non_zero_axes,
                    frame=frame,
                    plot_objects=contact_plot_objects,
                )
            
            if hasmuscles:
                model.actuator_model.plot(
                    states,
                    model,
                    mode="update",
                    ax=ax,
                    case=case_,
                    non_zero_axes=non_zero_axes,
                    frame=frame,
                    plot_objects=muscle_plot_objects,
                )

                return ax.collections

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=np.arange(n_frames),
            interval=50,
            blit=False,  # 50ms = 20 FPS for smooth animation
        )
        plt.show()
        return ani
