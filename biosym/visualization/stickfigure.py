import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_stick_figure(model, states, dt=0.01, frame=None, **kwargs):
    """
    Plot a stick figure of the model.

    Depending on the use-case, states may be:
        - A StatesDict instance, which is used for collocation and physics-informed learning.
    Different types of states are used for different purposes:
        - For forward simulation, the states are usually a list of state dictionaries
        - For collocation & physics-informed learning, the states are usually a dictionary where each entry is a 2D array of shape (t, n_states/n_constants)

    Parameters
    ----------
        model: The model to be plotted.
        states: The state vector of the model.
        dt: The time step for the simulation.
        frame: The frame to be plotted. If None, all frames will be plotted in a video (if there is more than one frame).
    """
    # Check inputs and frame selection
    states, globals = states
    if globals is not None:
        dt = globals.dur/len(states)
    n_frames = len(states)

    # Check if the model has a contact model
    hascontact = hasattr(model, "gc_model") and model.gc_model is not None

    # First node joint positions
    if isinstance(states, list):
        joint_positions = model.run["FK_vis"](
            states[0].states, states[0].constants
        )
    elif len(states.states.model.shape) == 1:
        joint_positions = model.run["FK_vis"](states[0].states, states[0].constants)
    else:
        joint_positions = model.run["FK_vis"](
            states[0].states, states[0].constants
        )
    # Following frames joint positions
    anim_joint_positions = []
    if n_frames > 1:
        for i in range(n_frames):
            if isinstance(states, list):
                joint_positions = model.run["FK_vis"](
                    states[i].states, states[i].constants
                )
            else:
                joint_positions = model.run["FK_vis"](
                    states[i].states, states[i].constants
                )
            anim_joint_positions.append(joint_positions)
        anim_joint_positions = np.array(anim_joint_positions)

    # Check if the scene is 2D or 3D, it is 2D if all coordinates in 1 dimension are 0
    if np.any(np.all(joint_positions == 0, axis=0)):  # 2D case
        # Setup 2d figure
        fig = (
            plt.figure()
        )  # Todo: allow passing an axis to plot to, so we can plot multiple figures in one window
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        case_ = "2D"
        non_zero_axes = np.where(np.any(joint_positions != 0, axis=0))[0]
        if len(non_zero_axes) == 1:
            if non_zero_axes in [0, 1]:
                non_zero_axes = [0, 1]  # There might be room for improvement here.
            else:
                non_zero_axes = [2, 1]  # y-up convention
    else:  # 3D case
        # Setup 3d figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        case_ = "3D"
        non_zero_axes = [0, 1, 2]  # All axes are non-zero

    # Set axis limits for video
    if n_frames > 1:
        # find limits of the plot
        min_ = np.min(anim_joint_positions, axis=(0, 1))
        max_ = np.max(anim_joint_positions, axis=(0, 1))
        d = np.abs((max_ - min_) * 0.05)  # 5% padding
        min_ -= d
        max_ += d
        # Set limits of the plot
        if case_ == "2D":
            ax.set_xlim(min_[non_zero_axes[0]], max_[non_zero_axes[0]])
            ax.set_ylim(min_[non_zero_axes[1]], max_[non_zero_axes[1]])
        else:
            ax.set_xlim(min_[0], max_[0])
            ax.set_ylim(min_[1], max_[1])
            ax.set_zlim(min_[2], max_[2])

    else: # Single frame, set limits to the joint positions
        min_ = np.min(joint_positions)
        max_ = np.max(joint_positions)
        d = np.abs((max_ - min_) * 0.05)  # 5% padding
        min_ -= d 
        max_ += d
        # Set limits of the plot
        if case_ == "2D":
            ax.set_xlim(min_, max_)
            ax.set_ylim(min_, max_)
        else:
            ax.set_xlim(min_, max_)
            ax.set_ylim(min_, max_)
            ax.set_zlim(min_, max_)

    # @todo: Make this function jittable, if it is slow for larger models
    connections = []

    def _get_child_connections(topology, model):
        """
        Recursively get all child connections from the topology tree.
        """
        for idx, node in enumerate(topology):
            body_name = node["name"]
            body_idx = [body["name"] for body in model.dicts["bodies"]].index(body_name)
            children = node["children"]
            for child in children:
                child_name = child["name"]
                child_idx = [body["name"] for body in model.dicts["bodies"]].index(
                    child_name
                )
                connections.append((body_idx, child_idx))
            _get_child_connections(children, model)
        for idx, site in enumerate(model.dicts["sites"]):
            parent_name = site["parent"]
            site_idx = len(model.dicts["bodies"]) + idx
            parent_idx = [body["name"] for body in model.dicts["bodies"]].index(
                parent_name
            )
            connections.append((parent_idx, site_idx))

    _get_child_connections(model.topology_tree, model)
    connections = np.array(connections)

    # Plot all joint centers for the first frame
    joints = []
    segments = []
    for i, joint in enumerate(model.positions):
        if case_ == "2D":
            (l,) = ax.plot(
                joint_positions[i, non_zero_axes[0]],
                joint_positions[i, non_zero_axes[1]],
                c="k",
                marker="o",
            )
        else:
            (l,) = ax.plot(
                joint_positions[i, 0],
                joint_positions[i, 1],
                joint_positions[i, 2],
                c="k",
                marker="o",
            )
        joints.append(l)
    # Plot all connections
    for connection in connections:
        if case_ == "2D":
            (l,) = ax.plot(
                joint_positions[connection][:, non_zero_axes[0]],
                joint_positions[connection][:, non_zero_axes[1]],
                c="k",
            )
        else:
            (l,) = ax.plot(
                joint_positions[connection][:, 0],
                joint_positions[connection][:, 1],
                joint_positions[connection][:, 2],
                c="k",
            )
        segments.append(l)

    if hascontact:

        plot_objects = model.gc_model.plot(
            states, model, mode="init", ax=ax, case=case_, non_zero_axes=non_zero_axes
        )

    # Set axis labels
    ax.set_xlabel("X-axis [m]")
    ax.set_ylabel("Y-axis [m]")
    if case_ == "3D":
        ax.set_zlabel("Z-axis [m]")
        ax.set_title("Stick Figure (3D)")
        ax.view_init(elev=20, azim=30)

    if n_frames == 1:
        ax.set_title("Stick Figure")
        plt.tight_layout()
        plt.show()
    else:  # Video display, all following frames are animated
        ispaused = False
        speed_multiplier = 1.0
        
        # Add speed text display at the top of the figure
        speed_text = fig.text(0.5, 0.98, f"Speed: {speed_multiplier:.1f}x | Controls: Space=Pause, ↑↓=Speed", 
                             ha='center', va='top', fontsize=10, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        def on_key_press(event):
            nonlocal ispaused, speed_multiplier
            if event.key == " ":  # Spacebar toggles pause
                ispaused = not ispaused
            elif event.key == "up":  # Up arrow increases speed
                speed_multiplier = min(speed_multiplier * 1.2, 10.0)  # Max 10x speed
                speed_text.set_text(f"Speed: {speed_multiplier:.1f}x | Controls: Space=Pause, ↑↓=Speed")
                fig.canvas.draw_idle()
            elif event.key == "down":  # Down arrow decreases speed
                speed_multiplier = max(speed_multiplier / 1.2, 0.1)  # Min 0.1x speed
                speed_text.set_text(f"Speed: {speed_multiplier:.1f}x | Controls: Space=Pause, ↑↓=Speed")
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key_press)

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
            ax.set_title(f"T = {frame*dt:.2f} s")
            if hascontact:
                model.gc_model.plot(
                    False,
                    model,
                    mode="update",
                    ax=ax,
                    case=case_,
                    non_zero_axes=non_zero_axes,
                    frame=frame,
                    plot_objects=plot_objects,
                )

                return ax.collections

        ani = animation.FuncAnimation(
            fig, update, frames=np.arange(n_frames), 
            interval=50, blit=False  # 50ms = 20 FPS for smooth animation
        )
        plt.show()
