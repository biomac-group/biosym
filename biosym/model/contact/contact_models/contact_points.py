import jax
import jax.numpy as jnp
import numpy as np
from sympy import Matrix, lambdify
from sympy.physics.mechanics import Point

from biosym.model.contact.base_contact import BaseContact


class ContactPoints(BaseContact):
    """
    Point-based contact model for ground reaction force calculation.
    
    This class implements a contact model based on discrete contact points
    attached to specified bodies. Each contact point can generate normal
    and friction forces when in contact with the ground, using a spring-damper
    model with Coulomb friction.
    
    Parameters
    ----------
    xml_root : xml.etree.ElementTree.Element
        Root element of the XML tree containing contact point definitions.
        Should contain 'contact_point' elements with position and parameter specs.
    body_weight : float, default=1
        Body weight in arbitrary units for scaling contact parameters.
        Contact stiffness and damping are scaled by body_weight * 9.81.
        
    Attributes
    ----------
    cps : dict
        Dictionary mapping contact point names to their properties.
    bodies : list of str
        List of body names that have contact points.
    body_mapping : numpy.ndarray
        Array mapping contact points to their parent bodies.
    k : list of float
        Contact stiffness values for each contact point.
    b : list of float  
        Contact damping values for each contact point.
    mu : list of float
        Friction coefficients for each contact point.
    p_cy_0 : list of float
        Transition region sizes for position (penetration depth).
    v_cx_0 : list of float
        Transition region sizes for velocity (sliding).
        
    Notes
    -----
    The contact model uses:
    - Hunt-Crossley contact mechanics for normal forces
    - Coulomb friction with smooth transitions
    - Penetration-based contact detection
    - Body-weight scaling for force parameters
    
    Contact forces are calculated as:
    - Normal force: F_n = k * penetration * (1 + b * penetration_velocity)
    - Friction force: F_f = mu * F_n * tanh(velocity / v_cx_0)
    
    XML Format
    ----------
    Expected XML structure:
    
    .. code-block:: xml
    
        <contact type="contact_points">
            <default>
                <contact_point k="1000" b="10" mu="0.8"/>
            </default>
            <contact_point name="heel_r" body="foot_r" pos="0 0 -0.05"/>
            <contact_point name="toe_r" body="foot_r" pos="0.15 0 -0.05"/>
        </contact>
        
    Examples
    --------
    Create contact model from XML:
    
    >>> import xml.etree.ElementTree as ET
    >>> root = ET.parse("contact.xml").getroot()
    >>> contact = ContactPoints(root, body_weight=70.0)
    
    Get contact information:
    
    >>> bodies = contact.get_bodies()
    >>> n_points = len(contact.cps)
    >>> forces = contact.forward(states, constants, model)
    
    See Also
    --------
    biosym.model.contact.base_contact.BaseContact : Base contact interface
    """
    
    def __init__(self, xml_root, body_weight=1):
        """
        Initialize the ContactPoints model from XML definition.
        
        Parses XML contact point definitions and sets up contact parameters
        including positions, stiffness, damping, and friction coefficients.
        
        Parameters
        ----------
        xml_root : xml.etree.ElementTree.Element
            Root element containing contact point definitions.
        body_weight : float, default=1
            Body weight for scaling contact parameters.
            
        Raises
        ------
        ValueError
            If any contact point is missing a required name attribute.
        """
        super().__init__(xml_root)
        # Get default values
        self.cp_defaults = (
            xml_root.find("default/contact_point").attrib if xml_root.find("default/contact_point") is not None else {}
        )
        cps = {}
        for cp in xml_root.findall("contact_point"):
            cp_name = cp.get("name")
            if cp_name is None:
                raise ValueError("Contact point must have a name")
            cps[cp_name] = {}
            for key, value in self.cp_defaults.items():
                if key == "pos":
                    cps[cp_name][key] = np.array([float(x) for x in cp.get(key, value).split()])
                else:
                    cps[cp_name][key] = cp.get(key, value)
        self.bodies = [cps[cp]["body"] for cp in cps]
        _, self.body_mapping = np.unique(self.bodies, return_inverse=True)
        self.body_mapping = np.tile(self.body_mapping, (3, 1))
        self.cps = cps

        # For every contact body, make a connection to each of its contact points

        # Get the contact point parameters
        self.k = [float(cps[cp]["k"]) for cp in cps]
        self.b = [float(cps[cp]["b"]) for cp in cps]
        self.mu = [float(cps[cp]["mu"]) for cp in cps]
        self.p_cy_0 = [1e-3] * len(self.k)  # Transition region size for position
        self.v_cx_0 = [1e-2] * len(self.k)  # Transition region size for velocity

    def process_eom(self, model, **kwargs):
        """
        Build the eom for the contact model with symbolic variables.
        """
        self.body_weight = kwargs.get("body_weight", 1)
        self.k = [k * self.body_weight * 9.81 for k in self.k]
        pos_vector, vel_vector = [], []
        force_vector = []
        for i, cp in enumerate(self.cps):
            # Create a sympy point for the contact point
            cp_ = self.cps[cp]
            ref_frame = model.reference_frames[cp_["body"]]
            origin = model.body_origins[cp_["body"]]

            cp = Point(cp_["name"])
            cp.set_pos(
                origin,
                ref_frame.x * cp_["pos"][0] + ref_frame.y * cp_["pos"][1] + ref_frame.z * cp_["pos"][2],
            )

            pos_vector.append(
                [
                    cp.pos_from(model.origin).dot(frame_dim)
                    for frame_dim in [
                        model.ground_frame.x,
                        model.ground_frame.y,
                        model.ground_frame.z,
                    ]
                ]
            )
            vel_vector.append(
                [
                    cp.vel(model.ground_frame).dot(frame_dim)
                    for frame_dim in [
                        model.ground_frame.x,
                        model.ground_frame.y,
                        model.ground_frame.z,
                    ]
                ]
            )
            pos_vector[-1] = model._replace_dyn(Matrix(pos_vector[-1])).T
            vel_vector[-1] = model._replace_dyn(Matrix(vel_vector[-1]))

            d = 0.5 * ((pos_vector[-1][1] ** 2 + self.p_cy_0[i] ** 2) ** 0.5 - pos_vector[-1][1])
            F_cy = (
                self.k[i] * d * (1 - self.b[i] * vel_vector[-1][1]) - pos_vector[-1][1]
            )  # small value to "point towards ground": -1 N/m 
            F_cx = -self.mu[i] * F_cy * vel_vector[-1][0] / (vel_vector[-1][0] ** 2 + self.v_cx_0[i] ** 2) ** 0.5
            F_cz = -self.mu[i] * F_cy * vel_vector[-1][2] / (vel_vector[-1][2] ** 2 + self.v_cx_0[i] ** 2) ** 0.5
            # Get F and M in the global frame
            force_vector.append([F_cx, F_cy, F_cz])
        force_vector = Matrix(force_vector)
        self.force_vector = lambdify(model._v, force_vector, modules="jax", cse=True, docstring_limit=2)
        pos_vector = Matrix(pos_vector)
        self.pos_vector = lambdify(model._v, pos_vector, modules="jax", cse=True, docstring_limit=2)

    def get_n_states(self):
        return 0

    def get_n_constants(self):
        # All constants are hardcoded in the process_eom sympy functions - can be changed in the future
        return 0

    def get_states(self):
        return []

    def get_constants(self):
        return []

    def get_bodies(self):
        """
        Returns the list of bodies that can be in contact.
        """
        return np.unique(self.bodies)

    def get_n_bodies(self):
        """
        Returns the number of bodies that can be in contact.
        """
        return len(np.unique(self.bodies))

    def get_cp_forces(self, states, constants, model):
        """
        Returns the contact forces for all bodies in the global frame.
        inputs:
            - states: The state of the model
            - model: The model object
        outputs:
            - contact_forces: The contact forces for all contact points in the global frame
        """
        cp_forces = self.force_vector(*states.model, *constants.model)
        return cp_forces

    def get_cp_moment_arms(self, states, constants, model, return_positions=False):
        """
        Returns the moment arms for every contact point wrt to the body origin.
        """
        body_idx = np.array([list(model.rigid_bodies.keys()).index(p) for p in self.bodies])
        pos_bodies = model.run["FK"](states, constants)[body_idx]
        pos_cps = self.pos_vector(*states.model, *constants.model)
        if return_positions:
            return pos_cps, pos_bodies, body_idx
        return pos_cps - pos_bodies

    def forward(self, states, constants, model):
        """
        Returns the contact forces for all bodies in the global frame.
        inputs:
            - states: The state of the model
            - model: The model object
        outputs:
            - contact_forces: The contact forces for all bodies in the global frame
        """
        moment_arms = self.get_cp_moment_arms(states, constants, model)
        cp_forces = self.get_cp_forces(states, constants, model)

        # Get F and M in the global frame
        length = self.get_n_bodies()

        # Bincount is only 1D, therefore vmap it to 2D (note: vmap the whole model --> gpu version)
        def _bincount(arr, weights):
            return jnp.bincount(arr, weights=weights, length=length)

        foot_forces = jax.vmap(_bincount)(self.body_mapping, cp_forces.T).T

        moment_cps = jnp.cross(moment_arms, cp_forces)
        foot_moments = jax.vmap(_bincount)(self.body_mapping, moment_cps.T).T
        return foot_forces, foot_moments

    def reset(self):
        # No hidden states to reset
        pass

    def plot(self, states, model, mode, ax, **kwargs):
        """
        Plots the contact points in the model.
        For the contact points, a connection to its body is made.
        For each contact point, a line is drawn to each other contact point on the same body.
        inputs:
            - states: The state of the model
            - model: The model object
            - mode: The mode of the plot "init" or "update"
        """
        if "case" in kwargs:
            case = kwargs["case"]
            if case not in ["2D", "3D"]:
                raise ValueError("Invalid case. Must be '2D' or '3D'.")
            if case == "2D":
                if "non_zero_axes" in kwargs:
                    non_zero_axes = kwargs["non_zero_axes"]
                else:
                    raise ValueError("2D case requires non_zero_axes as an input argument to the foot contact model.")
        else:
            case = "3D"

        self.plot_markers = []
        self.body_lines = []
        self.cp_lines = []
        self.force_lines = []

        if mode == "init":
            ## Get a list of all pos_cps and pos_bodies
            pos_cps = []
            pos_bodies = []
            cp_forces = []
            if isinstance(states, list):
                for i in range(len(states)):
                    pcp, pbody, body_idx = self.get_cp_moment_arms(
                        states[i].states,
                        states[i].constants,
                        model,
                        return_positions=True,
                    )
                    pos_cps.append(pcp)
                    pos_bodies.append(pbody)
                    cp_forces.append(self.get_cp_forces(states[i].states, states[i].constants, model))
                    # f = self.forward(states[i]['states'], states[i]['constants'], model)
                    # print("Forces: ", f[0], "moments:" ,f[1])
            elif len(states.states.model.shape) == 1:
                pcp, pbody, body_idx = self.get_cp_moment_arms(
                    states.states, states.constants, model, return_positions=True
                )
                pos_cps.append(pcp)
                pos_bodies.append(pbody)
                cp_forces.append(self.get_cp_forces(states[0].states, states[0].constants, model))
            else:
                for i in range(len(states)):
                    pcp, pbody, body_idx = self.get_cp_moment_arms(
                        states[i].states,
                        states[i].constants,
                        model,
                        return_positions=True,
                    )
                    pos_cps.append(pcp)
                    pos_bodies.append(pbody)
                    cp_forces.append(self.get_cp_forces(states[i].states, states[i].constants, model))
            self.pos_cps = np.array(pos_cps)
            self.pos_bodies = np.array(pos_bodies)
            self.cp_forces = np.array(cp_forces)

            for i in range(len(self.pos_cps[0])):
                if case == "2D":
                    (l,) = ax.plot(
                        self.pos_cps[0][i, non_zero_axes[0]],
                        self.pos_cps[0][i, non_zero_axes[1]],
                        c="k",
                        marker="o",
                    )
                else:
                    (l,) = ax.plot(
                        self.pos_cps[0][i, 0],
                        self.pos_cps[0][i, 1],
                        self.pos_cps[0][i, 2],
                        c="k",
                        marker="o",
                    )
                self.plot_markers.append(l)
                if case == "2D":
                    (l,) = ax.plot(
                        [
                            self.pos_bodies[0][i, non_zero_axes[0]],
                            self.pos_cps[0][i, non_zero_axes[0]],
                        ],
                        [
                            self.pos_bodies[0][i, non_zero_axes[1]],
                            self.pos_cps[0][i, non_zero_axes[1]],
                        ],
                        c="k",
                    )
                else:
                    (l,) = ax.plot(
                        [self.pos_bodies[0][i, 0], self.pos_cps[0][i, 0]],
                        [self.pos_bodies[0][i, 1], self.pos_cps[0][i, 1]],
                        [self.pos_bodies[0][i, 2], self.pos_cps[0][i, 2]],
                        c="k",
                    )
                self.body_lines.append(l)

                # Create a line to each other contact point on the same body
                for j in range(i, len(self.pos_cps[0])):
                    if body_idx[i] == body_idx[j]:
                        if case == "2D":
                            (l,) = ax.plot(
                                [
                                    self.pos_cps[0][i, non_zero_axes[0]],
                                    self.pos_cps[0][j, non_zero_axes[0]],
                                ],
                                [
                                    self.pos_cps[0][i, non_zero_axes[1]],
                                    self.pos_cps[0][j, non_zero_axes[1]],
                                ],
                                c="k",
                            )
                        else:
                            (l,) = ax.plot(
                                [self.pos_cps[0][i, 0], self.pos_cps[0][j, 0]],
                                [self.pos_cps[0][i, 1], self.pos_cps[0][j, 1]],
                                [self.pos_cps[0][i, 2], self.pos_cps[0][j, 2]],
                                c="k",
                            )
                        self.cp_lines.append(l)

                # Create a line for the contact force
                factor = 1 / 9.81 / self.body_weight
                self.factor = factor
                if case == "2D":
                    (l,) = ax.plot(
                        [
                            self.pos_cps[0][i, non_zero_axes[0]],
                            self.pos_cps[0][i, non_zero_axes[0]] + factor * cp_forces[0][i, non_zero_axes[0]],
                        ],
                        [
                            self.pos_cps[0][i, non_zero_axes[1]],
                            self.pos_cps[0][i, non_zero_axes[1]] + factor * cp_forces[0][i, non_zero_axes[1]],
                        ],
                        c="darkgreen",
                    )
                else:
                    (l,) = ax.plot(
                        [
                            self.pos_cps[0][i, 0],
                            self.pos_cps[0][i, 0] + factor * cp_forces[0][i, 0],
                        ],
                        [
                            self.pos_cps[0][i, 1],
                            self.pos_cps[0][i, 1] + factor * cp_forces[0][i, 1],
                        ],
                        [
                            self.pos_cps[0][i, 2],
                            self.pos_cps[0][i, 2] + factor * cp_forces[0][i, 2],
                        ],
                        c="darkgreen",
                    )
                self.force_lines.append(l)

            # Plot the floor as a grey box with alpha ...
            if case == "2D":
                ax.fill_between([-5, 5], -10, 0, color="grey", alpha=0.5)
            else:
                x = np.linspace(-5, 5, 10)
                y = np.linspace(-5, 5, 10)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros(X.shape)
                ax.plot_surface(X, Y, Z, color="grey", alpha=0.5)
            self.body_idx = body_idx
            return self.plot_markers, self.body_lines, self.cp_lines, self.force_lines

        if mode == "update":
            frame = kwargs.get("frame")
            plot_markers, body_lines, cp_lines, force_lines = kwargs.get("plot_objects")
            for i, joint in enumerate(plot_markers):
                if case == "2D":
                    joint.set_data(
                        [
                            [self.pos_cps[frame][i, non_zero_axes[0]]],
                            [self.pos_cps[frame][i, non_zero_axes[1]]],
                        ]
                    )
                else:
                    joint.set_data([[self.pos_cps[frame][i, 0]], [self.pos_cps[frame][i, 1]]])
                    joint.set_3d_properties(self.pos_cps[frame][i, 2])
            for i, line in enumerate(body_lines):
                if case == "2D":
                    line.set_data(
                        [
                            [self.pos_bodies[frame][i, non_zero_axes[0]]],
                            [self.pos_cps[frame][i, non_zero_axes[0]]],
                        ],
                        [
                            [self.pos_bodies[frame][i, non_zero_axes[1]]],
                            [self.pos_cps[frame][i, non_zero_axes[1]]],
                        ],
                    )
                else:
                    pos_a = self.pos_bodies[frame][i]
                    pos_b = self.pos_cps[frame][i]
                    line.set_data([[pos_a[0]], [pos_b[0]]], [[pos_a[1]], [pos_b[1]]])
                    line.set_3d_properties([[pos_a[2]], [pos_b[2]]])
            id = 0
            for i in range(len(self.pos_cps[frame])):
                for j in range(i, len(self.pos_cps[frame])):
                    if self.body_idx[i] == self.body_idx[j]:
                        line = cp_lines[id]
                        id += 1
                        if case == "2D":
                            line.set_data(
                                [
                                    [self.pos_cps[frame][i, non_zero_axes[0]]],
                                    [self.pos_cps[frame][j, non_zero_axes[0]]],
                                ],
                                [
                                    [self.pos_cps[frame][i, non_zero_axes[1]]],
                                    [self.pos_cps[frame][j, non_zero_axes[1]]],
                                ],
                            )
                        else:
                            pos_a = self.pos_cps[frame][i]
                            pos_b = self.pos_cps[frame][j]
                            line.set_data([[pos_a[0]], [pos_b[0]]], [[pos_a[1]], [pos_b[1]]])
                            line.set_3d_properties([[pos_a[2]], [pos_b[2]]])

            for i, line in enumerate(force_lines):
                if case == "2D":
                    line.set_data(
                        [
                            [self.pos_cps[frame][i, non_zero_axes[0]]],
                            [
                                self.pos_cps[frame][i, non_zero_axes[0]]
                                + self.factor * self.cp_forces[frame][i, non_zero_axes[0]]
                            ],
                        ],
                        [
                            [self.pos_cps[frame][i, non_zero_axes[1]]],
                            [
                                self.pos_cps[frame][i, non_zero_axes[1]]
                                + self.factor * self.cp_forces[frame][i, non_zero_axes[1]]
                            ],
                        ],
                    )
                else:
                    pos_a = self.pos_cps[frame][i]
                    pos_b = self.pos_cps[frame][i] + self.factor * self.cp_forces[frame][i]
                    line.set_data([[pos_a[0]], [pos_b[0]]], [[pos_a[1]], [pos_b[1]]])
                    line.set_3d_properties([[pos_a[2]], [pos_b[2]]])

        else:
            raise ValueError("Invalid mode. Must be 'init' or 'update'.")
