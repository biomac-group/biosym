import os

_cachedir = os.path.expanduser("~/.biosym/jax_cache")
_model_cache = os.path.expanduser("~/.biosym/")
os.environ["JAX_COMPILATION_CACHE_DIR"] = (
    _cachedir  # This needs to happen before importing jax
)
# os.environ["jax_persistent_cache_min_compile_time_secs".upper()] = "10"

os.makedirs((_cachedir), exist_ok=True)

import hashlib

import cloudpickle
import jax
import numpy as np
import pandas as pd
import yaml
from sympy import Matrix, lambdify, symbols
from sympy.physics.mechanics import (
    Inertia,
    KanesMethod,
    Point,
    ReferenceFrame,
    RigidBody,
    dynamicsymbols,
)

from biosym.model.actuators import *
from biosym.model.contact import *
from biosym.model.parsers import *


class BiosymModel:
    """
    Biomechanical models are defined in this class.
    It will contain all functionality to load, save, and manipulate the model.
    Docstrings need to be added.
    """

    def __init__(self, definition_file, get_hash=False):
        # I think that it makes sense to force .yaml files at some point, because there are settings that are not represented in the model files
        # .yaml files can define additional variables, for mujoco that would be ground contact
        cfg = None
        if definition_file.endswith(".yaml"):
            # Load the yaml file and check for the model name
            with open(definition_file) as f:
                cfg = yaml.safe_load(f)
            # Replace the definition file with the model name
            definition_file = os.path.join(
                os.path.dirname(definition_file), cfg["model"]["name"]
            )

        if definition_file.endswith(".xml"):
            parser = mujoco_parser.MujocoParser(definition_file)
            if parser.has_actuators():
                self.actuators = actuator_parser.get(parser.get_actuators())
                parser.actuators = self.actuators.get_actuators()
            if parser.has_contact_model():
                self.gc_model = contact_parser.get(parser.get_contact_model())
                parser.external_forces_bodies = self.gc_model.get_bodies()
        elif definition_file.endswith(".osim"):
            raise NotImplementedError("OSIM models not supported yet.")
        else:
            raise ValueError(
                "Model definition file must be in .xml, .osim, or .yaml format."
            )

        # Check for additional parameters
        if cfg is not None:
            if "additional_parameters" in cfg["model"]:
                if "ground_contact" in cfg["model"]["additional_parameters"]:
                    gc_model_file = os.path.join(
                        os.path.dirname(definition_file),
                        cfg["model"]["additional_parameters"]["ground_contact"]["file"],
                    )
                    self.gc_model = contact_parser.get(gc_model_file)
                    if (
                        cfg["model"]["additional_parameters"]["ground_contact"][
                            "replace_existing"
                        ]
                        == True
                    ):
                        parser.external_forces_bodies = self.gc_model.get_bodies()
                    else:
                        raise NotImplementedError(
                            "Adding ground contact to existing models is not implemented yet. Try replacing them instead."
                        )
                if "actuators" in cfg["model"]["additional_parameters"]:
                    actuator_model_file = os.path.join(
                        os.path.dirname(definition_file),
                        cfg["model"]["additional_parameters"]["actuators"]["file"],
                    )
                    self.actuators = actuator_parser.get(actuator_model_file)
                    if (
                        cfg["model"]["additional_parameters"]["actuators"][
                            "replace_existing"
                        ]
                        == True
                    ):
                        parser.actuators = self.actuators.get_actuators()
                    else:
                        raise NotImplementedError(
                            "Adding actuators to existing models is not implemented yet. Try replacing them instead."
                        )
                    # actuator_model = actuator_parser.get(actuator_model_file)
                else:
                    raise NotImplementedError(
                        "Taking actuators from a model definition file is not implemented yet. Try using additional_parameters in the yaml file instead."
                    )
                if "joints" in cfg["model"]["additional_parameters"]:
                    pass  # Joint limits not implemented yet # Limit actuators is also an actuator in the end

        self.run = {}
        self._create_dictionaries(parser)
        if get_hash:
            return
        self._create_sympy_model()
        self._set_default_values()

        # Future work, tbd: (These should be disabled or enabled by a flag in the config file); so that we don't need to compile everything every time
        self._create_eom()
        self._create_jax_eom()
        self._create_FK(True)
        if hasattr(self, "gc_model"):
            self.gc_model.process_eom(
                self, body_weight=sum(self.default_values[_slice(self.masses)])
            )  # If the GC model needs to add extra equations of motion, it will do so here
            self._register_contact_model(self.gc_model)
        if hasattr(self, "actuators"):
            self.actuators.process_eom(self)
            self._register_actuator_model(self.actuators)
        self._create_variable_dataframe()
        # self._create_FK(parser)
        # self._create_IMU(parser) ....

    def _create_dictionaries(self, parser):
        """
        Create dictionaries for coordinates, speeds, accelerations, forces, joints, bodies, and external forces.
        Each contains:
        - list of names
        - start_index in the state vector
        - number of items
        These data is needed to build the state vector _v correctly and index on it
        """
        """
            ToDos: 
            - Treat constrained joints - they actually change the number of DOFs
            - Add muscles, they have a different number of states than torques
            - Mujoco: allow assignment of less bodies for external forces
            - All in all: work in progress, to be iterated over
            The current state, however, is good enough to define the OCP
        """

        self.dicts = {
            "bodies": parser.get_bodies(),
            "joints": parser.get_joints(),
            "sites": parser.get_sites(),
        }

        # Create overviews of all states
        n_dof = parser.get_n_joints()
        self.coordinates = {
            "names": [f"q_{joint['name']}" for joint in parser.get_joints()],
            "idx": 0,
            "n": n_dof,
        }
        self.speeds = {
            "names": [f"qd_{joint['name']}" for joint in parser.get_joints()],
            "idx": n_dof,
            "n": n_dof,
        }
        self.accs = {
            "names": [f"qdd_{joint['name']}" for joint in parser.get_joints()],
            "idx": 2 * n_dof,
            "n": n_dof,
        }

        n_forces = parser.get_n_internal_forces()
        int_forces_dict = parser.get_internal_forces()
        self.forces = {
            "names": [
                f"M_{int_forces_dict[name]['joint']}" for name in int_forces_dict.keys()
            ],
            "idx": 3 * n_dof,
            "n": n_forces,
        }

        # The first representation of external forces is a list of bodies, where the forces can be applied
        n_ext_forces = parser.get_n_external_forces()
        self.ext_forces = {
            "names": [
                f"m_{force}_{dim}"
                for force in parser.get_external_forces_bodies()
                for dim in ["x", "y", "z"]
            ],
            "idx": 3 * n_dof + n_forces,
            "n": n_ext_forces,
        }

        # And torques
        self.ext_torques = {
            "names": [
                f"t_{force}_{dim}"
                for force in parser.get_external_forces_bodies()
                for dim in ["x", "y", "z"]
            ],
            "idx": 3 * n_dof + n_forces + n_ext_forces,
            "n": n_ext_forces,
        }

        self.state_vector = (
            self.coordinates["names"]
            + self.speeds["names"]
            + self.accs["names"]
            + self.forces["names"]
            + self.ext_forces["names"]
            + self.ext_torques["names"]
        )
        self.n_states = len(self.state_vector)

        # Create dictionaries for all constants
        i = len(self.state_vector)
        self.g = {
            "names": ["g_x", "g_y", "g_z"],
            "idx": i,
            "n": 3,
        }
        i += 3

        self.masses = {
            "names": [f"m_{body['name']}" for body in parser.get_bodies()],
            "idx": i,
            "n": len(parser.get_bodies()),
        }
        i += len(parser.get_bodies())

        inertia_tensor = ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]
        self.inertia = {
            "names": [
                f"I_{body['name']}_{dim}"
                for body in parser.get_bodies()
                for dim in inertia_tensor
            ],
            "idx": i,
            "n": len(parser.get_bodies()) * len(inertia_tensor),
        }
        i += len(parser.get_bodies()) * len(inertia_tensor)

        self.com = {
            "names": [
                f"com_{body['name']}_{dim}"
                for body in parser.get_bodies()
                for dim in ["x", "y", "z"]
            ],
            "idx": i,
            "n": len(parser.get_bodies()) * 3,
        }
        i += len(parser.get_bodies()) * 3

        self.offset = {
            "names": [
                f"offset_{body['name']}_{dim}"
                for body in parser.get_bodies()
                for dim in ["x", "y", "z"]
            ],
            "idx": i,
            "n": len(parser.get_bodies()) * 3,
        }
        i += len(parser.get_bodies()) * 3

        self.constants = (
            self.g["names"]
            + self.masses["names"]
            + self.inertia["names"]
            + self.com["names"]
            + self.offset["names"]
        )
        self.n_constants = len(self.constants)

        self.gravity = parser.get_gravity()
        self.default_inputs = {
            "states": {
                "model": np.zeros(self.n_states),
                "gc_model": np.zeros(0),
                "actuator_model": np.zeros(0),
            },
            "constants": {
                "model": np.zeros(self.n_constants),
                "gc_model": np.zeros(0),
                "actuator_model": np.zeros(0),
            },
            "input_names": ["model"],
        }

    def _set_default_values(self):
        """
        Create default values for the model.
        These are used to initialize the model and can be changed later.
        """
        # The slicing for bodies is not super clean - the idx are
        self.default_values = np.zeros(self._nv)
        self.default_values[_slice(self.g)] = np.array(self.gravity)
        self.default_values[_slice(self.masses)] = np.array(
            [body["mass"] for body in self.dicts["bodies"]]
        ).squeeze()

        def concat_defaults(value):
            """
            Concatenate all default values for the bodies
            """
            all_values = []
            for body in self.dicts["bodies"]:
                all_values.append(body[value])
            # return a
            return np.array(all_values).flatten()

        # Get all values that are stored as lists
        for value_dict, value in zip(
            [self.com, self.offset, self.inertia], ["com", "body_offset", "inertia"]
        ):
            self.default_values[_slice(value_dict)] = concat_defaults(value)

        self.default_inputs["constants"]["model"] = self.default_values[self.n_states :]

    def _create_sympy_model(self):
        """
        Create the equations of motion (EOM) for the model.
        Every value should be stored in the vector self._v.
        """
        self._nv = self.n_states + self.n_constants
        # We set everything with the IndexedBase, so that it is vectorized
        # However, coordinates and speeds need to be dynamicsymbols, therefore we create a separate vector for them and merge them later
        self._v = Matrix(self._nv, 1, lambda i, _: symbols(f"v{i}"))
        self._dynamic = Matrix(
            2 * self.coordinates["n"], 1, lambda i, _: dynamicsymbols(f"dyn{i}")
        )
        # Maybe the IndexedBase needs to be initialized with its data types
        # e.g. [self._v[i] for i in :self.n_states] = [dynamicsymbols(name) for name in self.state_vector]; [self._v[i] for i in self.n_states:] = [symbols(name) for name in self.constants]
        self.ground_frame = ReferenceFrame("ground")  # Fixed ground frame
        self.origin = Point("origin")
        self.origin.set_vel(
            self.ground_frame, 0
        )  # For treadmill models, we could just set a velocity here
        self.body_origins = {}
        self.rigid_bodies = {}
        self.reference_frames = {}
        self.mass_centers = {}
        self.loads = []
        # kinematic differential equations: d coordinates - speeds = 0
        self.kd_eqs = [
            a - b
            for a, b in zip(
                [self._dynamic[i].diff() for i in _slice(self.coordinates)],
                [self._dynamic[i] for i in _slice(self.speeds)],
            )
        ]

        # Get the model topology (A tree-like to help navigating the model)
        # This might change the indexing order of the bodies --> needs to be tested
        # Use body_idx as a substitute for now to be safe
        self.topology_tree = self._create_topology_tree()

        def build_reference_frames(topology, parent_frame=None, parent_origin=None):
            for idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body["name"] for body in self.dicts["bodies"]].index(
                    body_name
                )
                children = node["children"]

                # Get the current body data
                body = next(
                    (b for b in self.dicts["bodies"] if b["name"] == body_name), None
                )

                parent_frame = (
                    self.ground_frame if parent_frame is None else parent_frame
                )
                parent_origin = self.origin if parent_origin is None else parent_origin

                body_origin = Point(f"{body_name}_origin")
                joint_offset = _to_sympy_vector(
                    [
                        self._v[i]
                        for i in range(
                            self.offset["idx"] + 3 * body_idx,
                            self.offset["idx"] + 3 * (body_idx + 1),
                        )
                    ],
                    parent_frame,
                )
                joint_speed = 0
                # Slide joint logic
                n_hinges = 0
                for joint in body["joints"]:
                    if joint["type"] == "slide":
                        joint_offset += (
                            _to_sympy_vector(joint["axis"], parent_frame)
                            * self._dynamic[
                                self.coordinates["idx"]
                                + self.coordinates["names"].index(f"q_{joint['name']}")
                            ]
                        )
                        joint_speed += (
                            _to_sympy_vector(joint["axis"], parent_frame)
                            * self._dynamic[
                                self.speeds["idx"]
                                + self.speeds["names"].index(f"qd_{joint['name']}")
                            ]
                        )
                    elif joint["type"] == "hinge":
                        n_hinges += 1

                body_origin.set_pos(parent_origin, joint_offset)
                body_origin.set_vel(parent_frame, joint_speed)
                self.body_origins[body_name] = body_origin
                body_frame = ReferenceFrame(f"{body_name}_frame")

                intermediate_frames = [parent_frame]  # We use one frame per rotation
                # We may need to expose intermediate frames for adding joint torques to them
                idx_h = 0  # hidden iterator to only iterate over hinges
                idx = 0  # iterator for all jointss
                while idx < n_hinges:
                    joint = body["joints"][idx_h]
                    if joint["type"] == "hinge":
                        # Check if we need to add a new frame
                        symbol = self._dynamic[
                            self.coordinates["idx"]
                            + self.coordinates["names"].index(f"q_{joint['name']}")
                        ]
                        symbol_dot = self._dynamic[
                            self.speeds["idx"]
                            + self.speeds["names"].index(f"qd_{joint['name']}")
                        ]
                        joint_angle = _to_sympy_vector(
                            joint["axis"], intermediate_frames[-1]
                        )
                        joint_angvel = _to_sympy_vector(
                            joint["axis"], intermediate_frames[-1]
                        )
                        if idx == n_hinges - 1:
                            # I think that we need add the joints iteratively to the frame, order matters
                            body_frame.orient(
                                intermediate_frames[-1], "Axis", (symbol, joint_angle)
                            )
                            body_frame.set_ang_vel(
                                intermediate_frames[-1], joint_angvel * symbol_dot
                            )
                        else:
                            new_frame = ReferenceFrame(
                                f"{body_name}_{joint['name']}frame_{idx}"
                            )
                            new_frame.orient(
                                intermediate_frames[-1], "Axis", (symbol, joint_angle)
                            )
                            new_frame.set_ang_vel(
                                intermediate_frames[-1], joint_angvel * symbol_dot
                            )
                            intermediate_frames.append(new_frame)
                        idx += 1
                    idx_h += 1

                self.reference_frames[body_name] = body_frame
                build_reference_frames(children, body_frame, body_origin)

        build_reference_frames(self.topology_tree)

        def build_bodies(topology):
            for idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body["name"] for body in self.dicts["bodies"]].index(
                    body_name
                )
                children = node["children"]
                body_origin = self.body_origins[body_name]
                body_frame = self.reference_frames[body_name]

                # Set pos and lin_vel for the body
                mass_center_point = Point(f"{body_name}_mass_center")
                com_pos = _to_sympy_vector(
                    [
                        self._v[i]
                        for i in range(
                            self.com["idx"] + 3 * body_idx,
                            self.com["idx"] + 3 * (body_idx + 1),
                        )
                    ],
                    body_frame,
                )
                mass_center_point.set_pos(body_origin, com_pos)
                mass_center_point.v2pt_theory(
                    body_origin, self.ground_frame, body_frame
                )
                self.mass_centers[body_name] = mass_center_point

                # set inertia tensor
                inertia_tensor = [
                    self._v[i + self.inertia["idx"] + 6 * body_idx] for i in range(6)
                ]
                body_inertia = Inertia.from_inertia_scalars(
                    mass_center_point, body_frame, *inertia_tensor
                )

                # Create the body
                body = RigidBody(
                    body_name,
                    mass_center_point,
                    body_frame,
                    self._v[self.masses["idx"] + body_idx],
                    body_inertia,
                )
                self.rigid_bodies[body_name] = body

                build_bodies(children)

        build_bodies(self.topology_tree)

        # Add gravitational forces
        for bodyname, rigid_body in self.rigid_bodies.items():
            gravity_f = rigid_body.mass * (
                self.ground_frame.x * self._v[self.g["idx"]]
                + self.ground_frame.y * self._v[self.g["idx"] + 1]
                + self.ground_frame.z * self._v[self.g["idx"] + 2]
            )
            self.loads.append((rigid_body.masscenter, gravity_f))

        # Add internal forces
        for i, joint_name in enumerate(self.forces["names"]):
            joint_name = "_".join(joint_name.split("_")[1:])
            assert joint_name in [
                j["name"] for j in self.dicts["joints"]
            ], f"Joint {joint_name} not found in the model definition file, but has an actuator defined."
            joint_idx = [j["name"] for j in self.dicts["joints"]].index(joint_name)
            axis = self.dicts["joints"][joint_idx]["axis"]
            parent_body = self.dicts["joints"][joint_idx]["parent"]
            if self.dicts["joints"][joint_idx]["type"] != "hinge":
                raise NotImplementedError(
                    "Internal forces are only implemented for hinge joints. (Are you a hydraulic excavator or why do you need a slide joint?)"
                )
            force_idx = self.forces["idx"] + i
            force_vector = _to_sympy_vector(
                [self._v[force_idx] * axis[j] for j in range(3)], self.ground_frame
            )
            force_body = (self.body_origins[parent_body], force_vector)
            self.loads.append(force_body)

        # Add external forces - double check if this is correct
        # We are using body.origin to apply the force, but it could also be applied to the mass center or an arbitrary point - that is tbd
        for idx, force in enumerate(self.ext_forces["names"]):
            # Only parse every 3rd entry, because they are x,y,z
            if idx % 3 != 0:
                continue
            body_name = "_".join(force.split("_")[1:-1])
            assert body_name in [
                body["name"] for body in self.dicts["bodies"]
            ], f"Body {body_name} not found in the model, but has an external force defined."
            force_idx = self.ext_forces["idx"] + idx
            force_vector = _to_sympy_vector(
                [self._v[i] for i in range(force_idx, force_idx + 3)], self.ground_frame
            )
            # print(f"Force vector: {force_vector} at {self.body_origins[body_name]}")
            force_body = (self.body_origins[body_name], force_vector)
            self.loads.append(force_body)

        # Add external torques - also double check if this is correct
        for idx, torque in enumerate(self.ext_torques["names"]):
            if idx % 3 != 0:
                continue
            body_name = "_".join(torque.split("_")[1:-1])
            torque_idx = self.ext_torques["idx"] + idx
            torque_vector = _to_sympy_vector(
                [self._v[i] for i in range(torque_idx, torque_idx + 3)],
                self.ground_frame,
            )
            # print(f"Torque vector: {torque_vector} at {self.body_origins[body_name]}")
            torque_body = (self.reference_frames[body_name], torque_vector)
            self.loads.append(torque_body)

    def _create_topology_tree(self):
        """
        Create a tree-like topology structure based on the hierarchical relationship
        of bodies defined in self.dicts['bodies']
        """
        # # First, create a dictionary mapping to quickly look up parent-child relationships
        # body_dict = {body['name']: body for body in self.dicts['bodies']}

        # Define a recursive function to build the tree structure
        def build_tree(parent_name):
            children = [
                body["name"]
                for body in self.dicts["bodies"]
                if body["parent"] == parent_name
            ]
            # Build the current node
            tree = {
                "name": parent_name,
                "children": [build_tree(child) for child in children],
            }
            return tree

        # Build topology starting from the root nodes
        root_bodies = [
            body["name"] for body in self.dicts["bodies"] if body["parent"] is None
        ]
        topology_tree = [
            build_tree(root) for root in root_bodies if root != "ground"
        ]  # Exclude ground
        return topology_tree

    def _create_eom(self):
        """
        Create the equations of motion (EOM) for the model.
        Every value should be stored in the vector self._v.
        """
        # Create the equations of motion using KanesMethod
        km = KanesMethod(
            self.ground_frame,
            q_ind=[self._dynamic[i] for i in _slice(self.coordinates)],
            u_ind=[self._dynamic[i] for i in _slice(self.speeds)],
            kd_eqs=self.kd_eqs,
        )
        self.fr, self.frstar = km.kanes_equations(
            list(self.rigid_bodies.values()), self.loads
        )
        self.kane = km
        self.constants_sym = [self._v[i] for i in range(self.n_states, self._nv)]
        self.state_vector_sym = [self._v[i] for i in range(self.n_states)]
        self.eom = self.fr + self.frstar
        # replace the accelerations in the EOM with the v_ states
        print(
            "Replacing dynamic symbols in the EOM with the v_ states, this might take a while..."
        )
        self.eom = self._replace_dyn(self.eom)

    def _create_jax_eom(self):
        """
        Create the equations of motion (EOM) for the model using JAX.
        Every value should be stored in the vector self._v.
        """
        import time

        a = time.time()
        self.confun = lambdify(
            self._v, self.eom, modules="jax", cse=True, docstring_limit=2
        )
        print(f"Lambdifying the EOM took {time.time()-a} seconds")
        a = time.time()
        self._precompile_fn(self.confun, self.default_inputs, "jacobian", jacobian=True)
        print(f"Precompiling the Jacobian took {time.time()-a} seconds")
        a = time.time()
        self._precompile_fn(self.confun, self.default_inputs, "confun")
        print(f"Precompiling the confun took {time.time()-a} seconds")
        a = time.time()
        self.mass_matrix = lambdify(
            self._v,
            self._replace_dyn(self.kane.mass_matrix),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.run["mass_matrix_uncompiled"] = self.mass_matrix
        self._precompile_fn(self.mass_matrix, self.default_inputs, "mass_matrix")
        print(f"Precompiling the mass matrix took {time.time()-a} seconds")
        a = time.time()
        self.forcing = lambdify(
            self._v,
            self._replace_dyn(self.kane.forcing),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.run["forcing_uncompiled"] = self.forcing
        self._precompile_fn(self.forcing, self.default_inputs, "forcing")
        print(f"Precompiling the forcing took {time.time()-a} seconds")
        a = time.time()

    def _create_FK(self, get_FK_dot=True):
        """
        Create the forward kinematics (FK) for the model.
        Every value should be stored in the vector self._v.
        Currently, this just returns the positions of the body_origins in the global frame.
        FK for markers etc. should be added in different functions --> max speed
        Currently, this takes in the
        """
        self.positions = [body for body in self.body_origins.keys()]
        pos_vector = []
        for _, point in self.body_origins.items():
            pos_vector.append(
                [
                    point.pos_from(self.origin).dot(frame_dim)
                    for frame_dim in [
                        self.ground_frame.x,
                        self.ground_frame.y,
                        self.ground_frame.z,
                    ]
                ]
            )
        pos_vector_ = Matrix(pos_vector)
        pos_vector_ = self._replace_dyn(pos_vector_)
        pos_vector_ = lambdify(
            self._v, pos_vector_, modules="jax", cse=True, docstring_limit=2
        )
        self.run["FK_uncompiled"] = pos_vector_
        pos_vector_ = self._precompile_fn(pos_vector_, self.default_inputs, "FK")

        if self.dicts["sites"] is not None:
            # Visualization version of pos_vector
            for site_ in self.dicts["sites"]:
                # Create a sympy point for the site
                site = Point(site_["name"])
                parent = site_["parent"]
                parent_frame = self.reference_frames[parent]
                site.set_pos(
                    self.body_origins[parent],
                    _to_sympy_vector(site_["pos"], parent_frame),
                )
                pos_vector.append(
                    [
                        site.pos_from(self.origin).dot(frame_dim)
                        for frame_dim in [
                            self.ground_frame.x,
                            self.ground_frame.y,
                            self.ground_frame.z,
                        ]
                    ]
                )
            pos_vector = Matrix(pos_vector)
            pos_vector = self._replace_dyn(pos_vector)
            pos_vector = lambdify(
                self._v, pos_vector, modules="jax", cse=True, docstring_limit=2
            )
            self.run["FK_vis_uncompiled"] = pos_vector
            pos_vector = self._precompile_fn(
                pos_vector, self.default_inputs, "FK_vis", skip_export=True
            )
        else:
            self.run["FK_vis_uncompiled"] = self.run["FK_uncompiled"]
            self.run["FK_vis"] = self.run["FK"]

        if get_FK_dot:
            vel_vector = []
            for _, point in self.body_origins.items():
                vel_vector.append(
                    [
                        point.vel(self.ground_frame).dot(frame_dim)
                        for frame_dim in [
                            self.ground_frame.x,
                            self.ground_frame.y,
                            self.ground_frame.z,
                        ]
                    ]
                )
            vel_vector = Matrix(vel_vector)
            vel_vector = self._replace_dyn(vel_vector)
            vel_vector = lambdify(
                self._v, vel_vector, modules="jax", cse=True, docstring_limit=2
            )
            vel_vector = self._precompile_fn(vel_vector, self.default_inputs, "FK_dot")

    def _precompile_fn(self, function, inputs, name, jacobian=False, skip_export=False):
        """
        Precompile a function using JAX's jit for faster execution.
        This is useful for functions that will be called multiple times with the same input shape.
        We use a hacky way: by serializing the function and then deserializing it again, the caching mechanism of jax doesn't miss parts of the function.
        This actually doesn't even seem to be slower than the normal jax.jit
        """

        def _jit_function_template(function, states, constants, input_names=["model"]):
            """
            The to-be registered function should take states and constants dicts as inputs.
            We use this wrapper to unpack the vectors for each function.
            """
            l_inputs = []
            for key in input_names:
                if key not in states:
                    continue
                l_inputs += [*states[key]]
                l_inputs += [*constants[key]]
            return function(*l_inputs)

        states0, constants0, input_names0 = (
            inputs["states"],
            inputs["constants"],
            inputs["input_names"],
        )
        jit_function = lambda states_, constants_: _jit_function_template(
            function, states_, constants_, input_names0
        )

        if skip_export:
            self.run[name] = jit_function
            return
        states_input_dummy = {
            k: jax.ShapeDtypeStruct((len(v),), np.float32) for k, v in states0.items()
        }
        constants_input_dummy = {
            k: jax.ShapeDtypeStruct((len(v),), np.float32)
            for k, v in constants0.items()
        }
        # Export+Import of the same function reconfigures the "lambda" function somehow -> faster re-compilation from cache
        if not jacobian:
            exported = jax.export.export(jax.jit(jit_function))(
                states_input_dummy, constants_input_dummy
            )
        else:
            exported = jax.export.export(jax.jit(jax.jacobian(jit_function)))(
                states_input_dummy, constants_input_dummy
            )
        re_ = jax.export.deserialize(exported.serialize(vjp_order=1))
        # Trigger the jit compilation
        re_.call(states0, constants0)
        # Add the function to the run dictionary
        self.run[name] = re_.call

    def _replace_dyn(self, function):
        """
        Replace the dynamicsymbols in the function with the corresponding v_ states.
        This is needed to get the correct output from the function.
        """
        # First get rid of the accelerations
        in_ = [self._dynamic[i].diff() for i in _slice(self.speeds)]
        out_ = [self._v[i] for i in _slice(self.accs)]
        function = function.xreplace(dict(zip(in_, out_)))
        # This is not 100% clean: we assume that coordinates is always first (which is true for now)
        in_ = [
            self._dynamic[i] for i in range(self.coordinates["n"] + self.speeds["n"])
        ]
        out_ = [self._v[i] for i in range(self.coordinates["n"] + self.speeds["n"])]
        return function.xreplace(dict(zip(in_, out_)))

    def _register_contact_model(self, contact_model):
        """
        Register the forward function of the contact model as a function in the run dictionary and precompile it.
        """
        input_dummy = self.default_inputs.copy()
        input_dummy["states"]["gc_model"] = np.zeros(contact_model.get_n_states())
        input_dummy["constants"]["gc_model"] = np.zeros(contact_model.get_n_constants())
        input_dummy["input_names"] = ["model", "gc_model"]
        self.default_inputs["states"]["gc_model"] = np.zeros(
            contact_model.get_n_states()
        )
        self.default_inputs["constants"]["gc_model"] = np.zeros(
            contact_model.get_n_constants()
        )
        lambda_func = lambda states, constants: contact_model.forward(
            states, constants, self
        )
        states_input_dummy = {
            k: jax.ShapeDtypeStruct((len(v),), np.float32)
            for k, v in input_dummy["states"].items()
        }
        constants_input_dummy = {
            k: jax.ShapeDtypeStruct((len(v),), np.float32)
            for k, v in input_dummy["constants"].items()
        }
        for fun in ["forward", "jacobian"]:
            if fun == "jacobian":
                self.run["gc_model_jacobian_uncompiled"] = (
                    lambda_func  # We might need this for the constraints at some point
                )
                # Uncompiled functions are not exported, so we need to use the compiled version
                exported = jax.export.export(jax.jit(jax.jacobian(lambda_func)))(
                    states_input_dummy, constants_input_dummy
                )
            else:
                self.run["gc_model_uncompiled"] = (
                    lambda_func  # We might need this for the constraints at some point
                )
                exported = jax.export.export(jax.jit(lambda_func))(
                    states_input_dummy, constants_input_dummy
                )
            re_ = jax.export.deserialize(exported.serialize(vjp_order=1))
            # Trigger the jit compilation
            re_.call(input_dummy["states"], input_dummy["constants"])
            # Add the function to the run dictionary
            if fun == "jacobian":
                self.run["gc_model_jacobian"] = re_.call
            else:
                self.run["gc_model"] = re_.call

    def _register_actuator_model(self, actuator_model):
        """
        TODO: Refactor the _register functions, they are almost identical
        """
        if actuator_model.is_torque_actuator():
            return
        input_dummy = self.default_inputs.copy()
        input_dummy["states"]["actuator_model"] = np.zeros(
            actuator_model.get_n_states()
        )
        input_dummy["constants"]["actuator_model"] = np.zeros(
            actuator_model.get_n_constants()
        )
        input_dummy["input_names"] = ["model", "actuator_model"]
        self.default_inputs["states"]["actuator_model"] = np.zeros(
            actuator_model.get_n_states()
        )
        self.default_inputs["constants"]["actuator_model"] = np.zeros(
            actuator_model.get_n_constants()
        )
        lambda_func = lambda states, constants: actuator_model.forward(
            states, constants, self
        )
        states_input_dummy = {
            k: jax.ShapeDtypeStruct((len(v),), np.float32)
            for k, v in input_dummy["states"].items()
        }
        constants_input_dummy = {
            k: jax.ShapeDtypeStruct((len(v),), np.float32)
            for k, v in input_dummy["constants"].items()
        }
        for fun in ["forward", "jacobian"]:
            if fun == "jacobian":
                self.run["actuator_model_jacobian_uncompiled"] = lambda_func
                exported = jax.export.export(jax.jit(jax.jacobian(lambda_func)))(
                    states_input_dummy, constants_input_dummy
                )
            else:
                self.run["actuator_model_uncompiled"] = lambda_func
                exported = jax.export.export(jax.jit(lambda_func))(
                    states_input_dummy, constants_input_dummy
                )
            re_ = jax.export.deserialize(exported.serialize(vjp_order=1))
            # Trigger the jit compilation
            re_.call(input_dummy["states"], input_dummy["constants"])
            # Add the function to the run dictionary
            if fun == "jacobian":
                self.run["actuator_model_jacobian"] = re_.call
            else:
                self.run["actuator_model"] = re_.call

    def _create_variable_dataframe(self):
        """
        Create a dataframe with all variables in the model.
        """
        df = pd.DataFrame(columns=["type", "name", "x0", "xmin", "xmax"])
        for i, name in enumerate(self.state_vector):
            type = "state"
            x0 = self.default_values[i]
            xmin = -3.14  # -np.inf
            xmax = 3.14  # np.inf
            # @todo: parse limits and find reasonable limits
            df.loc[len(df)] = [type, name, x0, xmin, xmax]
        for i, name in enumerate(self.constants):
            type = "constant"
            x0 = self.default_values[i + self.n_states]
            xmin = -np.inf
            xmax = np.inf
            df.loc[len(df)] = [type, name, x0, xmin, xmax]
        self.variables = df

    def _get_hash(self):
        # Create a string of all model state names (Is that really enough?)
        all_state_names = self.state_vector + self.constants
        all_state_names = "".join(all_state_names)
        # Create a hash of the string
        return hashlib.sha256(all_state_names.encode()).hexdigest()


def _slice(dictionary):
    """
    Slice the state vector according to the dictionary
    To make accesing states / v_ easier
    """
    return np.arange(dictionary["idx"], dictionary["idx"] + dictionary["n"])


def _to_sympy_vector(values, reference_frame):
    """
    Convert a list of positional or directional values to a sympy.physics.mechanics.Vector.

    Args:
        values (list or tuple): A list or tuple containing [x, y, z] values.
        reference_frame (ReferenceFrame): The reference frame to use for the vector conversion.

    Returns
    -------
        Vector: A sympy Vector representation of the input values.
    """
    if len(values) != 3:
        raise ValueError("The input values must be a list or tuple of length 3.")
    return (
        values[0] * reference_frame.x
        + values[1] * reference_frame.y
        + values[2] * reference_frame.z
    )


def load_model(model_file, force_rebuild=False):
    """
    Load a model from a file.
    The file can be in .xml, .osim, or .yaml format.
    The function will return a Model object.
    """
    # Generate a hash of the config / or xml tree and save the cloudpickled model in the cache
    model_hash = BiosymModel(model_file, get_hash=True)._get_hash()
    # replace the hash with a string
    if not force_rebuild:
        if os.path.exists(os.path.join(_model_cache, f"{model_hash}.cpkl")):
            print(f"Loading model from cache: {model_hash}.cpkl")
            with open(os.path.join(_model_cache, f"{model_hash}.cpkl"), "rb") as f:
                model = cloudpickle.load(f)
                return model

    model = BiosymModel(model_file)
    # Save the model to the cache
    with open(os.path.join(_model_cache, f"{model_hash}.cpkl"), "wb") as f:
        cloudpickle.dump(model, f)
    return model


def clear_caches():
    """
    Clear the caches for JAX and the model.
    """
    # Clear the JAX cache
    print("Clearing the JAX cache...")
    if os.path.exists(_cachedir):
        for file in os.listdir(_cachedir):
            os.remove(os.path.join(_cachedir, file))
    print("Done.")
    # Clear the model cache
    assert (
        input(
            "Are you sure you want to clear the model cache? This deletes all compiled sympy models (y/n)"
        )
        == "y"
    ), "You need to confirm the deletion of the model cache."
    if os.path.exists(_model_cache):
        for file in os.listdir(_model_cache):
            if file.endswith(".cpkl"):
                os.remove(os.path.join(_model_cache, file))
    print("Done.")


# Small testing script
if __name__ == "__main__":
    import time

    start = time.time()
    model_file = "tests/test_models/pendulum.xml"
    # model_file = "tests/test_models/gait2d_torque/gait2d_torque.yaml"
    model = load_model(model_file, True)
    print(f"Reloading model in {time.time()-start} seconds")
    start = time.time()
    states = {
        "model": np.zeros(model.n_states),
        "gc_model": np.zeros(0),
        "actuator_model": np.zeros(0),
    }
    constants = {
        "model": np.zeros(model.n_constants),
        "gc_model": np.zeros(0),
        "actuator_model": np.zeros(0),
    }
    for _ in range(1):
        model.run["jacobian"](states, constants)
    print(f"1 jacobian in in {time.time()-start} seconds (with reimporting)")
    start = time.time()
    import timeit

    a = timeit.timeit(lambda: model.run["jacobian"](states, constants), number=10000)
    print(f"jacobian runs in {a/10000} seconds")

    start = time.time()
    for _ in range(1):
        model.run["confun"](states, constants)
    print(f"1 confun in in {time.time()-start} seconds (with reimporting)")
    start = time.time()
    import timeit

    a = np.ones(model._nv)
    a = timeit.timeit(lambda: model.run["confun"](states, constants), number=10000)
    print(f"confun runs in {a/10000} seconds")

    model.run["gc_model"](states, constants)
    a = timeit.timeit(lambda: model.run["gc_model"](states, constants), number=10000)
    print(f"gc_model runs in {a/10000} seconds")

    a = timeit.timeit(
        lambda: model.run["gc_model_jacobian"](states, constants), number=10000
    )
    print(f"gc_model jacobian runs in {a/10000} seconds")

    print("confun shape:", model.run["confun"](states, constants).shape)
    print("jacobian shape:", model.run["jacobian"](states, constants)["model"].shape)
    print("gc_model shape:", model.run["gc_model"](states, constants)[1].shape)
    print(
        "gc_model jacobian shape:",
        model.run["gc_model_jacobian"](states, constants)[0]["model"].shape,
    )
