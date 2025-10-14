"""
BiosymModel: Core biomechanical modeling functionality.

This module provides the main BiosymModel class for creating, manipulating,
and simulating biomechanical models from various input formats (XML, OSIM, YAML).
The module handles model parsing, symbolic equation generation, JAX compilation,
and provides interfaces for optimal control problems.
"""
import os
from typing import Any, Callable, List, Optional, Tuple, Union

_cachedir = os.path.expanduser("~/.biosym/jax_cache")
_model_cache = os.path.expanduser("~/.biosym/")
os.environ["JAX_COMPILATION_CACHE_DIR"] = _cachedir  # This needs to happen before importing jax
os.environ["jax_persistent_cache_min_compile_time_secs".upper()] = "0.01"
os.makedirs((_cachedir), exist_ok=True)

import hashlib
from functools import partial

import cloudpickle
import jax
import jax.export
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax import tree_util
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
from biosym.model.actuators.actuator_models.passive_torques import PassiveTorques
from biosym.model.contact import *
from biosym.model.parsers import *
from biosym.model.parsers.base_parser import BaseParser
from biosym.utils import states


class BiosymModel:
    """Biomechanical model class for biosym.
    
    This class provides functionality to load, save, and manipulate biomechanical
    models from various formats. It handles symbolic mechanics, JAX compilation,
    and provides interfaces for optimization and simulation.
    """

    def __init__(self, definition_file: str, get_hash: bool = False) -> None:
        """Initialize a BiosymModel from a definition file.
        
        Parameters
        ----------
        definition_file : str
            Path to model definition file (.xml, .osim, or .yaml)
        get_hash : bool, optional
            If True, only compute model hash without full initialization, by default False
            
        Raises
        ------
        ValueError
            If definition_file format is not supported
        NotImplementedError
            If trying to load .osim files (not yet supported)
        """
        # I think that it makes sense to force .yaml files at some point, because there are settings that are not represented in the model files
        # .yaml files can define additional variables, for mujoco that would be ground contact
        cfg = None
        if definition_file.endswith(".yaml"):
            # Load the yaml file and check for the model name
            with open(definition_file) as f:
                cfg = yaml.safe_load(f)
            # Replace the definition file with the model name
            definition_file = os.path.join(os.path.dirname(definition_file), cfg["model"]["name"])

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
            raise ValueError("Model definition file must be in .xml, .osim, or .yaml format.")

        # Check for additional parameters
        if cfg is not None:
            if "additional_parameters" in cfg["model"]:
                if "ground_contact" in cfg["model"]["additional_parameters"]:
                    gc_model_file = os.path.join(
                        os.path.dirname(definition_file),
                        cfg["model"]["additional_parameters"]["ground_contact"]["file"],
                    )
                    self.gc_model = contact_parser.get(gc_model_file)
                    if cfg["model"]["additional_parameters"]["ground_contact"]["replace_existing"] == True:
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
                    if cfg["model"]["additional_parameters"]["actuators"]["replace_existing"] == True:
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

    def _create_dictionaries(self, parser: 'BaseParser') -> None:
        """Create dictionaries for model components.
        
        Creates dictionaries for coordinates, speeds, accelerations, forces,
        joints, bodies, and external forces. Each contains list of names,
        start_index in the state vector, and number of items.
        
        Parameters
        ----------
        parser : BaseParser
            Model parser instance containing parsed model data
            
        Notes
        -----
        This data is needed to build the state vector correctly and index on it.
        
        Todo
        ----
        - Treat constrained joints - they actually change the number of DOFs
        - Add muscles, they have a different number of states than torques
        - Mujoco: allow assignment of less bodies for external forces
        """

        all_sites = parser.get_sites()
        markers = [s for s in all_sites if s.get("name") != "torso"]
        sites = [s for s in all_sites if s.get("name") == "torso"]

        self.dicts = {
            "bodies": parser.get_bodies(),
            "joints": parser.get_joints(),
            "sites": sites,     # keep only torso as a site
            "markers": markers,  # all other xml sites become markers
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

        # Passive actuators need to be defined here
        self.passive_actuators = PassiveTorques(self.dicts["joints"])
        passive_actuated_joints = self.passive_actuators.get_actuated_joints()

        active_forces_dict = parser.get_internal_forces()
        active_actuated_joints = [active_forces_dict[name]["joint"] for name in active_forces_dict.keys()]
        actuated_joints = list(dict.fromkeys(active_actuated_joints + passive_actuated_joints))
        all_joints = [joint["name"] for joint in self.dicts["joints"]]
        self.forces = {
            "names": [f"M_{joint}" for joint in all_joints if joint in actuated_joints],
            "idx": 3 * n_dof,
            "n": len(actuated_joints),
            "passive_idx": jnp.array([i for i, j in enumerate(all_joints) if j in passive_actuated_joints]),
            "active_idx": jnp.array([i for i, j in enumerate(all_joints) if j in active_actuated_joints]),
            "combined_idx": jnp.array([i for i, j in enumerate(all_joints) if j in actuated_joints]),
        }

        # The first representation of external forces is a list of bodies, where the forces can be applied
        n_ext_forces = parser.get_n_external_forces()
        self.ext_forces = {
            "names": [f"f_{force}_{dim}" for force in parser.get_external_forces_bodies() for dim in ["x", "y", "z"]],
            "idx": 3 * n_dof + len(actuated_joints),
            "n": n_ext_forces,
        }

        # And torques
        self.ext_torques = {
            "names": [f"m_{force}_{dim}" for force in parser.get_external_forces_bodies() for dim in ["x", "y", "z"]],
            "idx": 3 * n_dof + len(actuated_joints) + n_ext_forces,
            "n": n_ext_forces,
        }

        # Sites and markers are fixed relative to parent body frame -> NOT part of state vector
        
        n_sites = len(sites)
        self.sites_parsed = {
            "base_names": [f"site_{s['name']}" for s in sites],
            "n_sites": n_sites,
        }

        n_markers = len(markers)
        self.markers_parsed = {
            "base_names": [f"marker_{m['name']}" for m in markers],
            "n_markers": n_markers,
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
            "names": [f"I_{body['name']}_{dim}" for body in parser.get_bodies() for dim in inertia_tensor],
            "idx": i,
            "n": len(parser.get_bodies()) * len(inertia_tensor),
        }
        i += len(parser.get_bodies()) * len(inertia_tensor)

        self.com = {
            "names": [f"com_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ["x", "y", "z"]],
            "idx": i,
            "n": len(parser.get_bodies()) * 3,
        }
        i += len(parser.get_bodies()) * 3

        self.offset = {
            "names": [f"offset_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ["x", "y", "z"]],
            "idx": i,
            "n": len(parser.get_bodies()) * 3,
        }
        i += len(parser.get_bodies()) * 3

        self.constants = (
            self.g["names"] + self.masses["names"] + self.inertia["names"] + self.com["names"] + self.offset["names"]
        )
        self.n_constants = len(self.constants)

        self.gravity = parser.get_gravity()
        self.default_inputs = states.dict_to_dataclass(
            {
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
            }
        )

    def _set_default_values(self) -> None:
        """Create default values for the model.
        
        These are used to initialize the model and can be changed later.
        Sets gravity, masses, and other body parameters to their default values.
        """
        # The slicing for bodies is not super clean - the idx are
        self.default_values = np.zeros(self._nv)
        self.default_values[_slice(self.g)] = np.array(self.gravity)
        self.default_values[_slice(self.masses)] = np.array([body["mass"] for body in self.dicts["bodies"]]).squeeze()

        def concat_defaults(value: str) -> np.ndarray:
            """
            Concatenate all default values for the bodies
            """
            all_values = []
            for body in self.dicts["bodies"]:
                all_values.append(body[value])
            # return a
            return np.array(all_values).flatten()

        # Get all values that are stored as lists
        for value_dict, value in zip([self.com, self.offset, self.inertia], ["com", "body_offset", "inertia"]):
            self.default_values[_slice(value_dict)] = concat_defaults(value)

        self.default_inputs = self.default_inputs.replace_vector(
            "constants", "model", self.default_values[self.n_states :]
        )

    def _create_sympy_model(self) -> None:
        """Create the symbolic equations of motion (EOM) for the model.
        
        Creates symbolic representations using SymPy for coordinates, speeds,
        and forces. Every value is stored in the vector self._v for vectorization.
        Sets up reference frames, bodies, and mechanical constraints.
        """
        self._nv = self.n_states + self.n_constants
        # We set everything with the IndexedBase, so that it is vectorized
        # However, coordinates and speeds need to be dynamicsymbols, therefore we create a separate vector for them and merge them later
        self._v = Matrix(self._nv, 1, lambda i, _: symbols(f"v{i}"))
        self._dynamic = Matrix(2 * self.coordinates["n"], 1, lambda i, _: dynamicsymbols(f"dyn{i}"))
        # Maybe the IndexedBase needs to be initialized with its data types
        # e.g. [self._v[i] for i in :self.n_states] = [dynamicsymbols(name) for name in self.state_vector]; [self._v[i] for i in self.n_states:] = [symbols(name) for name in self.constants]
        self.ground_frame = ReferenceFrame("ground")  # Fixed ground frame
        self.origin = Point("origin")
        self.origin.set_vel(self.ground_frame, 0)  # For treadmill models, we could just set a velocity here
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

        def build_reference_frames(topology: list[dict[str, Any]], parent_frame: Optional[ReferenceFrame] = None, parent_origin: Optional[Point] = None) -> None:
            for idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body["name"] for body in self.dicts["bodies"]].index(body_name)
                children = node["children"]

                # Get the current body data
                body = next((b for b in self.dicts["bodies"] if b["name"] == body_name), None)

                parent_frame = self.ground_frame if parent_frame is None else parent_frame
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
                                self.coordinates["idx"] + self.coordinates["names"].index(f"q_{joint['name']}")
                            ]
                        )
                        joint_speed += (
                            _to_sympy_vector(joint["axis"], parent_frame)
                            * self._dynamic[self.speeds["idx"] + self.speeds["names"].index(f"qd_{joint['name']}")]
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
                            self.coordinates["idx"] + self.coordinates["names"].index(f"q_{joint['name']}")
                        ]
                        symbol_dot = self._dynamic[
                            self.speeds["idx"] + self.speeds["names"].index(f"qd_{joint['name']}")
                        ]
                        joint_angle = _to_sympy_vector(joint["axis"], intermediate_frames[-1])
                        joint_angvel = _to_sympy_vector(joint["axis"], intermediate_frames[-1])
                        if idx == n_hinges - 1:
                            # I think that we need add the joints iteratively to the frame, order matters
                            body_frame.orient(intermediate_frames[-1], "Axis", (symbol, joint_angle))
                            body_frame.set_ang_vel(intermediate_frames[-1], joint_angvel * symbol_dot)
                        else:
                            new_frame = ReferenceFrame(f"{body_name}_{joint['name']}_frame_{idx}")
                            new_frame.orient(intermediate_frames[-1], "Axis", (symbol, joint_angle))
                            new_frame.set_ang_vel(intermediate_frames[-1], joint_angvel * symbol_dot)
                            intermediate_frames.append(new_frame)
                        idx += 1
                    idx_h += 1

                self.reference_frames[body_name] = body_frame
                build_reference_frames(children, body_frame, body_origin)

        build_reference_frames(self.topology_tree)

        def build_bodies(topology: list[dict[str, Any]]) -> None:
            for idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body["name"] for body in self.dicts["bodies"]].index(body_name)
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
                mass_center_point.v2pt_theory(body_origin, self.ground_frame, body_frame)
                self.mass_centers[body_name] = mass_center_point

                # set inertia tensor
                inertia_tensor = [self._v[i + self.inertia["idx"] + 6 * body_idx] for i in range(6)]
                body_inertia = Inertia.from_inertia_scalars(mass_center_point, body_frame, *inertia_tensor)

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

        # create site Points so vel/acc exist symbolically (fixed offset in parent frame)
        self.markers = {}
        markers_list = self.dicts.get("markers")
        for marker_ in markers_list:
            name = marker_.get("name")
            parent = marker_.get("parent")
            parent_frame = self.reference_frames[parent]
            parent_origin = self.body_origins[parent]
            marker_pt = Point(f"{name}_marker")
            # Use parser-provided local position (fixed in parent frame), otherwise zero
            local_pos = marker_.get("pos", [0.0, 0.0, 0.0])
            sym_vec = _to_sympy_vector(list(local_pos), parent_frame)
            marker_pt.set_pos(parent_origin, sym_vec)
            marker_pt.v2pt_theory(parent_origin, self.ground_frame, parent_frame)
            self.markers[name] = marker_pt
        
        self.sites = {}
        sites_list = self.dicts.get("sites")
        for site_ in sites_list:
            name = site_.get("name")
            parent = site_.get("parent")
            parent_frame = self.reference_frames[parent]
            parent_origin = self.body_origins[parent]
            site_pt = Point(f"{name}_site")
            # Use parser-provided local position (fixed in parent frame), otherwise zero
            local_pos = site_.get("pos", [0.0, 0.0, 0.0])
            sym_vec = _to_sympy_vector(list(local_pos), parent_frame)
            site_pt.set_pos(parent_origin, sym_vec)
            site_pt.v2pt_theory(parent_origin, self.ground_frame, parent_frame)
            self.sites[name] = site_pt

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
            assert joint_name in [j["name"] for j in self.dicts["joints"]], (
                f"Joint {joint_name} not found in the model definition file, but has an actuator defined."
            )
            joint_idx = [j["name"] for j in self.dicts["joints"]].index(joint_name)
            axis = self.dicts["joints"][joint_idx]["axis"]
            parent_body = self.dicts["joints"][joint_idx]["parent"]
            child_body = self.dicts["joints"][joint_idx]["child"]

            parent_frame = self.reference_frames[parent_body] if parent_body != "ground_frame" else self.ground_frame
            child_frame = self.reference_frames[child_body]
            if self.dicts["joints"][joint_idx]["type"] != "hinge":
                raise NotImplementedError(
                    "Internal forces are only implemented for hinge joints. (Are you a hydraulic excavator or why do you need a slide joint?)"
                )
            force_idx = self.forces["idx"] + i
            force_vector = _to_sympy_vector(
                [self._v[force_idx] * axis[j] for j in range(3)],
                parent_frame if parent_frame is not None else self.ground_frame,
            )
            force_body = (child_frame, force_vector)
            self.loads.append(force_body)

        # Add external forces - double check if this is correct
        # We are using body.origin to apply the force, but it could also be applied to the mass center or an arbitrary point - that is tbd
        for idx, force in enumerate(self.ext_forces["names"]):
            # Only parse every 3rd entry, because they are x,y,z
            if idx % 3 != 0:
                continue
            body_name = "_".join(force.split("_")[1:-1])
            assert body_name in [body["name"] for body in self.dicts["bodies"]], (
                f"Body {body_name} not found in the model, but has an external force defined."
            )
            force_idx = self.ext_forces["idx"] + idx
            force_vector = _to_sympy_vector([self._v[i] for i in range(force_idx, force_idx + 3)], self.ground_frame)
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

    def _create_topology_tree(self) -> list[dict[str, Any]]:
        """Create a tree-like topology structure.
        
        Creates a hierarchical tree structure based on the parent-child
        relationships of bodies defined in self.dicts['bodies'].
        
        Returns
        -------
        List[Dict[str, Any]]
            Tree structure with nested body relationships
        """
        # # First, create a dictionary mapping to quickly look up parent-child relationships
        # body_dict = {body['name']: body for body in self.dicts['bodies']}

        # Define a recursive function to build the tree structure
        def build_tree(parent_name: str) -> list[dict[str, Any]]:
            children = [body["name"] for body in self.dicts["bodies"] if body["parent"] == parent_name]
            # Build the current node
            tree = {
                "name": parent_name,
                "children": [build_tree(child) for child in children],
            }
            return tree

        # Build topology starting from the root nodes
        root_bodies = [body["name"] for body in self.dicts["bodies"] if body["parent"] == "ground_frame"]
        topology_tree = [build_tree(root) for root in root_bodies if root != "ground_frame"]  # Exclude ground_frame
        return topology_tree

    def _create_eom(self) -> None:
        """Create the equations of motion (EOM) for the model.
        
        Uses Kane's method to generate the symbolic equations of motion
        from the defined mechanical system. Stores results in the model
        for later compilation.
        """
        # Create the equations of motion using KanesMethod
        km = KanesMethod(
            self.ground_frame,
            q_ind=[self._dynamic[i] for i in _slice(self.coordinates)],
            u_ind=[self._dynamic[i] for i in _slice(self.speeds)],
            kd_eqs=self.kd_eqs,
        )
        self.fr, self.frstar = km.kanes_equations(list(self.rigid_bodies.values()), self.loads)
        self.kane = km
        self.constants_sym = [self._v[i] for i in range(self.n_states, self._nv)]
        self.state_vector_sym = [self._v[i] for i in range(self.n_states)]
        self.eom = self.fr + self.frstar
        # replace the accelerations in the EOM with the v_ states
        print("Replacing dynamic symbols in the EOM with the v_ states, this might take a while...")
        self.eom = self._replace_dyn(self.eom)

    def _create_jax_eom(self) -> None:
        """Create JAX-compiled equations of motion.
        
        Converts the symbolic equations of motion to JAX-compatible functions
        for high-performance numerical computation. Includes automatic
        differentiation capabilities.
        """
        import time

        a = time.time()
        self.confun = lambdify(self._v, self.eom, modules="jax", cse=True, docstring_limit=2)

        print(f"Lambdifying the EOM took {time.time() - a} seconds")
        a = time.time()
        self._precompile_fn(self.confun, self.default_inputs, "jacobian", jacobian=True)
        print(f"Precompiling the Jacobian took {time.time() - a} seconds")
        a = time.time()
        self._precompile_fn(self.confun, self.default_inputs, "confun")
        print(f"Precompiling the confun took {time.time() - a} seconds")
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
        print(f"Precompiling the mass matrix took {time.time() - a} seconds")
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
        print(f"Precompiling the forcing took {time.time() - a} seconds")
        a = time.time()

    def _create_FK(self, get_FK_dot: bool = True) -> None:
        """Create forward kinematics (FK) functions for the model.
        
        Creates symbolic and compiled functions for computing body positions
        and velocities in the global reference frame.
        
        Parameters
        ----------
        get_FK_dot : bool, optional
            Whether to also compute velocity kinematics, by default True
            
        Notes
        -----
        Currently returns positions of body_origins in the global frame.
        FK for markers etc. should be added in different functions for max speed.
        """
        self.positions = [body for body in self.body_origins.keys()]
        pos_vector = []
        for _, point in self.body_origins.items():
            pos_vector.append(
                [
                    point.pos_from(self.origin).dot(frame_dim) # position of bidy origins in ground (global) frame
                    for frame_dim in [
                        self.ground_frame.x,
                        self.ground_frame.y,
                        self.ground_frame.z,
                    ]
                ]
            )
        pos_vector_ = Matrix(pos_vector)
        pos_vector_ = self._replace_dyn(pos_vector_)
        pos_vector_ = lambdify(self._v, pos_vector_, modules="jax", cse=True, docstring_limit=2)
        self.run["FK_uncompiled"] = pos_vector_
        pos_vector_ = self._precompile_fn(pos_vector_, self.default_inputs, "FK")

        # Visualization FK: append site positions (use pre-built SymPy Points)
        if self.dicts.get("markers") is not None:
            # Marker Visualization FK: append markerpositions (use pre-built SymPy Points)
            if self.dicts.get("markers") is not None:
                if hasattr(self, "markers") and self.markers:
                    for marker_pt in self.markers.values():
                        pos_vector.append(
                            [
                                marker_pt.pos_from(self.origin).dot(frame_dim)
                                for frame_dim in (self.ground_frame.x, self.ground_frame.y, self.ground_frame.z)
                            ]
                        )
                
            pos_vector_marker = Matrix(pos_vector)
            pos_vector_marker = self._replace_dyn(pos_vector_marker)
            pos_vector_marker = lambdify(self._v, pos_vector_marker, modules="jax", cse=True, docstring_limit=2)
            self.run["FK_marker_uncompiled"] = pos_vector
            # store compiled/jitted visualization function
            self._precompile_fn(pos_vector_marker, self.default_inputs, "FK_marker", skip_export=True)
            
            if (self.dicts.get("sites") is not None) and (self.dicts.get("markers") is None):
                if hasattr(self, "sites") and self.sites:
                    for site_pt in self.sites.values():
                        pos_vector.append(
                            [
                                site_pt.pos_from(self.origin).dot(frame_dim)
                                for frame_dim in (self.ground_frame.x, self.ground_frame.y, self.ground_frame.z)
                            ]
                        )

                pos_vector_vis = Matrix(pos_vector)
                pos_vector_vis = self._replace_dyn(pos_vector_vis)
                pos_vector_vis = lambdify(self._v, pos_vector_vis, modules="jax", cse=True, docstring_limit=2)
                self.run["FK_vis_uncompiled"] = pos_vector
                # store compiled/jitted visualization function
                self._precompile_fn(pos_vector_vis, self.default_inputs, "FK_vis", skip_export=True)

        else: 
            # no sites defined -> visualization FK is same as body-only FK
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
            vel_vector = lambdify(self._v, vel_vector, modules="jax", cse=True, docstring_limit=2)
            vel_vector = self._precompile_fn(vel_vector, self.default_inputs, "FK_dot")

            acc_vector = []
            for _, point in self.body_origins.items():
                acc_vector.append(
                    [
                        point.acc(self.ground_frame).dot(frame_dim)
                        for frame_dim in [
                            self.ground_frame.x,
                            self.ground_frame.y,
                            self.ground_frame.z,
                        ]
                    ]
                )
            acc_vector = Matrix(acc_vector)
            acc_vector = self._replace_dyn(acc_vector, replace_d_q=True)  # acc causes dq/dt
            acc_vector = lambdify(self._v, acc_vector, modules="jax", cse=True, docstring_limit=2)
            acc_vector = self._precompile_fn(acc_vector, self.default_inputs, "FK_ddot", skip_export=True)

    def _precompile_fn(self, function: Callable, inputs: List[str], name: str, jacobian: bool = False, skip_export: bool = True) -> None:
        """Precompile a function using JAX's JIT for faster execution.
        
        Parameters
        ----------
        function : Callable
            The function to precompile
        inputs : List[str]
            Input specification for the function
        name : str
            Name for storing the compiled function
        jacobian : bool, optional
            Whether to also compile the Jacobian, by default False
        skip_export : bool, optional
            Whether to skip JAX export, by default True
            
        Notes
        -----
        Uses serialization/deserialization to avoid JAX caching issues.
        This approach doesn't seem slower than normal jax.jit.
        """

        def _jit_function_template(function_: Callable, input_names: Tuple[str, ...] = ("model",)) -> Callable:
            def wrapped(states: Any, constants: Any) -> Any:
                selected = tuple(getattr(states, key) for key in input_names) + tuple(
                    getattr(constants, key) for key in input_names
                )
                flat_inputs = jnp.concatenate(tree_util.tree_leaves(selected))
                return function_(*flat_inputs)

            return jax.jit(wrapped)

        jit_function = _jit_function_template(function)
        # Cause jit compilation
        jit_function(self.default_inputs.states, self.default_inputs.constants)

        if skip_export:
            if jacobian:
                jit_function = jax.jit(jax.jacobian(jit_function))
            else:
                jit_function = jax.jit(jit_function)
            self.run[name] = jit_function
            return

    def _replace_dyn(self, function: Callable, replace_d_q: bool = False) -> Callable:
        """
        Replace the dynamicsymbols in the function with the corresponding v_ states.
        This is needed to get the correct output from the function.
        """
        # Get rid of speeds first if needed
        if replace_d_q:
            in_ = [self._dynamic[i].diff() for i in _slice(self.coordinates)]
            out_ = [self._v[i] for i in _slice(self.speeds)]
            function = function.xreplace(dict(zip(in_, out_)))

        # First get rid of the accelerations
        in_ = [self._dynamic[i].diff() for i in _slice(self.speeds)]
        out_ = [self._v[i] for i in _slice(self.accs)]
        function = function.xreplace(dict(zip(in_, out_)))
        # This is not 100% clean: we assume that coordinates is always first (which is true for now)
        in_ = [self._dynamic[i] for i in range(self.coordinates["n"] + self.speeds["n"])]
        out_ = [self._v[i] for i in range(self.coordinates["n"] + self.speeds["n"])]
        function = function.xreplace(dict(zip(in_, out_)))

        return function

    def _register_contact_model(self, contact_model: Any, skip_export: bool = True) -> None:
        """
        Register the forward function of the contact model as a function in the run dictionary and precompile it.
        """
        self.default_inputs = self.default_inputs.replace_vector(
            "states", "gc_model", np.zeros(contact_model.get_n_states())
        )
        self.default_inputs = self.default_inputs.replace_vector(
            "constants", "gc_model", np.zeros(contact_model.get_n_constants())
        )
        lambda_func = partial(contact_model.forward, model=self)
        self.run["gc_model_jacobian"] = jax.jit(jax.jacobian(lambda_func))
        self.run["gc_model"] = jax.jit(lambda_func)

    def _register_actuator_model(self, actuator_model: Any) -> None:
        """
        TODO: Refactor the _register functions, they are almost identical
        """
        self.default_inputs = self.default_inputs.replace_vector(
            "states", "actuator_model", np.zeros(actuator_model.get_n_states())
        )
        self.default_inputs = self.default_inputs.replace_vector(
            "constants", "actuator_model", np.zeros(actuator_model.get_n_constants())
        )

        actuator_function = actuator_model.forward

        lambda_func = lambda states, constants, model: (
            actuator_function(states, constants, model) + self.passive_actuators.forward(states, constants, model)
        )[self.forces["combined_idx"]]
        lambda_func = partial(lambda_func, model=self)
        self.run["actuator_model"] = jax.jit(lambda_func)
        self.run["actuator_model_jacobian"] = jax.jit(jax.jacobian(lambda_func))

    def _create_variable_dataframe(self) -> None:
        """
        Create a dataframe with all variables in the model.
        """
        df = pd.DataFrame(columns=["type", "name", "x0", "xmin", "xmax"])
        for i, name in enumerate(self.state_vector):
            type = "state"
            x0 = self.default_values[i]
            xmin = -3.14  # -np.inf
            xmax = 3.14  # np.inf
            if name.startswith("q_"):
                # Read the min / max values from the joint limits
                j_names = [j["name"] for j in self.dicts["joints"]]
                curr_joint = self.dicts["joints"][j_names.index(name[2:])]
                xmin = curr_joint["range"][0] - np.deg2rad(15)  # Allow for some margin
                xmax = curr_joint["range"][1] + np.deg2rad(15)
            elif name.startswith("qd_"):
                xmin = -30  # From Bio-Sim-Toolbox
                xmax = 30
            elif name.startswith("qdd_"):
                xmin = -300
                xmax = 300
            elif name.startswith("f_"):
                # External forces, up to 100 kN seems reasonable
                xmin = -100000
                xmax = 100000
            elif name.startswith("m_"):
                xmin = -10000
                xmax = 10000
            elif name.startswith("M_"):
                # Joint moments, up to 1000 Nm seems reasonable
                # Might need adjustment later, hardcoding isn't great
                xmin = -1000
                xmax = 1000

            # @todo: parse limits and find reasonable limits
            df.loc[len(df)] = [type, name, x0, xmin, xmax]
        for i, name in enumerate(self.constants):
            type = "constant"
            x0 = self.default_values[i + self.n_states]
            xmin = -np.inf
            xmax = np.inf
            df.loc[len(df)] = [type, name, x0, xmin, xmax]
        self.variables = df

    def _get_hash(self) -> str:
        # Create a string of all model state names (Is that really enough?)
        all_state_names = self.state_vector + self.constants
        all_state_names = "".join(all_state_names)
        # Create a hash of the string
        return hashlib.sha256(all_state_names.encode()).hexdigest()


def _slice(dictionary: dict[str, Any]) -> slice:
    """
    Slice the state vector according to the dictionary
    To make accesing states / v_ easier
    """
    return np.arange(dictionary["idx"], dictionary["idx"] + dictionary["n"])


def _to_sympy_vector(values: List[Any], reference_frame: ReferenceFrame) -> Any:
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
    return values[0] * reference_frame.x + values[1] * reference_frame.y + values[2] * reference_frame.z


def load_model(model_file: str, force_rebuild: bool = False) -> 'BiosymModel':
    """Load a model from a file with caching support.
    
    Parameters
    ----------
    model_file : str
        Path to model file (.xml, .osim, or .yaml format)
    force_rebuild : bool, optional
        If True, rebuild model even if cached version exists, by default False
        
    Returns
    -------
    BiosymModel
        Loaded model instance
        
    Notes
    -----
    Uses hash-based caching to avoid recompiling identical models.
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


def clear_caches() -> None:
    """Clear the caches for JAX and the model.
    
    Removes all cached compiled functions and model files to force
    recompilation on next use. Useful for development and debugging.
    """
    # Clear the JAX cache
    print("Clearing the JAX cache...")
    if os.path.exists(_cachedir):
        for file in os.listdir(_cachedir):
            os.remove(os.path.join(_cachedir, file))
    print("Done.")
    # Clear the model cache
    assert (
        input("Are you sure you want to clear the model cache? This deletes all compiled sympy models (y/n)") == "y"
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
    # model_file = "tests/models/pendulum.xml"
    model_file = "tests/models/gait2d_torque/gait2d_torque.yaml"
    model = load_model(model_file, True)
    print(model.dicts["joints"])
    print(f"Reloading model in {time.time() - start} seconds")
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
    print(f"1 jacobian in in {time.time() - start} seconds (with reimporting)")
    start = time.time()
    import timeit

    a = timeit.timeit(lambda: model.run["jacobian"](states, constants), number=10000)
    print(f"jacobian runs in {a / 10000} seconds")

    start = time.time()
    for _ in range(1):
        model.run["confun"](states, constants)
    print(f"1 confun in in {time.time() - start} seconds (with reimporting)")
    start = time.time()
    import timeit

    a = np.ones(model._nv)
    a = timeit.timeit(lambda: model.run["confun"](states, constants), number=10000)
    print(f"confun runs in {a / 10000} seconds")

    model.run["gc_model"](states, constants)
    a = timeit.timeit(lambda: model.run["gc_model"](states, constants), number=10000)
    print(f"gc_model runs in {a / 10000} seconds")

    a = timeit.timeit(lambda: model.run["gc_model_jacobian"](states, constants), number=10000)
    print(f"gc_model jacobian runs in {a / 10000} seconds")

    print("confun shape:", model.run["confun"](states, constants).shape)
    print("jacobian shape:", model.run["jacobian"](states, constants)["model"].shape)
    print("gc_model shape:", model.run["gc_model"](states, constants)[1].shape)
    print(
        "gc_model jacobian shape:",
        model.run["gc_model_jacobian"](states, constants)[0]["model"].shape,
    )
