"""
BiosymModel: Core biomechanical modeling functionality.

This module provides the main BiosymModel class for creating, manipulating,
and simulating biomechanical models from various input formats (XML, OSIM, YAML).
The module handles model parsing, symbolic equation generation, JAX compilation,
and provides interfaces for optimal control problems.
"""

import os
import time
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

_cachedir = os.path.expanduser("~/.biosym/jax_cache")
_model_cache = os.path.expanduser("~/.biosym/")
os.environ["JAX_COMPILATION_CACHE_DIR"] = _cachedir  # This needs to happen before importing jax
os.environ["jax_persistent_cache_min_compile_time_secs".upper()] = "0.01"
os.makedirs((_cachedir), exist_ok=True)

import contextlib
import hashlib
from functools import partial

import cloudpickle
import jax
import jax.export
import jax.numpy as jnp
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

from biosym.model.joints import *
from biosym.model.joints.joint_models import *
from biosym.model.actuators import *
from biosym.model.actuators.actuator_models import *
from biosym.model.contact import *
from biosym.model.contact.contact_geometries import *
from biosym.model.contact.contact_models import *
from biosym.model.parsers import *
from biosym.model.parsers.base_parser import BaseParser
from biosym.utils import states as states_module
from biosym.utils import rnea, aba

if TYPE_CHECKING:
    from biosym.model.parsers.base_parser import BaseParser


class _ModelProperties(NamedTuple):
    names: list[str]
    symbols: list
    n: int

    def __str__(self) -> str:
        return f"Properties Field: {self.names}\nNumber of items: {self.n}\nSymbols: {self.symbols}"

    def __repr__(self) -> str:
        return self.__str__()


class _ForceProperties(NamedTuple):
    names: list[str]
    symbols: list
    n: int
    passive_idx: jnp.ndarray
    active_idx: jnp.ndarray
    combined_idx: jnp.ndarray

    def __str__(self) -> str:
        return f"Force Properties: {self.names}\nNumber of items: {self.n}\nSymbols: {self.symbols}\nPassive indices: {self.passive_idx}\nActive indices: {self.active_idx}\nCombined indices: {self.combined_idx}"

    def __repr__(self) -> str:
        return self.__str__()


class BiosymModel:
    """Biomechanical model class for biosym.

    This class provides functionality to load, save, and manipulate biomechanical
    models from various formats. It handles symbolic mechanics, JAX compilation,
    and provides interfaces for optimization and simulation.
    """

    def __init__(self, definition_file: str, get_hash: bool = False, compile_eom: bool = True) -> None:
        """Initialize a BiosymModel from a definition file.

        Parameters
        ----------
        definition_file : str
            Path to model definition file (.xml, .osim, or .yaml)
        get_hash : bool, optional
            If True, only compute model hash without full initialization, by default False
        compile_eom : bool, optional
            If True, compile symbolic equations of motion and JAX functions, by default True

        Raises
        ------
        ValueError
            If definition_file format is not supported
        NotImplementedError
            If trying to load not yet supported files 
        """
        definition_file = os.path.abspath(os.path.expanduser(definition_file))
        self.compile_eom = compile_eom
        self.definition_file_yaml = None
        self.cfg = None
        cfg = None
        if definition_file.endswith(".yaml"):
            self.definition_file_yaml = definition_file
            # Load the yaml file and check for the model name
            with open(definition_file) as f:
                cfg = yaml.safe_load(f)
            self.cfg = cfg
            # Replace the definition file with the model name
            definition_file = os.path.join(os.path.dirname(definition_file), cfg["model"]["name"])
        self.definition_file = definition_file

        if definition_file.endswith(".xml"):
            parser = mujoco_parser.MujocoParser(definition_file)
            if parser.has_actuators():
                self.actuators = actuator_parser.get_from_xml(parser.get_actuators())
                parser.actuators = self.actuators.get_actuators()
            if parser.has_contact_model():
                self.gc_model = contact_parser.get_from_xml(parser.get_contact_model())
                parser.external_forces_bodies = self.gc_model.get_bodies()

        # Parsing the OpenSim models
        elif definition_file.endswith(".osim"):
            parser = osim_parser.OsimParser(definition_file)
            # Translate OpenSim to Biosym standard
            self._translate_osim_to_biosym(parser)

        else:
            raise ValueError("Model definition file must be in .xml, .osim, or .yaml format.")

        # Check for additional parameters if there is a .yaml file
        if cfg is not None:
            if "additional_parameters" in cfg["model"]:
                if "ground_contact" in cfg["model"]["additional_parameters"]:
                    gc_model_file = os.path.join(
                        os.path.dirname(definition_file),
                        cfg["model"]["additional_parameters"]["ground_contact"]["file"],
                    )
                    self.gc_model = contact_parser.get_from_xml(gc_model_file)
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
                    self.actuators = actuator_parser.get_from_xml(actuator_model_file, joint_names=[j["name"] for j in parser.get_joints()])
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
        self._create_fk(True)

        # Future work, tbd: (These should be disabled or enabled by a flag in the config file); so that we don't need to compile everything every time
        if hasattr(self, "gc_model"):
            self.gc_model.process_eom(
                self, body_weight=sum(self.default_constants.mass)
            )  # If the GC model needs to add extra equations of motion, it will do so here
            self._register_contact_model(self.gc_model)
        if hasattr(self, "actuators"):
            self.actuators.process_eom(self)
            self._register_actuator_model(self.actuators)

        if self.compile_eom:
            self._create_eom()
            self._create_jax_eom()

        self._create_variable_dataframe()

        self._register_aba_rnea()

        # self._create_FK(parser)
        # self._create_IMU(parser) ....

    def _translate_osim_to_biosym(self, parser):
        """Translates raw OpenSim parser output into the Biosym architecture."""

        _JOINT_BUILDERS = {
            "PinJoint":    pin_joint.PinJoint,
            "PlanarJoint": planar_joint.PlanarJoint,
            "WeldJoint":   weld_joint.WeldJoint,
            # NOTE: register any newly implemented joint type here.
            }

        # Translate bodies
        # Shift body origins to joint center, since the body origins in OpenSim models are not placed at the joints
        for body in parser.get_bodies():
            parser.convert_body_frame(body)

        # Translate Joints
        # Flatten the OSIM joints for consistency with the biosym pipeline and replace parser.data["joints"]
        new_joints = []
        for joint in parser.get_joints():
            joint_type = joint.get("type")
            builder = _JOINT_BUILDERS.get(joint_type)
            if builder is None:
                raise NotImplementedError(
                    f"Joint '{joint.get('name')}' of type '{joint_type}' is not yet "
                    f"supported by the translator. Register it in _JOINT_BUILDERS."
                )
            new_joints.extend(builder(joint).flat_joints)
        parser.data["joints"] = new_joints

        # Translate Internal Forces (Actuators, Muscles, etc.)
        if parser.get_n_internal_forces() > 0:
            actuators = actuator_parser.get_from_osim(
                parser.data["forces"],
                parser.data["joints"],
                joint_names=[j["name"] for j in parser.get_joints()],
            )
            if actuators is not None:
                self.actuators = actuators
            else:
                print("No actuators found in the OpenSim force set "
                      "(only contact or other non-actuator forces present).")

        # Translate External Forces (Contact Geometries + Forces)
        # Build the contact model from the parsed OSIM contact geometries and
        # force set. This mirrors what the MuJoCo path does with 
        # get_from_xml: produce a gc_model object, then tell the
        # parser which bodies receive external forces so _create_dictionaries
        # sizes the force/torque slots.
        if parser.get_n_external_forces() > 0:
            gc_model = contact_parser.get_from_osim(parser.data["contact_geometries"], parser.data["forces"],)
            if gc_model is not None:
                self.gc_model = gc_model
                parser.external_forces_bodies = gc_model.get_bodies()

        print("Successfully translated OpenSim model to Biosym standard.")
         
        

    def _create_dictionaries(self, parser: "BaseParser") -> None:
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
        markers = [s for s in all_sites if not s.get("name").startswith("vis_")]
        sites = [s for s in all_sites if s.get("name").startswith("vis_")]

        self.dicts = {
            "bodies": parser.get_bodies(),
            "joints": parser.get_joints(),
            "sites": sites,  # keep only torso as a site
            "markers": markers,  # all other xml sites become markers
        }

        # Create overviews of all states
        n_dof = parser.get_n_joints()
        self.coordinates = _ModelProperties(
            names=[f"q_{joint['name']}" for joint in parser.get_joints()],
            symbols=Matrix(symbols([f"q_{joint['name']}" for joint in parser.get_joints()])),
            n=n_dof,
        )
        self.speeds = _ModelProperties(
            names=[f"qd_{joint['name']}" for joint in parser.get_joints()],
            symbols=Matrix(symbols([f"qd_{joint['name']}" for joint in parser.get_joints()])),
            n=n_dof,
        )
        self.accs = _ModelProperties(
            names=[f"qdd_{joint['name']}" for joint in parser.get_joints()],
            symbols=Matrix(symbols([f"qdd_{joint['name']}" for joint in parser.get_joints()])),
            n=n_dof,
        )
        # Passive actuators need to be defined here
        self.passive_actuators = passive_torques.PassiveTorques(self.dicts["joints"])
        passive_actuated_joints = self.passive_actuators.get_actuated_joints()

        # Gaurding against no-active-actuator case
        if hasattr(self, "actuators"):
            active_actuated_joints = list(self.actuators.get_actuated_joints())
        else:
            active_actuated_joints = []
        
        actuated_joints = list(dict.fromkeys(active_actuated_joints + passive_actuated_joints))
        all_joints = [joint["name"] for joint in self.dicts["joints"]]
        self.tau = _ForceProperties(
            names=[f"t_{joint}" for joint in all_joints if joint in actuated_joints],
            n=len(actuated_joints),
            symbols=Matrix(symbols([f"tau_{joint}" for joint in all_joints if joint in actuated_joints])),
            passive_idx=jnp.array([i for i, j in enumerate(all_joints) if j in passive_actuated_joints]),
            active_idx=jnp.array([i for i, j in enumerate(all_joints) if j in active_actuated_joints]),
            combined_idx=jnp.array([i for i, j in enumerate(all_joints) if j in actuated_joints]),
        )
        # The first representation of external forces is a list of bodies, where the forces can be applied
        n_ext_forces = parser.get_n_external_forces()
        self.ext_forces = _ModelProperties(
            names=[f"f_{force}_{dim}" for force in parser.get_external_forces_bodies() for dim in ["x", "y", "z"]],
            symbols=Matrix(
                symbols(
                    [f"f_{force}_{dim}" for force in parser.get_external_forces_bodies() for dim in ["x", "y", "z"]]
                )
            ),
            n=n_ext_forces,
        )

        # And torques
        self.ext_torques = _ModelProperties(
            names=[f"m_{force}_{dim}" for force in parser.get_external_forces_bodies() for dim in ["x", "y", "z"]],
            symbols=Matrix(
                symbols(
                    [f"m_{force}_{dim}" for force in parser.get_external_forces_bodies() for dim in ["x", "y", "z"]]
                )
            ),
            n=n_ext_forces,
        )

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
            self.coordinates.names
            + self.speeds.names
            + self.accs.names
            + self.tau.names
            + self.ext_forces.names
            + self.ext_torques.names
        )
        self.n_states = len(self.state_vector)

        # Create dictionaries for all constants
        i = len(self.state_vector)
        self.g = _ModelProperties(
            names=["g_x", "g_y", "g_z"],
            symbols=Matrix(symbols(["g_x", "g_y", "g_z"])),
            n=3,
        )
        i += 3

        self.mass = _ModelProperties(
            names=[f"m_{body['name']}" for body in parser.get_bodies()],
            symbols=Matrix(symbols([f"m_{body['name']}" for body in parser.get_bodies()])),
            n=len(parser.get_bodies()),
        )
        i += len(parser.get_bodies())

        inertia_tensor = ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]
        self.inertia = _ModelProperties(
            names=[f"I_{body['name']}_{dim}" for body in parser.get_bodies() for dim in inertia_tensor],
            symbols=Matrix(
                symbols([f"I_{body['name']}_{dim}" for body in parser.get_bodies() for dim in inertia_tensor])
            ),
            n=len(parser.get_bodies()) * len(inertia_tensor),
        )
        i += len(parser.get_bodies()) * len(inertia_tensor)

        self.com = _ModelProperties(
            names=[f"com_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ["x", "y", "z"]],
            symbols=Matrix(
                symbols([f"com_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ["x", "y", "z"]])
            ),
            n=len(parser.get_bodies()) * 3,
        )
        i += len(parser.get_bodies()) * 3

        self.offset = _ModelProperties(
            names=[f"offset_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ["x", "y", "z"]],
            symbols=Matrix(
                symbols([f"offset_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ["x", "y", "z"]])
            ),
            n=len(parser.get_bodies()) * 3,
        )
        i += len(parser.get_bodies()) * 3

        self.constants = self.g.names + self.mass.names + self.inertia.names + self.com.names + self.offset.names
        self.n_constants = len(self.constants)

        self.gravity = parser.get_gravity()

        # Initialize default states and constants
        self.default_states = states_module.States(
            q=np.zeros(self.coordinates.n),
            qd=np.zeros(self.speeds.n),
            qdd=np.zeros(self.accs.n),
            tau=np.zeros(self.tau.n),
            ext_forces=np.zeros(self.ext_forces.n),
            ext_torques=np.zeros(self.ext_torques.n),
        )
        self.default_constants = states_module.Constants(
            g=self.gravity,
            mass=np.zeros(self.mass.n),
            inertia=np.zeros(self.inertia.n),
            com=np.zeros(self.com.n),
            offset=np.zeros(self.offset.n),
        )

        self._symbols = Matrix(
            [
                p.symbols
                for p in [
                    self.coordinates,
                    self.speeds,
                    self.accs,
                    self.tau,
                    self.ext_forces,
                    self.ext_torques,
                    self.g,
                    self.mass,
                    self.inertia,
                    self.com,
                    self.offset,
                ]
            ]
        )

    def _set_default_values(self) -> None:
        """Create default values for the model.

        These are used to initialize the model and can be changed later.
        Sets gravity, masses, and other body parameters to their default values.
        """

        def concat_defaults(value: str) -> np.ndarray:
            """Concatenate all default values for the bodies."""
            if value == "g":
                return np.array(self.gravity)
            all_values = []
            for body in self.dicts["bodies"]:
                all_values.append(body[value])
            # return a
            return np.array(all_values).flatten()

        # Get all values that are stored as lists
        default_constants_dict = {}
        for _value_dict, value in zip(
            [self.g, self.mass, self.com, self.offset, self.inertia], ["g", "mass", "com", "body_offset", "inertia"]
        ):
            default_constants_dict[value if value != "body_offset" else "offset"] = concat_defaults(value)

        self.default_constants = self.default_constants.replace(**default_constants_dict)

    def _create_sympy_model(self) -> None:
        """Create the symbolic equations of motion (EOM) for the model.

        Creates symbolic representations using SymPy for coordinates, speeds,
        and forces. Every value is stored in the vector self._v for vectorization.
        Sets up reference frames, bodies, and mechanical constraints.
        """
        self._nv = self.n_states + self.n_constants
        # We set everything with the IndexedBase, so that it is vectorized
        # However, coordinates and speeds need to be dynamicsymbols, therefore we create a separate vector for them and merge them later
        self._dynamic_q = Matrix(self.coordinates.n, 1, lambda i, _: dynamicsymbols(f"dyn_q{i}"))
        self._dynamic_qd = Matrix(self.speeds.n, 1, lambda i, _: dynamicsymbols(f"dyn_qd{i}"))

        len(self.dicts["bodies"])

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
            *[x.diff() - y for x, y in zip(self._dynamic_q, self._dynamic_qd)],
        ]

        # Get the model topology (A tree-like to help navigating the model)
        # This might change the indexing order of the bodies --> needs to be tested
        # Use body_idx as a substitute for now to be safe
        self.topology_tree = self._create_topology_tree()

        def build_reference_frames(
            topology: list[dict[str, Any]],
            parent_frame: Optional[ReferenceFrame] = None,
            parent_origin: Optional[Point] = None,
        ) -> None:
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
                    self.offset.symbols[3 * body_idx : 3 * (body_idx + 1)],
                    parent_frame,
                )
                joint_speed = 0
                # Slide joint logic
                n_hinges = 0
                for joint in body["joints"]:
                    if joint["type"] == "slide":
                        joint_offset += (
                            _to_sympy_vector(joint["axis"], parent_frame)
                            * self._dynamic_q[self.coordinates.names.index(f"q_{joint['name']}")]
                        )
                        joint_speed += (
                            _to_sympy_vector(joint["axis"], parent_frame)
                            * self._dynamic_qd[self.speeds.names.index(f"qd_{joint['name']}")]
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
                hinge_idx = 0  # iterator for all jointss
                while hinge_idx < n_hinges:
                    joint = body["joints"][idx_h]
                    if joint["type"] == "hinge":
                        # Check if we need to add a new frame
                        symbol = self._dynamic_q[self.coordinates.names.index(f"q_{joint['name']}")]
                        symbol_dot = self._dynamic_qd[self.speeds.names.index(f"qd_{joint['name']}")]
                        joint_angle = _to_sympy_vector(joint["axis"], intermediate_frames[-1])
                        joint_angvel = _to_sympy_vector(joint["axis"], intermediate_frames[-1])
                        if hinge_idx == n_hinges - 1:
                            # I think that we need add the joints iteratively to the frame, order matters
                            body_frame.orient(intermediate_frames[-1], "Axis", (symbol, joint_angle))
                            body_frame.set_ang_vel(intermediate_frames[-1], joint_angvel * symbol_dot)
                        else:
                            new_frame = ReferenceFrame(f"{body_name}_{joint['name']}_frame_{hinge_idx}")
                            new_frame.orient(intermediate_frames[-1], "Axis", (symbol, joint_angle))
                            new_frame.set_ang_vel(intermediate_frames[-1], joint_angvel * symbol_dot)
                            intermediate_frames.append(new_frame)
                        hinge_idx += 1
                    idx_h += 1

                self.reference_frames[body_name] = body_frame
                build_reference_frames(children, body_frame, body_origin)

        build_reference_frames(self.topology_tree)

        def build_bodies(topology: list[dict[str, Any]]) -> None:
            for _idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body["name"] for body in self.dicts["bodies"]].index(body_name)
                children = node["children"]
                body_origin = self.body_origins[body_name]
                body_frame = self.reference_frames[body_name]

                # Set pos and lin_vel for the body
                mass_center_point = Point(f"{body_name}_mass_center")
                com_pos = _to_sympy_vector(
                    self.com.symbols[3 * body_idx : 3 * (body_idx + 1)],
                    body_frame,
                )
                mass_center_point.set_pos(body_origin, com_pos)
                mass_center_point.v2pt_theory(body_origin, self.ground_frame, body_frame)
                self.mass_centers[body_name] = mass_center_point

                # set inertia tensor
                inertia_tensor = self.inertia.symbols[6 * body_idx : 6 * (body_idx + 1)]
                body_inertia = Inertia.from_inertia_scalars(mass_center_point, body_frame, *inertia_tensor)

                # Create the body
                body = RigidBody(
                    body_name,
                    mass_center_point,
                    body_frame,
                    self.mass.symbols[body_idx],
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
        for _bodyname, rigid_body in self.rigid_bodies.items():
            gravity_f = rigid_body.mass * (
                self.ground_frame.x * self.g.symbols[0]
                + self.ground_frame.y * self.g.symbols[1]
                + self.ground_frame.z * self.g.symbols[2]
            )
            self.loads.append((rigid_body.masscenter, gravity_f))

        # Add internal forces
        for i, raw_joint_name in enumerate(self.tau.names):
            joint_name = "_".join(raw_joint_name.split("_")[1:])
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
            force_vector = _to_sympy_vector(
                [self.tau.symbols[i] * axis[j] for j in range(3)],
                parent_frame if parent_frame is not None else self.ground_frame,
            )
            force_body = (child_frame, force_vector)
            force_parent_body = (parent_frame, -force_vector)
            self.loads.append(force_body)
            self.loads.append(force_parent_body)

        # Add external forces - double check if this is correct
        # We are using body.origin to apply the force, but it could also be applied to the mass center or an arbitrary point - that is tbd
        for idx, force in enumerate(self.ext_forces.names):
            # Only parse every 3rd entry, because they are x,y,z
            if idx % 3 != 0:
                continue
            body_name = "_".join(force.split("_")[1:-1])
            assert body_name in [body["name"] for body in self.dicts["bodies"]], (
                f"Body {body_name} not found in the model, but has an external force defined."
            )
            force_vector = _to_sympy_vector(self.ext_forces.symbols[idx : idx + 3], self.ground_frame)
            # print(f"Force vector: {force_vector} at {self.body_origins[body_name]}")
            force_body = (self.body_origins[body_name], force_vector)
            self.loads.append(force_body)

        # Add external torques - also double check if this is correct
        for idx, torque in enumerate(self.ext_torques.names):
            if idx % 3 != 0:
                continue
            body_name = "_".join(torque.split("_")[1:-1])
            torque_vector = _to_sympy_vector(self.ext_torques.symbols[idx : idx + 3], self.ground_frame)
            torque_body = (self.reference_frames[body_name], torque_vector)
            self.loads.append(torque_body)

        # Define separate SymPy vector symbols for decoupled JAX lambdification
        len(self.dicts["bodies"])

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
            q_ind=list(self._dynamic_q),
            u_ind=list(self._dynamic_qd),
            kd_eqs=self.kd_eqs,
        )
        self.fr, self.frstar = km.kanes_equations(list(self.rigid_bodies.values()), self.loads)
        self.kane = km
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
        a = time.time()
        self.confun = lambdify(self._symbols, self.eom, modules="jax", cse=True, docstring_limit=2)

        self._precompile_fn(
            self.confun, (self.default_states, self.default_constants), "kane_jacobian", jacobian=True
        )

        self._precompile_fn(self.confun, (self.default_states, self.default_constants), "kane")


        mm_replaced = self._replace_dyn(self.kane.mass_matrix)
        self.mass_matrix = lambdify(
            self._symbols,
            mm_replaced,
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.run["mass_matrix_uncompiled"] = self.mass_matrix
        self._precompile_fn(self.mass_matrix, (self.default_states, self.default_constants), "mass_matrix")
        self._precompile_fn(self.mass_matrix, (self.default_states, self.default_constants), "mass_matrix_jacobian", jacobian=True)


        forcing_replaced = self._replace_dyn(self.kane.forcing)
        self.forcing = lambdify(
            self._symbols,
            forcing_replaced,
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.run["forcing_uncompiled"] = self.forcing
        self._precompile_fn(self.forcing, (self.default_states, self.default_constants), "forcing")
        self._precompile_fn(self.forcing, (self.default_states, self.default_constants), "forcing_jacobian", jacobian=True)


    def _create_fk(self, get_fk_dot: bool = True) -> None:
        """Create forward kinematics (FK) functions for the model.

        Creates symbolic and compiled functions for computing body positions
        and velocities in the global reference frame.

        Parameters
        ----------
        get_fk_dot : bool, optional
            Whether to also compute velocity kinematics, by default True

        Notes
        -----
        Currently returns positions of body_origins in the global frame.
        FK for markers etc. should be added in different functions for max speed.
        """
        self.positions = list(self.body_origins.keys())
        pos_vector = []
        pos_vector_markers = []
        for _, point in self.body_origins.items():
            pos_vector.append(
                [
                    point.pos_from(self.origin).dot(frame_dim)  # position of bidy origins in ground (global) frame
                    for frame_dim in [
                        self.ground_frame.x,
                        self.ground_frame.y,
                        self.ground_frame.z,
                    ]
                ]
            )
        pos_vector_ = Matrix(pos_vector)
        pos_vector_ = self._replace_dyn(pos_vector_)
        pos_vector_ = lambdify(self._symbols, pos_vector_, modules="jax", cse=True, docstring_limit=2)
        self.run["FK_uncompiled"] = pos_vector_
        pos_vector_ = self._precompile_fn(pos_vector_, (self.default_states, self.default_constants), "FK")

        # Visualization FK: gather markers (if any)
        if self.dicts.get("markers") is not None:
            for marker_pt in self.markers.values():
                pos_vector_markers.append(
                    [
                        marker_pt.pos_from(self.origin).dot(frame_dim)
                        for frame_dim in (self.ground_frame.x, self.ground_frame.y, self.ground_frame.z)
                    ]
                )

            pos_vector_marker_ = Matrix(pos_vector_markers)
            pos_vector_marker_ = self._replace_dyn(pos_vector_marker_)
            pos_vector_marker_ = lambdify(self._symbols, pos_vector_marker_, modules="jax", cse=True, docstring_limit=2)
            self.run["FK_marker_uncompiled"] = pos_vector_marker_
            # store compiled/jitted visualization function
            pos_vector_marker_ = self._precompile_fn(
                pos_vector_marker_, (self.default_states, self.default_constants), "FK_marker"
            )

        # Visualization FK: bodies + markers + sites (when available)
        pos_vector_sites: list[list] = []
        if self.dicts.get("sites") is not None:
            for site_pt in self.sites.values():
                pos_vector_sites.append(
                    [
                        site_pt.pos_from(self.origin).dot(frame_dim)
                        for frame_dim in (self.ground_frame.x, self.ground_frame.y, self.ground_frame.z)
                    ]
                )

        # Assemble visualization rows: start from bodies, then markers, then sites
        pos_vector_vis_rows = list(pos_vector)  # bodies
        if pos_vector_markers:
            pos_vector_vis_rows.extend(pos_vector_markers)
        if pos_vector_sites:
            pos_vector_vis_rows.extend(pos_vector_sites)

        if len(pos_vector_vis_rows) > len(pos_vector):
            pos_vector_vis = Matrix(pos_vector_vis_rows)
            pos_vector_vis = self._replace_dyn(pos_vector_vis)
            pos_vector_vis = lambdify(self._symbols, pos_vector_vis, modules="jax", cse=True, docstring_limit=2)
            self.run["FK_vis_uncompiled"] = pos_vector_vis
            # store compiled/jitted visualization function
            pos_vector_vis = self._precompile_fn(
                pos_vector_vis, (self.default_states, self.default_constants), "FK_vis"
            )
        else:
            # No additional rows beyond bodies; fall back to body-only FK
            self.run["FK_vis_uncompiled"] = self.run["FK_uncompiled"]
            self.run["FK_vis"] = self.run["FK"]

        if get_fk_dot:
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
            vel_vector = lambdify(self._symbols, vel_vector, modules="jax", cse=True, docstring_limit=2)
            self.run["FK_dot_uncompiled"] = vel_vector
            vel_vector = self._precompile_fn(vel_vector, (self.default_states, self.default_constants), "FK_dot")

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
            acc_vector = self._replace_dyn(acc_vector)  # acc causes dq/dt
            acc_vector = lambdify(self._symbols, acc_vector, modules="jax", cse=True, docstring_limit=2)
            self.run["FK_ddot_uncompiled"] = acc_vector
            acc_vector = self._precompile_fn(
                acc_vector, (self.default_states, self.default_constants), "FK_ddot"
            )

    def _precompile_fn(
        self, function: Callable, _inputs: tuple[str], name: str, jacobian: bool = False, is_jax_fn: bool = False
    ) -> None:
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


        Notes
        -----
        Uses serialization/deserialization to avoid JAX caching issues.
        This approach doesn't seem slower than normal jax.jit.
        """

        def _jit_function_template(function_: Callable, jacobian:bool=False, is_jax_fn: bool = False) -> Callable:
            def wrapped(states: Any, constants: Any) -> Any:
                states = states.filter('model')
                constants = constants.filter('model')
                if not is_jax_fn:
                    f = lambda states, constants: function_(*states.flatten(), *constants.flatten())
                else:
                    f = function_
                
                if sum(states.shape()) > 1:
                    if not jacobian:
                        n_dims = 2
                        vmapable = lambda states_, constants_: f(states_, constants_)
                    else:
                        n_dims = 3
                        vmapable = lambda states_, constants_: jax.jacobian(f)(states_, constants_)
                    if len(states.shape()) > 2:
                        states_shape = states.shape
                        reshaped_states = states.reshape(-1, states_shape[-1])
                        res = jax.vmap(vmapable, in_axes=(0, None))(reshaped_states, constants)
                        res = res.reshape(*states_shape[:-1], *res.shape[-n_dims:])
                    else:
                        res = jax.vmap(vmapable, in_axes=(0, None))(states, constants)
                else:
                    if not jacobian:
                        res = f(states, constants)
                    else:
                        res = jax.jacobian(f)(states, constants)
                return res.squeeze()

            return jax.jit(wrapped)

        self.run[name] = _jit_function_template(function, jacobian=jacobian, is_jax_fn=is_jax_fn)

    def _replace_dyn(self, function: Callable, _replace_d_q: bool = False) -> Callable:
        """
        Replace the dynamicsymbols in the function with the corresponding v_ states.
        This is needed to get the correct output from the function.
        """
        # Replace .diff first, then the others
        function = function.xreplace(dict(zip(self._dynamic_q.diff(), self.speeds.symbols)))
        function = function.xreplace(dict(zip(self._dynamic_qd.diff(), self.accs.symbols)))
        function = function.xreplace(dict(zip(self._dynamic_q, self.coordinates.symbols)))
        function = function.xreplace(dict(zip(self._dynamic_qd, self.speeds.symbols)))
        return function

    def _register_contact_model(self, contact_model: Any, _skip_export: bool = True) -> None:
        """Register the forward function of the contact model as a function in the run dictionary and precompile it."""
        self.default_states = self.default_states.replace(gc_model=np.zeros(contact_model.get_n_states()))
        self.default_constants = self.default_constants.replace(gc_model=np.zeros(contact_model.get_n_constants()))
        lambda_func = partial(contact_model.forward, model=self)
        self.run["gc_model_jacobian"] = jax.jit(jax.jacobian(lambda_func))
        self.run["gc_model"] = jax.jit(lambda_func)
        self.contact_model = contact_model

    def _register_actuator_model(self, actuator_model: Any) -> None:
        """TODO: Refactor the _register functions, they are almost identical."""
        self.default_states = self.default_states.replace(actuator_model=np.zeros(actuator_model.get_n_states()))
        self.default_constants = self.default_constants.replace(
            actuator_model=np.zeros(actuator_model.get_n_constants())
        )

        actuator_function = actuator_model.forward

        def lambda_func(states: Any, constants: Any, model: Any) -> Any:
            f = actuator_function(states, constants, model) + self.passive_actuators.forward(states, constants, model)
            return f[self.tau.combined_idx] if f.ndim == 1 else f[:, self.tau.combined_idx]

        self.actuator_model = actuator_model
        lambda_func = partial(lambda_func, model=self)
        self.run["actuator_model"] = jax.jit(lambda_func)
        self.run["actuator_model_jacobian"] = jax.jit(jax.jacobian(lambda_func))

    def _register_aba_rnea(self) -> None:
        """Precompile the ABA (Articulated Body Algorithm) and RNEA (Recursive Newton-Euler Algorithm) functions."""
        # The idea is to have aba and rnea inside the model's run dictionary
        rnea_fun = rnea.get_rnea_biosym(self)
        aba_fun = aba.get_aba_biosym(self)
        self._precompile_fn(rnea_fun, (self.default_states, self.default_constants), "rnea", is_jax_fn=True)
        self._precompile_fn(aba_fun, (self.default_states, self.default_constants), "aba", is_jax_fn=True)
        self._precompile_fn(rnea_fun, (self.default_states, self.default_constants), "rnea_jacobian", jacobian=True, is_jax_fn=True)
        self._precompile_fn(aba_fun, (self.default_states, self.default_constants), "aba_jacobian", jacobian=True, is_jax_fn=True)

    def _create_variable_dataframe(self) -> None:
        """Create a dataframe with all variables in the model."""
        df = pd.DataFrame(columns=["type", "name", "x0", "xmin", "xmax"])
        for name in self.state_vector:
            var_type = "state"
            xmin = -3.14  # -np.inf
            xmax = 3.14  # np.inf
            if name.startswith("q_"):
                # Read the min / max values from the joint limits
                j_names = [j["name"] for j in self.dicts["joints"]]
                idx = self.coordinates.names.index(name)
                curr_joint = self.dicts["joints"][j_names.index(name[2:])]
                xmin = curr_joint["range"][0] - np.deg2rad(15)  # Allow for some margin
                xmax = curr_joint["range"][1] + np.deg2rad(15)
                x0 = self.default_states.q[idx]
            elif name.startswith("qd_"):
                idx = self.speeds.names.index(name)
                xmin = -30  # From Bio-Sim-Toolbox
                xmax = 30
                x0 = self.default_states.qd[idx]
            elif name.startswith("qdd_"):
                idx = self.accs.names.index(name)
                xmin = -300
                xmax = 300
                x0 = self.default_states.qdd[idx]
            elif name.startswith("f_"):
                # External forces, up to 100 kN seems reasonable
                idx = self.ext_forces.names.index(name)
                xmin = -100000
                xmax = 100000
                x0 = self.default_states.ext_forces[idx]
            elif name.startswith("m_"):
                idx = self.ext_torques.names.index(name)
                xmin = -10000
                xmax = 10000
                x0 = self.default_states.ext_torques[idx]
            elif name.startswith("t_"):
                # Joint moments, up to 1000 Nm seems reasonable
                # Might need adjustment later, hardcoding isn't great
                idx = self.tau.names.index(name)
                xmin = -1000
                xmax = 1000
                x0 = self.default_states.tau[idx]

            # @todo: parse limits and find reasonable limits
            df.loc[len(df)] = [var_type, name, x0, xmin, xmax]

        x0 = self.default_constants.flatten()
        for i, name in enumerate(self.constants):
            var_type = "constant"
            xmin = -np.inf
            xmax = np.inf
            df.loc[len(df)] = [var_type, name, x0[i], xmin, xmax]
        self.variables = df

    def _get_hash(self) -> str:
        hasher = hashlib.sha256()

        # 1. Hash the names of state_vector and constants
        all_state_names = "".join(self.state_vector + self.constants)
        hasher.update(all_state_names.encode())

        # 2. Hash files if self.definition_file is set
        def_file = getattr(self, "definition_file_yaml", None) or getattr(self, "definition_file", None)
        if def_file and os.path.exists(def_file):
            with open(def_file, "rb") as f:
                hasher.update(f.read())

        # If we have the yaml def file, also hash the xml definition_file
        xml_file = getattr(self, "definition_file", None)
        if xml_file and xml_file != def_file and os.path.exists(xml_file):
            with open(xml_file, "rb") as f:
                hasher.update(f.read())

        # Hash other referenced files
        cfg = getattr(self, "cfg", None)
        if cfg is not None and def_file:
            model_dir = os.path.dirname(def_file)
            try:
                gc_file = cfg["model"]["additional_parameters"]["ground_contact"]["file"]
                gc_path = os.path.join(model_dir, gc_file)
                if os.path.exists(gc_path):
                    with open(gc_path, "rb") as f:
                        hasher.update(f.read())
            except KeyError:
                pass
            try:
                act_file = cfg["model"]["additional_parameters"]["actuators"]["file"]
                act_path = os.path.join(model_dir, act_file)
                if os.path.exists(act_path):
                    with open(act_path, "rb") as f:
                        hasher.update(f.read())
            except KeyError:
                pass

        # 3. Hash all python files in the biosym package to invalidate cache on any code change
        try:
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            py_files = []
            for root, _, files in os.walk(package_dir):
                for file in files:
                    if file.endswith(".py"):
                        py_files.append(os.path.join(root, file))
            # Sort to make hashing deterministic
            py_files.sort()
            for py_file in py_files:
                with open(py_file, "rb") as f:
                    hasher.update(f.read())
        except Exception:
            pass

        return hasher.hexdigest()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Keep only SymPy functions/symbols and configuration but not JAX JIT compiled functions in self.run
        # We clear the compiled JAX functions to avoid serializing brittle hardware-specific tracer objects.
        safe_run = {}
        for k, v in state.get("run", {}).items():
            if k not in [
                "FK",
                "FK_marker",
                "FK_vis",
                "FK_ddot",
                "FK_dot",
                "jacobian",
                "confun",
                "mass_matrix",
                "forcing",
                "gc_model",
                "gc_model_jacobian",
                "actuator_model",
                "actuator_model_jacobian",
            ]:
                safe_run[k] = v
        state["run"] = safe_run
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-precompile/re-jit all dynamic JAX functions upon unpickling to ensure fresh JAX session tracers.
        self.run = getattr(self, "run", {})

        # 1. FK functions
        if "FK_uncompiled" in self.run:
            self._precompile_fn(self.run["FK_uncompiled"], (self.default_states, self.default_constants), "FK")
        if "FK_marker_uncompiled" in self.run:
            self._precompile_fn(self.run["FK_marker_uncompiled"], (self.default_states, self.default_constants), "FK_marker")
        if "FK_vis_uncompiled" in self.run:
            self._precompile_fn(self.run["FK_vis_uncompiled"], (self.default_states, self.default_constants), "FK_vis")
        elif "FK" in self.run:
            self.run["FK_vis"] = self.run["FK"]
        if "FK_dot_uncompiled" in self.run:
            self._precompile_fn(self.run["FK_dot_uncompiled"], (self.default_states, self.default_constants), "FK_dot")
        if "FK_ddot_uncompiled" in self.run:
            self._precompile_fn(self.run["FK_ddot_uncompiled"], (self.default_states, self.default_constants), "FK_ddot")

        # 2. Dynamics JAX functions
        if hasattr(self, "confun"):
            self._precompile_fn(self.confun, (self.default_states, self.default_constants), "jacobian", jacobian=True)
            self._precompile_fn(self.confun, (self.default_states, self.default_constants), "confun")
        if "mass_matrix_uncompiled" in self.run:
            self._precompile_fn(self.run["mass_matrix_uncompiled"], (self.default_states, self.default_constants), "mass_matrix")
        if "forcing_uncompiled" in self.run:
            self._precompile_fn(self.run["forcing_uncompiled"], (self.default_states, self.default_constants), "forcing")

        # 3. Ground contact and actuator models
        if hasattr(self, "gc_model"):
            self._register_contact_model(self.gc_model)
        if hasattr(self, "actuators"):
            self._register_actuator_model(self.actuators)


def _slice(dictionary: dict[str, Any]) -> slice:
    """
    Slice the state vector according to the dictionary
    To make accesing states / v_ easier.
    """
    return np.arange(dictionary["idx"], dictionary["idx"] + dictionary["n"])


def _to_sympy_vector(values: list[Any], reference_frame: ReferenceFrame) -> Any:
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


def load_model(model_file: str, force_rebuild: bool = False, compile_eom: bool = True) -> "BiosymModel":
    """Load a model from a file with caching support.

    Parameters
    ----------
    model_file : str
        Path to model file (.xml, .osim, or .yaml format)
    force_rebuild : bool, optional
        If True, rebuild model even if cached version exists, by default False
    compile_eom : bool, optional
        If True, compile symbolic equations of motion and JAX functions, by default True

    Returns
    -------
    BiosymModel
        Loaded model instance

    Notes
    -----
    Uses hash-based caching to avoid recompiling identical models.
    """
    model_file = os.path.abspath(os.path.expanduser(model_file))

    disable_cache = os.environ.get("BIOSYM_DISABLE_CACHE", "0").lower() in ("1", "true", "yes")
    force_rebuild_env = os.environ.get("BIOSYM_FORCE_REBUILD", "0").lower() in ("1", "true", "yes")
    force_rebuild = force_rebuild or force_rebuild_env

    if disable_cache:
        print("Caching is disabled via BIOSYM_DISABLE_CACHE. Compiling model from source...")
        return BiosymModel(model_file, compile_eom=compile_eom)

    # Generate a hash of the config / or xml tree and save the cloudpickled model in the cache
    model_hash = BiosymModel(model_file, get_hash=True, compile_eom=compile_eom)._get_hash()

    cache_dir = os.environ.get("BIOSYM_CACHE_DIR", _model_cache)
    os.makedirs(cache_dir, exist_ok=True)
    precision_tag = "x64" if bool(jax.config.read("jax_enable_x64")) else "x32"
    cache_path = os.path.join(cache_dir, f"{model_hash}_{precision_tag}.cpkl")

    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading model from cache: {model_hash}_{precision_tag}.cpkl")
        try:
            with open(cache_path, "rb") as f:
                model = cloudpickle.load(f)
            return model
        except Exception as e:
            print(f"Failed to load cached model ({e}). Rebuilding from source...")
            with contextlib.suppress(Exception):
                os.remove(cache_path)

    model = BiosymModel(model_file, compile_eom=compile_eom)
    # Save the model to the cache
    try:
        with open(cache_path, "wb") as f:
            cloudpickle.dump(model, f)
    except Exception as e:
        print(f"Warning: Failed to save model to cache ({e}).")
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

    from biosym.utils import states as states_module

    start = time.time()
    # model_file = "tests/models/pendulum.xml"
    model_file = "tests/models/gait2d/gait2d.yaml"
    # model_file = "tests/models/gait2d_torque/gait2d_torque.yaml"

    model = load_model(model_file, True)
    print(f"Reloading model in {time.time() - start} seconds")
    start = time.time()
    states = model.default_states
    constants = model.default_constants
    print(states)
    model.run["actuator_model"](states, constants)
    states_dict = states_module.concatenate([model.default_states] * 100)
    model.run["actuator_model"](states_dict, constants)

    for _ in range(1):
        model.run["jacobian"](states, constants)
    print(f"1 jacobian in in {time.time() - start} seconds (with reimporting)")
    print(model.run["jacobian"](states, constants))
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
    print("jacobian shape:", model.run["jacobian"](states, constants).model.shape)
    print("gc_model shape:", model.run["gc_model"](states, constants)[1].shape)
    print(
        "gc_model jacobian shape:",
        model.run["gc_model_jacobian"](states, constants)[0].model.shape,
    )
