import jax.numpy as jnp

from biosym.model.actuators.base_actuator import BaseActuator


class TorqueActuator(BaseActuator):
    """
    Body-pair torque actuator. Equivalent to OpenSim's TorqueActuator: a commanded
    torque M is applied as +M on bodyA's (parent) frame and -M on bodyB's (child) frame (about a
    given axis). This is NOT the same as a coordinate (joint) torque -- it acts on
    the two bodies directly, and produces different motion in general.

    Receives parsed actuator dicts like:
        {"name": str, "bodyA": str, "bodyB": str, "axis": [x,y,z],
         "torque_is_global": bool, "min": float, "max": float}

    Produces a "body" load: forward() returns the torque magnitudes; the body/axis
    info is exposed for model.py to build the +M/-M frame loads in the EOM.
    """

    def __init__(self, actuator_dicts):
        super().__init__()
        self.actuators = {a["name"]: a for a in actuator_dicts}
        self.n_actuators = len(self.actuators)
        self._defs = list(actuator_dicts)  # ordered, for body/axis lookup
        self.states = [f"torque_{name}" for name in self.actuators.keys()]
        self.state_vector = self.states
        self.bounds = {
            "states": {
                "min": jnp.array([float(a.get("min", -1e4)) for a in actuator_dicts]),
                "max": jnp.array([float(a.get("max",  1e4)) for a in actuator_dicts]),
            }
        }

    def get_load_type(self):
        return "body"

    def get_actuators(self):
        return self.actuators

    def get_n_actuators(self):
        return self.n_actuators

    def get_actuated_joints(self):
        # A TorqueActuator applies a body-frame torque pair, not a joint-slot
        # torque, so it drives no joint moment slots. Its targets are bodies,
        # reported via get_body_pairs(). Returning [] keeps it out of the
        # coordinate-torque (M_) machinery entirely.
        return []

    def get_body_pairs(self):
        """
        The (bodyA, bodyB, axis, torque_is_global) for each actuator, in state
        order. model.py uses this to build the +M/-M body-frame loads, applying
        +M*axis on bodyA and -M*axis on bodyB. Exposed because body loads can't
        go through the joint-moment slot machinery.
        """
        return [
            (a["bodyA"], a["bodyB"],
             a.get("axis", [0.0, 0.0, 1.0]),
             bool(a.get("torque_is_global", False)))
            for a in self._defs
        ]

    def get_n_states(self):
        return self.get_n_actuators()

    def get_n_constants(self):
        return 0

    def reset(self):
        pass

    def forward(self, states, constants, model):
        # Body-load actuator: return the commanded torque magnitudes themselves
        # (one per actuator), in state order. model.py multiplies each by its
        # axis and applies +M on bodyA / -M on bodyB when building the EOM loads.
        # (Contrast CoordinateActuator, which scatters torques into joint slots.)
        return states.actuator_model