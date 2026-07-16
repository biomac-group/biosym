import jax.numpy as jnp

from biosym.model.actuators.base_actuator import BaseActuator


class TorqueActuator(BaseActuator):
    """
    Body-pair torque actuator (OpenSim TorqueActuator). Natively applies +M on
    bodyA and -M on bodyB. Under the joint-torque model, this is resolved into
    equivalent joint torques on the joints connecting bodyA and bodyB.

    Resolution (done once, at construction, from the flattened joints):
      - Find the joint chain from bodyB up to bodyA (bodyA must be an ancestor
        of bodyB in the body tree).
      - Every joint on that chain must be a hinge (else raise).
      - Feed -M to each chain joint. Because the joint law is +M on child /
        -M on parent, feeding -M gives +M on the parent side and -M on the child
        side; along a chain the intermediate bodies' contributions cancel, leaving
        net +M on bodyA and -M on bodyB -- matching the TorqueActuator's intent.
    Raises if the two bodies are not connected, if bodyA is not an ancestor of
    bodyB, or if any connecting joint is not a hinge.

    forward() returns a joint-sized torque array: -M placed at each
    connecting joint, summed across actuators, zero elsewhere.

    Receives parsed actuator dicts like:
        {"name": str, "bodyA": str, "bodyB": str, "axis": [x,y,z],
         "torque_is_global": bool, "min": float, "max": float}
    plus the flattened joints list (for chain resolution).
    """

    def __init__(self, actuator_dicts, joints):
        super().__init__()
        self.actuators = {a["name"]: a for a in actuator_dicts}
        self.n_actuators = len(self.actuators)
        self._defs = list(actuator_dicts)  # ordered, matches actuator_model order
        self._joints = list(joints)
        self.states = [f"torque_{name}" for name in self.actuators.keys()]
        self.state_vector = self.states
        self.bounds = {
            "states": {
                "min": jnp.array([float(a.get("min", -1e4)) for a in actuator_dicts]),
                "max": jnp.array([float(a.get("max",  1e4)) for a in actuator_dicts]),
            }
        }

        # Resolve each actuator's (bodyA, bodyB) into the chain of connecting
        # joint names, once at construction. self._chains[i] is the list of joint
        # names connecting actuator i's bodyA (ancestor) to bodyB (descendant).
        self._chains = [self._resolve_chain(a["name"], a["bodyA"], a["bodyB"])
                        for a in self._defs]

    # ------------------------------------------------------------------
    # Chain resolution helpers (pure Python, run once at construction).
    # ------------------------------------------------------------------
    def _joint_by_child(self, child_name):
        """The joint whose child is `child_name` (each body has one parent joint),
        or None if child_name is a root."""
        for j in self._joints:
            if j["child"] == child_name:
                return j
        return None

    def _resolve_chain(self, act_name, bodyA, bodyB):
        """
        Walk up the body tree from bodyB toward the root, collecting joints,
        until we reach bodyA. Returns the list of connecting joint names.
        Raises if bodyA is never reached (not an ancestor / unconnected) or if
        any connecting joint is not a hinge.
        """
        chain = []
        current = bodyB
        # Walk up: at each step, the joint whose child == current connects
        # `current` to its parent. Stop when the parent is bodyA.
        while current != bodyA:
            joint = self._joint_by_child(current)
            if joint is None:
                # Reached a root without finding bodyA -> not connected the
                # ancestor way.
                raise ValueError(
                    f"TorqueActuator '{act_name}': bodyA '{bodyA}' is not an "
                    f"ancestor of bodyB '{bodyB}' (walked up from bodyB to the "
                    f"root without reaching bodyA). Only bodyA-ancestor-of-bodyB "
                    f"torque actuators are supported; a torque between bodies in "
                    f"different branches is not yet handled."
                )
            if joint["type"] != "hinge":
                # Geniunly a limitation of the internal force application approach of model.py
                raise NotImplementedError(
                    f"TorqueActuator '{act_name}': connecting joint "
                    f"'{joint['name']}' is type '{joint['type']}', but torque "
                    f"actuators are only supported across hinge joints."
                )
            chain.append(joint["name"])
            current = joint["parent"]
        if not chain:
            # bodyA == bodyB, or adjacent with no joint -- degenerate.
            raise ValueError(
                f"TorqueActuator '{act_name}': found no connecting joints between "
                f"bodyA '{bodyA}' and bodyB '{bodyB}'."
            )
        return chain

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def get_actuators(self):
        return self.actuators

    def get_n_actuators(self):
        return self.n_actuators

    def get_actuated_joints(self):
        # All joints this actuator model drives: the union of every actuator's
        # connecting chain. May repeat a joint if two actuators share one; the
        # model sums contributions per joint.
        joints = []
        for chain in self._chains:
            joints.extend(chain)
        return joints

    def get_n_states(self):
        return self.get_n_actuators()

    def get_n_constants(self):
        return 0

    def reset(self):
        pass

    def forward(self, states, constants, model):
        """
        Return a joint-sized torque array. For each actuator, place
        -M (its commanded magnitude, negated) at every joint on its connecting
        chain. Multiple actuators / shared joints accumulate by summation.
        """
        joint_names = [j["name"] for j in model.dicts["joints"]]
        n_dof = model.coordinates.n
        commanded = states.actuator_model  # one magnitude per actuator, _defs order

        if commanded.ndim == 1:
            out = jnp.zeros(n_dof)
            for a_idx, chain in enumerate(self._chains):
                M = commanded[a_idx]
                for jname in chain:
                    j = joint_names.index(jname)
                    out = out.at[j].add(-M)   # -M: reverse of default joint law
            return out
        else:
            n_samples = commanded.shape[0]
            out = jnp.zeros((n_samples, n_dof))
            for a_idx, chain in enumerate(self._chains):
                M = commanded[:, a_idx]
                for jname in chain:
                    j = joint_names.index(jname)
                    out = out.at[:, j].add(-M)
            return out