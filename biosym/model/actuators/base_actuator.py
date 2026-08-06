from abc import ABC, abstractmethod


class BaseActuator(ABC):
    """
    Abstract base class for actuator models (internal torques) in biomechanical
    simulations.

    All actuators produce joint torques: each actuator contributes a generalized
    torque to one or more of the model's joints (DOFs). The model sums every
    actuator's contribution per joint, so a joint driven by (say) a coordinate
    actuator and passive damping and a muscle ends up with a single summed torque
    entry. model.py then applies each joint's net torque to the two bodies the
    joint connects (+M on child, -M on parent).

    An actuator may touch several joints:
      - CoordinateActuator / PassiveTorques: one joint each.
      - Hill2d muscle: every joint it spans (via its moment arms).
      - TorqueActuator: every joint on the chain between its two bodies.
    This "one actuator, several joints" shape is why forward() returns a torque
    laid out over the model's joints, not a single scalar.

    Concrete actuators receive an already-parsed, source-agnostic definition from
    the actuator parser, so a CoordinateActuator built from an OpenSim model and
    one from a MuJoCo file are the same object, differing only in the parsed dict.

    Notes
    -----
    Subclasses implement get_n_actuators, get_actuated_joints, get_n_states,
    get_n_constants, forward, and reset. process_eom / get_n_constraints / get_nnz
    have safe defaults and are overridden only by actuators that need them (muscles).

    See Also
    --------
    biosym.model.actuators.actuator_models.coordinate_actuator.CoordinateActuator
    biosym.model.actuators.actuator_models.hill2d.Hill2d : Hill-type muscle model
    biosym.model.actuators.actuator_models.torque_actuator.TorqueActuator
    biosym.model.actuators.actuator_models.passive_torques.PassiveTorques
    """

    def __init__(self, *args, **kwargs):
        # Concrete actuators define their own constructors (they take different
        # inputs: a parsed actuator dict, a muscle list, or the joints list).
        self.actuator = None

    @abstractmethod
    def get_n_actuators(self):
        """Number of individual actuator elements in this model."""

    @abstractmethod
    def get_actuated_joints(self):
        """
        Names of the joints this actuator model contributes torque to.

        One actuator may contribute to several joints (a muscle spanning two
        joints, a torque actuator spanning a chain), so the returned list may be
        longer than get_n_actuators(). Joints may repeat across actuators; the
        model sums contributions per joint.
        """

    @abstractmethod
    def get_n_states(self) -> int:
        """
        Number of states this actuator adds to its own sub-vector
        (states.actuator_model). 0 for direct torque actuators; several per
        muscle (fiber length, velocity, activation); 0 for passive.
        """
        return 0

    @abstractmethod
    def get_n_constants(self):
        """Number of constants this actuator adds to constants.actuator_model. 0 if none."""
        return 0

    @abstractmethod
    def reset(self):
        """Reset any internal state (no-op for memoryless actuators)."""

    @abstractmethod
    def forward(self, states, constants, model, states_prev=None, h=None):
        """
        Compute this actuator's joint-torque contribution for the current state.

        Returns an array laid out over the model's joints (shape matching the
        model's coordinates / DOFs), with this actuator's torque placed at the
        joints it actuates and zero elsewhere. The model sums these arrays across
        all actuators (active + passive) to get the net torque per joint.

        Because every actuator returns a full joint-sized array, actuators that
        touch several joints (muscles, chain-spanning torque actuators) simply
        fill several entries -- no special routing needed.

        states_prev/h give the previous node's state and the step size to it
        (None when there is no meaningful predecessor: node 0 of a
        non-periodic problem, a single-node equilibrium problem, or a caller
        outside the main dynamics-constraint evaluation, e.g. GRF tracking).
        Only actuators whose contraction dynamics need a time derivative that
        isn't itself a decision variable (Hill2d's fiber velocity) use these;
        everything else ignores them.
        """

    def process_eom(self, model):
        """
        Symbolic setup, called during EOM generation. Default: nothing. Muscles
        override to build path geometry / moment arms / activation dynamics.
        """

    def get_n_constraints(self, *args, **kwargs) -> int:
        """Number of constraints this actuator adds. Default 0; muscles override."""
        return 0

    def get_nnz(self, *args, **kwargs) -> int:
        """Non-zeros in this actuator's constraint Jacobian. Default 0; muscles override."""
        return 0