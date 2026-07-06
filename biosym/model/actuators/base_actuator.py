from abc import ABC, abstractmethod


class BaseActuator(ABC):
    """
    Abstract base class for actuator models (internal forces) in biomechanical
    simulations.

    This class defines the interface that all actuator implementations must follow.
    Actuators represent force/torque generating elements in the biomechanical model,
    such as muscles, motors, or other active/passive components. Concrete actuators
    receive an already-parsed, source-agnostic definition from the actuator parser 
    so a CoordinateActuator built from an OpenSim model and one
    built from a MuJoCo file are the same object, differing only in the dict the
    parser handed them.

    Load type
    ---------
    Actuators differ in what kind of load they produce, which determines where
    their forward() output is wired in the EOM:

    - "coordinate": a generalized torque per joint (fills the M_ joint-moment
      slots); e.g., CoordinateActuator, PassiveTorques, Hill2d.
    - "body": an equal-and-opposite torque pair applied to two bodies' frames
      (+M on bodyA, -M on bodyB); e.g., TorqueActuator. This is genuinely different
      physics from a coordinate torque and routes to different EOM slots.

    get_load_type() defaults to "coordinate"; only body-load actuators override it.

    Attributes
    ----------
    actuator : object or None
        The actuator object instance (implementation-specific).

    Notes
    -----
    Subclasses implement get_n_actuators, get_actuated_joints, get_n_states,
    get_n_constants, forward, and reset. process_eom / get_n_constraints / get_nnz
    have safe defaults and are overridden only by actuators that need them (e.g., muscles).

    See Also
    --------
    biosym.model.actuators.actuator_models.coordinate_actuator.CoordinateActuator : coordinate torque actuators (Osim); 
                                                                                    similar to general and motor (Mujoco)
    biosym.model.actuators.actuator_models.hill2d.Hill2D : Hill-type muscle model
    biosym.model.actuators.actuator_models.torque_actuator.TorqueActuator : torque actuator (Osim)
    biosym.model.actuators.actuator_models.passive_torques.PassiveTorques : Passive torques
    """

    def __init__(self, *args, **kwargs):
        # Concrete actuators define their own constructors (they need different
        # inputs: a parsed actuator dict, a muscle list, or the joints list).
        self.actuator = None
    
    # Load routing
    def get_load_type(self):
        """
        What kind of load this actuator produces: "coordinate" (joint torques,
        the default) or "body" (a +M/-M pair on two body frames). model.py uses
        this to route forward() output to the correct EOM slots.
        """
        return "coordinate"
    
    @abstractmethod
    def get_n_actuators(self):
        """
        Get the number of actuators in the model.

        Returns
        -------
        int
            Number of actuators defined in this actuator model.

        Notes
        -----
        This method must be implemented by all actuator subclasses to specify
        how many individual actuator elements are present in the model.
        """

    @abstractmethod
    def get_actuated_joints(self):
        """
        Get the list of joints actuated by this actuator model.

        Returns
        -------
        list of str
            Names of joints that are actuated by this actuator model.

        Notes
        -----
        This method must be implemented by all actuator subclasses to specify
        which joints in the biomechanical model are influenced by the actuators.
        """

    @abstractmethod
    def get_n_states(self) -> int:
        """
        Get the number of states associated with this actuator model.

        Returns
        -------
        int
            Number of states defined by this actuator model.

        Notes
        -----
        The default implementation returns 0. Actuator subclasses that define
        internal states (e.g., muscle activation, fiber length) should override
        this method to return the correct number of states.
        """
        return 0

    @abstractmethod
    def get_n_constants(self):
        """Number of constants this actuator adds to constants.actuator_model. 0 if none."""
        return 0

    @abstractmethod
    def reset(self):
        """
        Reset the actuator model to its initial state.

        Notes
        -----
        This method is called at the beginning of simulations to ensure
        actuators start from a clean state. Implementations should reset
        any internal state variables, cached values, or dynamic properties.

        The exact reset behavior depends on the specific actuator type:
        - Muscle models may reset activation states
        - Motor models may reset control states
        - Passive elements may reset stored energy states
        """

    @abstractmethod
    def forward(self, states, constants, model):
        """
        Compute this actuator's load for the current state.

        For a "coordinate" actuator: return an array of joint torques shaped to
        the model's coordinates (the General pattern -- torques placed at the
        actuated joint indices, zeros elsewhere).

        For a "body" actuator: return the per-actuator torque magnitudes (and the
        class also exposes the body/axis info model.py needs to apply the +M/-M
        pair). See TorqueActuator for the exact contract.
        """
    
    def process_eom(self, model):
        """
        Process the equations of motion for the actuator model.

        This method is called during the symbolic equation generation phase
        to integrate actuator dynamics into the overall system equations.

        Parameters
        ----------
        model : biosym.model.model.BiosymModel
            The biomechanical model containing the actuator.

        Notes
        -----
        The default implementation does nothing. Actuator subclasses should
        override this method if they need to add additional equations of motion,
        constraints, or symbolic relationships to the model.

        Examples of when this is needed:
        - Muscle activation dynamics
        - Force-length-velocity relationships
        - Internal state evolution equations
        """

    def get_n_constraints(self, *args, **kwargs) -> int:
        """
        Get the number of constraints defined by this actuator model.

        Returns
        -------
        int
            Number of constraints. Default is 0.

        Notes
        -----
        The default implementation returns 0. Actuator subclasses that define
        constraints (e.g., activation dynamics, force equilibrium) should override
        this method to return the correct number of constraints.
        """
        return 0

    def get_nnz(self) -> int:
        """
        Get the number of non-zero entries in the Jacobian of the actuator model.

        Returns
        -------
        int
            Number of non-zero entries in the Jacobian. Default is 0.

        Notes
        -----
        The default implementation returns 0. Actuator subclasses that define
        constraints or dynamics should override this method to return the correct
        number of non-zero entries in their Jacobian matrices.
        """
        return 0
