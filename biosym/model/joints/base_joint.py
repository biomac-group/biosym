from abc import ABC, abstractmethod


class BaseJoint(ABC):
    """
    Abstract base class for joint type translation from OpenSim models.
    
    This class defines the interface that all joint type implementations must follow.
    Joints define the kinematic constraints between rigid bodies in biomechanical models.
    
    The base class provides a common framework for different joint formulations,
    such as PinJoints, PlanarJoints, WeldJoints, etc.
    
    Parameters
    ----------
    joint :
        Parsed joint data from the OpenSim model, typically containing:
        - Joint type (e.g., "PinJoint", "WeldJoint")
        - Joint Name
        - Parent body
        - Child body
        - Parent offset
        - Parent orientation
        - Child offset
        - Child orientation
        - Coordinates: 
                        - Name
                        - Range
                        - Damping
                        - Stiffness
                        - Armature
        
    Notes
    -----
    Joint types are responsible for:
    - Defining kinematic constraints between rigid bodies
    - Managing joint state during simulation
    - Handling joint limit enforcement
    - Computing friction and normal forces
    
    Subclasses must implement all abstract methods to define specific joint
    behavior for different joint types (e.g., PinJoints, PlanarJoints, WeldJoints, etc.).
    
    See Also
    --------
    biosym.model.joints.pin_joint.PinJoint : Pinjoint implementation
    biosym.model.joints.planar_joint.PlanarJoint : Planarjoint implementation
    biosym.model.joints.weld_joint.WeldJoint : Weldjoint implementation
    """

    def __init__(self, joint):
        pass
    
    @abstractmethod
    def get_names(self):
        """Returns a list of the joint names."""
    
    @abstractmethod
    def get_type(self):
        """
        Get the list of types of the joints.
        
        Returns
        -------
        str
            Either "slide" or "hinge"
            
        Notes
        -----
        This method identifies which type of movement the joint has based on the 
        definition of MuJoCo models. Hinge refers to a rotational joint, while slide 
        refers to a translational joint. The specific implementation of the joint 
        type will determine how the joint behaves during simulation and optimization.
        """

    @abstractmethod
    def get_axis(self):
        """
        Get the list of the axis of rotation for the joints.
        
        Returns
        -------
        list of float
            The axis of rotation for the joint.
            
        Notes
        -----
        The axis of rotation is a 3D vector that defines the direction of rotation 
        for hinge joints. For slide joints, this determines the direction of translation. 
        The axis may be a zero vector or not applicable (e.g., for WeldJoints). The axis has 
        one element equal to 1 for the direction of rotation/translation and zero elsewhere.
        """

    @abstractmethod
    def get_range(self):
        """
        Get the list of range of motion for the joints.
        
        Returns
        -------
        list of float
            The lower and upper bounds of the joint's range of motion.
            
        Notes
        -----
        The range of motion defines the limits within which the joint can move.
        For hinge joints, this typically represents the angular limits.
        For slide joints, this represents the linear limits. WeldJoints may 
        have a range of (0, 0) since they do not allow movement.
        """

    @abstractmethod
    def get_parent_body(self):
        """
        Get the list of the parent body names to which the joints are attached.
        
        Returns
        -------
        str
            The parent rigid body of the joint.
        """
    @abstractmethod
    def get_child_body(self):
        """
        Get the list of child body names to which the joints are attached.
        
        Returns
        -------
        str
            The child rigid body of the joint.
        """

    @abstractmethod
    def get_damping(self):
        """
        Get the list of damping coefficient for the joints.
        
        Returns
        -------
        float
            The damping coefficient for the joint.
        
        Notes
        -----
        Damping represents the resistive force that opposes joint motion, typically 
        modeled as a linear function of joint velocity. It should return zero if 
        not determined from the OpenSim model or if the joint does not have damping.
        """
    
    @abstractmethod
    def get_stiffness(self):
        """
        Get the list of stiffness coefficient for the joints.
        
        Returns
        -------
        float
            The stiffness coefficient for the joint.
        
        Notes
        -----
        Stiffness represents the restoring force that opposes joint displacement, typically 
        modeled as a linear function of joint position. It should return zero if 
        not determined from the OpenSim model or if the joint does not have stiffness.
        """