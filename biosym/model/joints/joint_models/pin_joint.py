
from ..base_joint import BaseJoint

class PinJoint(BaseJoint):
    """
    PinJoint OpenSim joint model implementation.
    
    Parameters
    ----------
    joint :
        Parsed joint data from the OpenSim model, typically containing:
        - Joint type (e.g., "PinJoint", "WeldJoint")
        
    Notes
    -----
    The PinJoint class represents a rotational joint that allows movement around a single axis. 
    It inherits from the BaseJoint class, which defines the common interface for all joint types.
    """

    def __init__(self, joint):
        
        self.flat_joints = []

        # Extract common joint properties from the input data
        self.joint_type = joint.get("type")
        self.coords = joint.get("coordinates", [])

        if self.joint_type != "PinJoint":
            raise ValueError(f"Expected a PinJoint, but got '{self.joint_type}'.")
            
        if len(self.coords) != 1:
            raise ValueError(f"PinJoint must have exactly one coordinate. Found {len(self.coords)}.")
      
        flat_joint = {
            "name": self.coords[0].get("name"),
            "type": "hinge", # The joint type is translated to "hinge" for compatibility with MuJoCo's joint definitions
            "axis": self.coords[0].get("axis", [0.0, 0.0, 1.0]), # Default to Z-axis if not specified
            "range": self.coords[0].get("range", [-3.14, 3.14]),
            "parent": joint.get("parent"),
            "child": joint.get("child"),
            "damping": self.coords[0].get("damping", 0.0),
            "stiffness": self.coords[0].get("stiffness", 0.0),
            "armature": self.coords[0].get("armature", 0.0) # Armature is typically not used for OpenSim joints, set to zero
            }
        
        self.flat_joints.append(flat_joint)

    def get_names(self):
        return [j["name"] for j in self.flat_joints]
    
    def get_type(self):
        return [j["type"] for j in self.flat_joints]

    def get_axis(self):
        return [j["axis"] for j in self.flat_joints]

    def get_range(self):
        return [j["range"] for j in self.flat_joints]

    def get_parent_body(self):
        return [j["parent"] for j in self.flat_joints]

    def get_child_body(self):
        return [j["child"] for j in self.flat_joints]

    def get_damping(self):
        return [j["damping"] for j in self.flat_joints]
    
    def get_stiffness(self):
        return [j["stiffness"] for j in self.flat_joints]