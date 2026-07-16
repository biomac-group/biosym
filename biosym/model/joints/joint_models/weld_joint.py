
from ..base_joint import BaseJoint

class WeldJoint(BaseJoint):
    """
    WeldJoint OpenSim joint model implementation.
    
    A WeldJoint in OpenSim rigidly connects two bodies, allowing 0 DOF.
    """

    def __init__(self, joint):
        self.flat_joints = []
        
        self.joint_type = joint.get("type")
        self.base_name = joint.get("name")
        
        # Validation
        if self.base_name is None:
            raise ValueError("Joint name is missing from the parsed OpenSim dictionary.")
            
        if self.joint_type != "WeldJoint":
            raise ValueError(f"Expected a WeldJoint, but got '{self.joint_type}'.")

        # WeldJoints have 0 DOFs, so we define a "dummy" flat joint 
        # to pass the parent/child topology to the parser.
        flat_joint = {
            "name": self.base_name,
            "type": "weld",          # Custom type flag for the parser
            "axis": [0.0, 0.0, 0.0], # No axis of rotation/translation
            "range": [0.0, 0.0],     # No movement allowed
            "parent": joint.get("parent"),
            "child": joint.get("child"),
            "damping": 0.0,
            "stiffness": 0.0,
            "armature": 0.0
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