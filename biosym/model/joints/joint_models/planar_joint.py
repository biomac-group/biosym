
from ..base_joint import BaseJoint

class PlanarJoint(BaseJoint):
    """
    PlanarJoint OpenSim joint model implementation.
    
    Parameters
    ----------
    joint :
        Parsed joint data from the OpenSim model, typically containing:
        - Joint type (e.g., "PinJoint", "WeldJoint")
        
    Notes
    -----
    A PlanarJoint in OpenSim allows 3 DOFs: 
    1 Rotation (typically around Z) and 2 Translations (typically along X and Y).
    This class splits the PlanarJoint into 3 separate 1-DOF MuJoCo-compatible joints.
    """

    def __init__(self, joint):
                
        self.flat_joints = []

        # Extract common joint properties from the input data
        self.joint_type = joint["type"]
        self.base_name = joint["name"]
        self.coords = joint["coordinates"]

        if self.base_name is None:
            raise ValueError("Joint name is missing from the parsed OpenSim dictionary.")
        if self.joint_type != "PlanarJoint":
            raise ValueError(f"Expected a PlanarJoint, but got '{self.joint_type}'.")
        if len(self.coords) != 3:
            raise ValueError(f"PlanarJoint '{self.base_name}' must have exactly 3 coordinates. Found {len(self.coords)}.")
        
        for coord in self.coords:
            coord_name = coord["name"].lower()
                
            if "tx" in coord_name:
                axis = [1.0, 0.0, 0.0]
                translated_type = "slide"
            elif "ty" in coord_name:
                axis = [0.0, 1.0, 0.0]
                translated_type = "slide"
            elif "tz" in coord_name:
                axis = [0.0, 0.0, 1.0]
                translated_type = "slide"
            elif "rx" in coord_name: 
                axis = [1.0, 0.0, 0.0]
                translated_type = "hinge"
            elif "ry" in coord_name: 
                axis = [0.0, 1.0, 0.0]
                translated_type = "hinge"
            elif "rz" in coord_name: 
                axis = [0.0, 0.0, 1.0]
                translated_type = "hinge"
            else:
                raise ValueError("The joint coordinate is not well-defined. The name of the coordinate should include rx, ry, rz, tx, ty, or tz.")
            
            flat_joint = {
                "name": f"{self.base_name}_{coord_name}",
                "type": translated_type,
                "axis": axis,
                "range": coord.get("range", [-3.14, 3.14]),
                "parent": joint["parent"],
                "child": joint["child"],
                "damping": coord.get("damping", 0.0),
                "stiffness": coord.get("stiffness", 0.0),
                "armature": coord.get("armature", 0.0)
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