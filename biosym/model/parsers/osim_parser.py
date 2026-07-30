import xml.etree.ElementTree as ET
import contextlib
import os
import sys
import numpy as np
import opensim as osim

from biosym.model.parsers.base_parser import BaseParser
from biosym.utils import opensim_utils as osu
from biosym.utils import useful_functions as uf

@contextlib.contextmanager
def _suppress_native_output():
    """Suppress stdout/stderr emitted by native libraries during model loading."""
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)

class OsimParser(BaseParser):
    """
    General Parser for OpenSim model files (.osim).
    This class leverages the OpenSim API for exact kinematic and inertial transforms,
    and the XML ElementTree for extracting Custom Forces and Contact Geometries.
    
    """

    def __init__(self, model_file, verbose=False):
        super().__init__(model_file)
        
        # Load the model via the API to access exact spatial kinematics
        with _suppress_native_output():
            self.model = osim.Model(model_file)
            self.state = self.model.initSystem()
            
        # --- NEW XML CLEANING STEP ---
        # OpenSim sometimes writes illegal C++ namespaces (::) into XML tags.
        # We read the file as raw text, clean the colons, and then parse the tree.
        with open(model_file, 'r', encoding='utf-8') as file:
            raw_xml_text = file.read()
            
        clean_xml_text = raw_xml_text.replace("::", "_")
        self.root = ET.fromstring(clean_xml_text)
        # -----------------------------

        self.data = {
            "bodies": [],
            "joints": [],
            "sites": [], 
            "contact_geometries": [], 
            "forces": []
        }
        
        self._parse(verbose)

    def _parse(self, verbose=False):
        model = self.model
        
        # 1. Parse Gravity
        try:
            gravity_vec = osu.call_first(model, ("getGravity", "get_gravity"))
            self.gravity = osu.vec3_to_list(gravity_vec) if gravity_vec else [0.0, -9.81, 0.0]
        except Exception:
            self.gravity = [0.0, -9.81, 0.0]

        # ---------------------------------------------------------
        # PASS 1: Parse Bodies (using OpenSim API)
        # ---------------------------------------------------------
        body_set = osu.call_first(model, ("getBodySet",)) if hasattr(model, "getBodySet") else None
        bodies_dict = {}
        
        if body_set is not None:
            for i in range(body_set.getSize()):
                body = body_set.get(i)
                body_name = body.getName()
                if not body_name:
                    continue
                
                mass = osu.call_first(body, ("getMass", "get_mass"))
                com_vec = osu.call_first(body, ("getMassCenter", "get_mass_center"))
                com = osu.vec3_to_list(com_vec)
                inertia = osu.call_first(body, ("getInertia", "get_inertia"))
                inertia_vals = osu.inertia_to_list(inertia) 

                if (mass is None) or (com is None) or (inertia_vals is None) or (len(inertia_vals) != 6):
                        continue

                body_data = {
                    "name": body_name,
                    "parent": "ground_frame", # Default, overwritten in Pass 2
                    "mass": [float(mass)],
                    "inertia": inertia_vals,
                    "com": com,
                    "joints": [],
                }
                bodies_dict[body_name] = body_data
                self.data["bodies"].append(body_data)

        # ---------------------------------------------------------
        # PASS 2: Parse Joints & Construct Hierarchy (using OpenSim API)
        # ---------------------------------------------------------
        joint_set = osu.call_first(model, ("getJointSet",)) if hasattr(model, "getJointSet") else None
        
        if joint_set is not None:
            for i in range(joint_set.getSize()):
                joint = joint_set.get(i)
                
                # Use utility to get parent/child strings and frames
                parent_frame, child_frame, parent_name, child_name = osu.joint_frames(joint)
                if not parent_name or not child_name:
                    continue

                # Extract raw spatial arrays for parent/child frames (translation and orientation)
                parent_trans, parent_orient, child_trans, child_orient = osu.joint_offsets(joint)

                # Update the child body's topological parent
                if child_name in bodies_dict:
                    bodies_dict[child_name]["parent"] = parent_name

                # Get the exact OpenSim joint type (e.g., "WeldJoint", "PinJoint", etc.)
                osim_joint_type = joint.getConcreteClassName() if joint else "UnknownJointType"
                
                # Create generic Joint container
                joint_data = {
                    "name": joint.getName(),
                    "type": osim_joint_type,
                    "parent": parent_name,
                    "child": child_name,
                    "parent_offset": np.round(parent_trans, 10).tolist(),
                    "parent_orientation": np.round(parent_orient, 10).tolist(),
                    "child_offset": np.round(child_trans, 10).tolist(),
                    "child_orientation": np.round(child_orient, 10).tolist(),
                    "coordinates": []
                }

                # Parse coordinates (DOF) inside this joint
                coordinates = osu.coordinate_list(joint)

                for coord in coordinates:
                    # Extract physiological ranges (if exists)
                    try:
                        coord_range = [coord.getRangeMin(), coord.getRangeMax()]
                    except Exception:
                        coord_range = [-3.14, 3.14]
                    
                    # Extract dynamic properties (if exists)
                    try:
                        damping = float(osu.call_first(coord, ("getDamping", "get_damping")))
                    except Exception:
                        damping = 0.0
                    try:
                        stiffness = float(osu.call_first(coord, ("getStiffness", "get_stiffness")))
                    except Exception:
                        stiffness = 0.0
                    
                    # Append the parsed data
                    joint_data["coordinates"].append({
                        "name": coord.getName(),
                        "range": coord_range,
                        "damping": damping,
                        "stiffness": stiffness,
                        "armature": 0.0 # OpenSim doesn't have a direct armature equivalent, set to 0.0 or use a heuristic if needed
                    })
                
                # Save the joint to the dictionary and also link it to the child body for easy access later
                self.data["joints"].append(joint_data)
                if child_name in bodies_dict:
                    bodies_dict[child_name]["joints"].append(joint_data)

        # ---------------------------------------------------------
        # PASS 3: Parse Markers / Sites (using OpenSim API)
        # ---------------------------------------------------------
        markers = osu.iter_markers(model)
        if markers:
            for marker in markers:
                marker_name = marker.getName()

                parent_body = ""
                if hasattr(marker, "getParentFrame"):
                    parent_frame = marker.getParentFrame()
                    parent_body = osu.normalize_body_name(parent_frame.getName()) if parent_frame is not None else ""
                if not parent_body and hasattr(marker, "getBodyName"):
                    parent_body = osu.normalize_body_name(marker.getBodyName())
                
                if not parent_body:
                    continue

                marker_pos = None
                if hasattr(marker, "getLocation"):
                    marker_pos = osu.vec3_to_list(marker.getLocation())
                elif hasattr(marker, "get_location"):
                    marker_pos = osu.vec3_to_list(marker.get_location())
                
                if marker_pos is None:
                    raise ValueError(f"Marker '{marker_name}' location not found.")
                
                # Extract marker weight (how trusted the marker is) if available (default to 1.0 if not specified)
                try:
                    weight = float(osu.call_first(marker, ("getWeight", "get_weight")))
                except Exception:
                    weight = 1.0 # Default weight if not specified
                
                self.data["sites"].append({
                    "name": marker_name,
                    "pos": marker_pos,
                    "parent": parent_body,
                    "weight": weight
                    })

        # ---------------------------------------------------------
        # PASS 4: Parse Contact Geometries (using XML)
        # ---------------------------------------------------------
        # Find ContacGeometrySet in the XML tree (handles different OpenSim versions and conventions)
        contact_set = self.root.find(".//ContactGeometrySet/objects")

        if contact_set is not None:
            for contact in contact_set:
                geometry_data = {
                    "name": contact.get("name"),
                    "type": contact.tag, # e.g., "ContactSphere", "ContactHalfSpace", etc.
                    "parent_body": None,
                    "location": [0.0, 0.0, 0.0], # Default, overwritten if location is specified
                    "orientation": [0.0, 0.0, 0.0], # Default, overwritten if orientation is specified
                    "parameters": {} # Everything else goes here; e.g., radius for ContactSphere
                }

                # Scrape the XML tags
                for param in contact:
                    if param.text and param.text.strip():
                        tag = param.tag
                        text_val = param.text.strip()
                        
                        # Intercept known tags for parent body, location, and orientation; everything else goes into parameters dict
                        if tag == "socket_frame":
                            geometry_data["parent_body"] = osu.normalize_body_name(text_val)
                        elif tag == "location":
                            geometry_data["location"] = [float(x) for x in text_val.split()]
                        elif tag == "orientation":
                            geometry_data["orientation"] = [float(x) for x in text_val.split()]
                        else:
                            # Generic catch-all (radius, stiffness, friction, etc.)
                            geometry_data["parameters"][tag] = text_val

                self.data["contact_geometries"].append(geometry_data)
    
        # ---------------------------------------------------------
        # PASS 5: Parse Forces (using XML)
        # ---------------------------------------------------------
        # Find the ForceSet in the XML tree (handles different OpenSim versions and conventions)
        force_set = self.root.find(".//ForceSet/objects")
        
        if force_set is not None:
            for force in force_set:
                force_data = {
                    "name": force.get("name"),
                    "type": force.tag, # e.g., "CoordinateActuator", "PointToPointSpring", etc.
                    "parameters": osu.parse_xml_element(force)
                }
                self.data["forces"].append(force_data)
    
               
        if verbose:
            print(f"Parsed {len(self.data['bodies'])} bodies, {len(self.data['joints'])} joints.")
            print(f"Detected {len(self.data['contact_geometries'])} contact geometries.")
            print(f"Detected {len(self.data['forces'])} forces.")

    # Utility function for converting parsed Bodies
    def convert_body_frame(self, body):
        # Convert a body's origin to the joint origin to which it is a child
        
        # Find the parent_joint where this body is a child
        joint = None
        for j in body.get("joints", []):
            if j.get("child") == body["name"]:
                joint = j
                break
        
        # If there is no parent_joint return as is
        if not joint:
            body["body_offset"] = [0.0, 0.0, 0.0]
            body["body_orientation"] = [0.0, 0.0, 0.0]
            return body
        
        # Extract OpenSim joint offsets and orientations
        child_offset = np.array(joint.get("child_offset", [0.0, 0.0, 0.0]), dtype=float)
        child_orient = np.array(joint.get("child_orientation", [0.0, 0.0, 0.0]))

        # OpenSim's child_orientation defines the joint frame relative to the body frame
        # Therefore, using the rotation_matrix_xyz we calculate the rot_joint_to_body
        rot_joint_to_body = uf.rotation_matrix_xyz(child_orient)

        # Transform the body COM position
        com_old = np.array(body.get("com", [0.0, 0.0, 0.0]), dtype=float)
        com_new = (rot_joint_to_body.T) @ (com_old - child_offset)
        com_new[np.abs(com_new) < 1e-7] = 0.0

        # Transform the inertia tensor
        inertia_old = body.get("inertia", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        inertia_old_mat = osu.inertia_list_to_mat(inertia_old)
        inertia_new_mat = (rot_joint_to_body.T) @ inertia_old_mat @ rot_joint_to_body
        inertia_new_mat[np.abs(inertia_new_mat) < 1e-7] = 0.0
        inertia_new = osu.inertia_mat_to_list(inertia_new_mat)

        # Update the dictionary
        body["com"] = com_new.tolist()
        body["inertia"] = inertia_new
        body["body_offset"] = joint.get("parent_offset", [0.0, 0.0, 0.0])
        body["body_orientation"] = joint.get("parent_orientation", [0.0, 0.0, 0.0])

        # Transform Outgoing Joints
        # Since we moved this body's origin, any joint that considers this body as its "parent"
        # must have its parent_offset and parent_orientation updated
        for j_out in self.data["joints"]:
            if j_out["parent"] == body["name"]:
                # Get the old offset and orientation relative to the old body origin
                parent_offset_old = np.array(j_out.get("parent_offset", [0.0, 0.0, 0.0]), dtype=float)
                parent_orientation_old = np.array(j_out.get("parent_orientation", [0.0, 0.0, 0.0]))

                # Calculate the new offset
                parent_offset_new = (rot_joint_to_body.T) @ (parent_offset_old - child_offset)
                parent_offset_new[np.abs(parent_offset_new) < 1e-7] = 0.0

                # Calculate the new orientation
                R_old = uf.rotation_matrix_xyz(parent_orientation_old)
                R_new = (rot_joint_to_body.T) @ R_old
                R_new[np.abs(R_new) < 1e-7] = 0.0
                # Convert back to XYZ
                parent_orientation_new = uf.rot_mat_to_xyz(R_new)
                parent_orientation_new[np.abs(parent_orientation_new) < 1e-7] = 0.0

                # Overwrite the outgoing joint's parent data
                j_out["parent_offset"] = parent_offset_new.tolist()
                j_out["parent_orientation"] = parent_orientation_new.tolist()

        # Transform Contact Geometries
        # Since we moved this body's origin, any contact geometry attached to this body must have its 
        # location and orientation updated
        for cg in self.data["contact_geometries"]:
            if cg.get("parent_body") == body["name"]:
                # Get the location and orientation relative to the old body origin
                loc_old = np.array(cg.get("location", [0.0, 0.0, 0.0]), dtype=float)
                orient_old = np.array(cg.get("orientation", [0.0, 0.0, 0.0]))

                # Calculate the new location
                loc_new = (rot_joint_to_body.T) @ (loc_old - child_offset)
                loc_new[np.abs(loc_new) < 1e-7] = 0.0

                # Calculate the new orientation
                R_cg_old = uf.rotation_matrix_xyz(orient_old)
                R_cg_new = (rot_joint_to_body.T) @ R_cg_old
                R_cg_new[np.abs(R_cg_new) < 1e-7] = 0.0
                # Convert back to XYZ
                orient_new = uf.rot_mat_to_xyz(R_cg_new)
                orient_new[np.abs(orient_new) < 1e-7] = 0.0

                # Overwrite the contact geometry's data
                cg["location"] = loc_new.tolist()
                cg["orientation"] = orient_new.tolist()

        return body
    
    # ---------------------------------------------------------
    # Getter Methods (Required by BaseParser contract)
    # ---------------------------------------------------------
    def get_n_bodies(self): # Returns the number of bodies in the model
        return len(self.data["bodies"])
    
    def get_bodies(self): # Returns the list of bodies in the model
        return self.data["bodies"]
    
    def get_n_joints(self): # Returns the number of joints in the model
        return len(self.data["joints"])
    
    def get_joints(self): # Returns the list of joints in the model
        return self.data["joints"]
    
    def get_n_sites(self): # Returns the number of sites in the model
        return len(self.data["sites"])
    
    def get_sites(self): # Returns the list of sites in the model
        return self.data["sites"]
    
    def get_gravity(self): # Returns the gravity vector in the model 
        return self.gravity
    
    def get_n_external_forces(self):
        return len(self.get_external_forces_bodies()) * 3
    
    def get_external_forces_bodies(self):
        if getattr(self, "external_forces_bodies", None) is not None:
            return self.external_forces_bodies
        return list(set(cg["parent_body"] for cg in self.data["contact_geometries"] if cg.get("parent_body")))
    
    def get_n_internal_forces(self): # Returns the number of internal forces in the model (i.e. actuators)
        return len(self.data["forces"])
        
    def get_internal_forces(self): # Returns the list of internal forces (actuators) in the model
        return self.data["forces"]