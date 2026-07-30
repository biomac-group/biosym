"""Small OpenSim API helpers used by the OpenSim parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import opensim as osim

from biosym.utils.useful_functions import rotation_matrix_xyz

# Normalizes body names by stripping paths and suffixes, and handling special cases like "ground"
def normalize_body_name(frame_or_path: str | None) -> str: 
    if frame_or_path is None:
        return ""

    name = frame_or_path.strip()
    if "/" in name:
        name = name.split("/")[-1]
    if name.endswith("_offset"):
        name = name[:-7]
    if name == "ground":
        name = "ground_frame"
    return name

# Utility function to call the first available method from a list of possible method names on an object
def call_first(obj, names, *args): 
    for name in names:
        func = getattr(obj, name, None)
        if func is not None:
            return func(*args)
    raise AttributeError(f"None of {names} found on {type(obj)}")

# Converts an OpenSim Vec3 to a Python list of floats, returns None if input is None
def vec3_to_list(vec3) -> list[float] | None: 
    if vec3 is None:
        return None
    return [float(vec3.get(0)), float(vec3.get(1)), float(vec3.get(2))]

# Converts an OpenSim rotation to body-fixed XYZ angles, returns zeros if input is None or invalid
def rotation_to_body_fixed_xyz(rotation) -> list[float]: 
    if rotation is None:
        return [0.0, 0.0, 0.0]

    try:
        angles = rotation.convertRotationToBodyFixedXYZ()
    except Exception:
        angles = osim.Vec3(0.0, 0.0, 0.0)
        try:
            rotation.convertRotationToBodyFixedXYZ(angles)
        except Exception:
            return [0.0, 0.0, 0.0]
    return vec3_to_list(angles)

# Converts an OpenSim inertia object to a list of 6 floats (Ixx, Iyy, Izz, Ixy, Ixz, Iyz), returns None if input is None or invalid
def inertia_to_list(inertia) -> list[float] | None: 
    if inertia is None:
        return None

    if hasattr(inertia, "getMoments") and hasattr(inertia, "getProducts"):
        moments = inertia.getMoments()
        products = inertia.getProducts()
        return [
            float(moments.get(0)),
            float(moments.get(1)),
            float(moments.get(2)),
            float(products.get(0)),
            float(products.get(1)),
            float(products.get(2)),
        ]

    try:
        return [
            float(inertia.get(0, 0)),
            float(inertia.get(1, 1)),
            float(inertia.get(2, 2)),
            float(inertia.get(0, 1)),
            float(inertia.get(0, 2)),
            float(inertia.get(1, 2)),
        ]
    except Exception:
        return None

# Returns the translation and orientation (as body-fixed XYZ angles) of a frame, or zeros if the frame is None or invalid
def frame_offset(frame) -> tuple[list[float], list[float]]: 
    if frame is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    pf = osim.PhysicalOffsetFrame.safeDownCast(frame)
    if pf is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    transform = pf.getOffsetTransform()
    trans = vec3_to_list(transform.p())
    orient = rotation_to_body_fixed_xyz(transform.R())
    return trans, orient

# Returns the list of coordinates associated with a joint, or an empty list if none found
def coordinate_list(joint) -> list: 
    if hasattr(joint, "getCoordinateSet"):
        coord_set = joint.getCoordinateSet()
        return [coord_set.get(i) for i in range(coord_set.getSize())]

    if hasattr(joint, "numCoordinates"):
        coords = []
        for i in range(joint.numCoordinates()):
            if hasattr(joint, "get_coordinates"):
                coords.append(joint.get_coordinates(i))
            elif hasattr(joint, "getCoordinate"):
                coords.append(joint.getCoordinate(i))
        return coords

    return []

# Determines if a coordinate is translational or rotational based on its motion type, defaults to "rotational" if unknown
def coordinate_motion_type(coord) -> str: 
    if not hasattr(coord, "getMotionType"):
        return "rotational"

    try:
        motion_str = str(coord.getMotionType()).lower()
    except Exception:
        return "rotational"

    if "trans" in motion_str:
        return "translational"
    return "rotational"

# Returns the parent and child frames of a joint, along with their normalized names, handling various OpenSim versions and conventions
def joint_frames(joint): 
    parent_frame = joint.getParentFrame() if hasattr(joint, "getParentFrame") else None
    child_frame = joint.getChildFrame() if hasattr(joint, "getChildFrame") else None

    parent_name = normalize_body_name(parent_frame.getName()) if parent_frame is not None else ""
    child_name = normalize_body_name(child_frame.getName()) if child_frame is not None else ""

    if not parent_name and hasattr(joint, "getParentBody"):
        parent_name = normalize_body_name(joint.getParentBody().getName())
    if not child_name and hasattr(joint, "getChildBody"):
        child_name = normalize_body_name(joint.getChildBody().getName())

    return parent_frame, child_frame, parent_name, child_name


def joint_child_to_parent_transform(parent_frame, child_frame, force_2d: bool = False) -> tuple[np.ndarray, np.ndarray]:
    parent_trans, parent_orient = frame_offset(parent_frame)
    child_trans, child_orient = frame_offset(child_frame)

    R_parent = rotation_matrix_xyz(parent_orient, force_2d=force_2d)
    R_child = rotation_matrix_xyz(child_orient, force_2d=force_2d)
    R = R_parent @ R_child.T
    t = np.array(parent_trans, dtype=float) - R @ np.array(child_trans, dtype=float)
    return R, t


def joint_transform(joint, force_2d: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if joint is None:
        return np.eye(3), np.zeros(3)

    parent_frame, child_frame, _, _ = joint_frames(joint)
    return joint_child_to_parent_transform(parent_frame, child_frame, force_2d)


def joint_offsets(joint) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if joint is None:
        zeros = np.zeros(3)
        return zeros, zeros, zeros, zeros

    parent_frame, child_frame, _, _ = joint_frames(joint)
    parent_trans, parent_orient = frame_offset(parent_frame)
    child_trans, child_orient = frame_offset(child_frame)
    return (
        np.array(parent_trans, dtype=float),
        np.array(parent_orient, dtype=float),
        np.array(child_trans, dtype=float),
        np.array(child_orient, dtype=float),
    )


def find_joint(joint_set, name_fragment: str):
    if joint_set is None:
        return None

    name_fragment = name_fragment.lower()
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        if name_fragment in joint.getName().lower():
            return joint
    return None


def inertia_list_to_mat(inertia: list[float]) -> np.ndarray:
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = inertia
    return np.array(
        [
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ],
        dtype=float,
    )


def inertia_mat_to_list(I: np.ndarray) -> list[float]:
    return [
        float(I[0, 0]),
        float(I[1, 1]),
        float(I[2, 2]),
        float(I[0, 1]),
        float(I[0, 2]),
        float(I[1, 2]),
    ]


def shift_inertia_to_point(I_at_com: np.ndarray, mass: float, r_com_to_point: np.ndarray) -> np.ndarray:
    r = np.array(r_com_to_point, dtype=float).reshape(3)
    r2 = float(np.dot(r, r))
    return I_at_com + mass * (r2 * np.eye(3) - np.outer(r, r))

# Returns the list of markers in the model, handling different OpenSim versions and conventions
def iter_markers(model) -> list: 
    markers = []
    if hasattr(model, "getComponentList"):
        try:
            markers.extend(model.getComponentList(osim.Marker))
        except Exception:
            pass

    if not markers and hasattr(model, "getMarkerSet"):
        try:
            marker_set = call_first(model, ("getMarkerSet",))
        except Exception:
            marker_set = None
        if marker_set is not None:
            markers = [marker_set.get(i) for i in range(marker_set.getSize())]

    return markers


def iter_markers_from_xml(root: ET.Element) -> list[tuple[str, str, list[float]]]:
    model_elem = root.find(".//Model")
    if model_elem is None:
        model_elem = root.find(".//OpenSimDocument/Model")
    if model_elem is None:
        return []

    marker_set = model_elem.find(".//MarkerSet/objects")
    if marker_set is None:
        marker_set = model_elem.find(".//MarkerSet")
    if marker_set is None:
        return []

    markers = []
    for marker_elem in marker_set.findall("Marker"):
        marker_name = marker_elem.get("name")
        if not marker_name:
            continue

        parent_body = marker_elem.findtext(".//socket_parent_frame")
        if parent_body is None:
            parent_body = marker_elem.findtext(".//body")
        parent_body = normalize_body_name(parent_body) if parent_body else ""

        loc_text = marker_elem.findtext(".//location")
        if not loc_text:
            continue

        marker_pos = [float(x) for x in loc_text.strip().split()]
        markers.append((marker_name, parent_body, marker_pos))

    return markers


def marker_location(model, root: ET.Element, marker_name: str) -> list[float] | None:
    for marker in iter_markers(model):
        if marker.getName() != marker_name:
            continue
        if hasattr(marker, "getLocation"):
            return vec3_to_list(marker.getLocation())
        if hasattr(marker, "get_location"):
            return vec3_to_list(marker.get_location())

    for name, _, loc in iter_markers_from_xml(root):
        if name == marker_name:
            return loc

    return None

# Recursively parses an XML element into a dictionary, handling OpenSim's use of <objects> to hold lists of things like PathPoints
def parse_xml_element(element):
            """Recursively parses an XML element into a dictionary."""
            data = {}
            for child in element:
                # If the child has its own children, recurse!
                if len(child) > 0:
                    # OpenSim often uses <objects> to hold lists of things (like PathPoints)
                    if child.tag == "objects":
                        obj_list = []
                        for obj in child:
                            obj_dict = parse_xml_element(obj)
                            obj_dict["_tag"] = obj.tag # Remember if it's a PathPoint or ConditionalPathPoint
                            obj_dict["name"] = obj.get("name", "")
                            obj_list.append(obj_dict)
                        data[child.tag] = obj_list
                    else:
                        data[child.tag] = parse_xml_element(child)
                # Otherwise, it's just a raw value (like <max_isometric_force>1000</max_isometric_force>)
                elif child.text and child.text.strip():
                    data[child.tag] = child.text.strip()
            return data