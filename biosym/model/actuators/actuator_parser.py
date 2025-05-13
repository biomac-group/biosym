from biosym.model.actuators.actuator_models import *
import xml.etree.ElementTree as ET

def get(file_path, body_weight=None):
    """
        Parses the contact model file and returns a list of contact points.
    """
    if type(file_path) is str:
        tree = ET.parse(file_path)
        root = tree.getroot()
    else:
        root = file_path
    if root.get("type") in ["actuator", "general"]:
        return general.General(root)
    elif root.get("type") is None:
        # This might not work for every model
        return general.GeneralMujoco(root.findall("motor"))
    else:
        raise ValueError(f"Unknown actuator model type: {root.get('type')}")


    