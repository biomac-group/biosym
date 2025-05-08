from biosym.model.actuators.actuator_models import *
import xml.etree.ElementTree as ET

def get(file_path, body_weight=None):
    """
        Parses the contact model file and returns a list of contact points.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    if root.get("type") in ["actuator", "general"]:
        return general.General(root)
    else: 
        raise ValueError(f"Unknown contact model type: {root.get('type')}")


    