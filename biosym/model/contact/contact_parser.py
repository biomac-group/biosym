import xml.etree.ElementTree as ET

from biosym.model.contact.contact_models import *


def get(file_path, body_weight=None):
    """
    Parses the contact model file and returns a list of contact points.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    if root.get("type") == "contact_points":
        return contact_points.ContactPoints(root, body_weight=body_weight)
    raise ValueError(f"Unknown contact model type: {root.get('type')}")
