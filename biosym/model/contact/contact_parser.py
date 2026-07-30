
"""
Builds contact model(s) from:
  - XML contact file (the yaml-referenced <ground_contact_model> format),
    via get_from_xml(file_path)
  - OpenSim parsed data (parser.data["contact_geometries"] + ["forces"]),
    via get_from_osim(contact_geometries, forces).

Both paths normalize into the same intermediate form -- a list of contact
groups, each (force_law_name, pairs, params) -- then assemble BaseContact
instances. One group -> the single instance; several -> a MultiContact wrapper,
so model.py always sees one gc_model.

Extension points are three tables near the top:
  _GEOMETRY_BUILDERS   : geometry type name      -> builder(geo_dict)
  _OSIM_FORCE_HANDLERS : OpenSim Force type name -> handler(force_dict, geo_objs) -> [groups]
  _FORCE_LAWS          : force-law name          -> constructor(pairs, params)
Adding a geometry, an OpenSim contact-force type, or a force law is a single
registration in the relevant table -- no control-flow edits.

Defaults:
  - geometry type missing (xml)      -> ContactPoint
  - no surface/ground in a group     -> a ContactHalfSpace at global zero
  - no force law named (xml only)    -> SpringDamper
  - OSIM geometries with no force    -> inert (no model), matching OpenSim semantics

Importing the geometry classes here also triggers the @register_penetration
decorators (via the contact_geometries package __init__), so the penetration
registry is populated before any pair's penetration() is used.
"""

import xml.etree.ElementTree as ET

import numpy as np

from biosym.model.contact import base_contact, base_geometry, multi_contact
from biosym.model.contact.contact_geometries import *
from biosym.model.contact.contact_models import *


# ----------------------------------------------------------------------
# Geometry types that act as the "surface" side of a pair (the `other` arg to
# penetration()). Everything else (points, spheres) is a "feature".
# NOTE: any future surface geometry (e.g. mesh, height field) is added here.
# ----------------------------------------------------------------------
_SURFACE_CLASSES = (contact_halfspace.ContactHalfSpace,)


# ----------------------------------------------------------------------
# Geometry builders: type name -> function(geo_dict) -> BaseGeometry.
# A builder owns the knowledge of which fields its geometry needs. Both the
# xml and osim readers normalize their raw entries into the same geo_dict shape
# (name, type, parent_body, location/pos, orientation, parameters) and call
# _make_geometry, so this table is the single place geometry construction lives.
# NOTE: new contact geometry implementations are registered here.
# ----------------------------------------------------------------------
def _build_point(geo):
    return contact_point.ContactPoint(geo["name"], geo["parent_body"], pos=geo["pos"])

def _build_sphere(geo):
    return contact_sphere.ContactSphere(
        geo["name"], geo["parent_body"], pos=geo["pos"],
        radius=float(geo["parameters"]["radius"]),
    )

def _build_halfspace(geo):
    return contact_halfspace.ContactHalfSpace(
        geo["name"], geo["parent_body"], pos=geo["pos"], orientation=geo["orientation"],
    )

_GEOMETRY_BUILDERS = {
    "ContactPoint": _build_point,
    "ContactSphere": _build_sphere,
    "ContactHalfSpace": _build_halfspace,
}

# Default geometry type when a source leaves it unspecified.
_DEFAULT_GEOMETRY_TYPE = "ContactPoint"


# ----------------------------------------------------------------------
# Force laws: name -> constructor(pairs, params_dict) -> BaseContact.
# NOTE: any new contact force law is registered here.
# ----------------------------------------------------------------------
_FORCE_LAWS = {
    "huntcrossley": lambda pairs, p: contact_huntcrossley.HuntCrossley(pairs, **p),
    "springdamper": lambda pairs, p: contact_springdamper.SpringDamper(pairs, **p),
}

# Force law assumed by the xml format when none is named in the file.
_XML_DEFAULT_FORCE_LAW = "springdamper"


# ----------------------------------------------------------------------
# Shared geometry / pairing helpers
# ----------------------------------------------------------------------
def _normalize_geo_dict(raw):
    """Bring a raw geometry entry (from either source) to the common shape the
    builders expect, filling defaults for absent fields."""
    return {
        "name": raw["name"],
        "type": raw.get("type") or _DEFAULT_GEOMETRY_TYPE,
        "parent_body": raw.get("parent_body") or raw.get("parent") or raw.get("body"),
        "pos": raw.get("location", raw.get("pos", [0.0, 0.0, 0.0])),
        "orientation": raw.get("orientation", [0.0, 0.0, 0.0]),
        "parameters": raw.get("parameters", {}),
    }


def _make_geometry(raw):
    """Construct a BaseGeometry from a raw geometry dict (xml or osim), via the
    builder registered for its type. Unknown/empty type -> the default builder."""
    geo = _normalize_geo_dict(raw)
    builder = _GEOMETRY_BUILDERS.get(geo["type"])
    if builder is None:
        raise NotImplementedError(
            f"Contact geometry type '{geo['type']}' (geometry '{geo['name']}') "
            f"is not supported. Register it in _GEOMETRY_BUILDERS."
        )
    return builder(geo)


def _default_ground():
    """Horizontal ground half-space at global zero. orientation [0,0,-pi/2]
    rotates OpenSim's local +X normal up to +Y."""
    return contact_halfspace.ContactHalfSpace(
        "default_ground", "ground_frame", pos=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, -np.pi / 2],
    )


def _make_pairs(geometries):
    """Split a group's geometries into features and surfaces and form
    (feature, surface) pairs. No surface -> default ground at zero."""
    surfaces = [g for g in geometries if isinstance(g, _SURFACE_CLASSES)]
    features = [g for g in geometries if not isinstance(g, _SURFACE_CLASSES)]
    if not features:
        raise ValueError("Contact group has no feature geometries to apply forces to.")
    if not surfaces:
        surfaces = [_default_ground()]
    return [(f, s) for f in features for s in surfaces]

# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def get_from_xml(file_path, body_weight=None):
    """Build the contact model from XML contact file. body_weight is
    accepted for backward compatibility but unused (scaling happens in
    SpringDamper.process_eom)."""
    return _assemble(_parse_xml(file_path))


def get_from_osim(contact_geometries, forces, body_weight=None):
    """Build the contact model(s) from OpenSim parsed data. body_weight unused."""
    return _assemble(_parse_osim(contact_geometries, forces))

# ----------------------------------------------------------------------
# XML parsing. The legacy <ground_contact_model> stores geometries as
# <contact_point> children with shared <default> attributes, and (in the legacy
# format) names no force law. This reader generalizes that: each child element's
# tag-or-"type"-attribute selects the geometry, a "force_law" attribute (on the
# root or per element) selects the law, and per-element force parameters are
# read from the remaining attributes. Absent type -> point; absent law ->
# the xml default (spring-damper).
# ----------------------------------------------------------------------
def _parse_xml(file_path):
    root = ET.parse(file_path).getroot()

    default_el = root.find("default/contact_point")
    defaults = default_el.attrib if default_el is not None else {}

    # Root-level force law, if the file names one; else the xml default.
    root_force_law = root.get("force_law", _XML_DEFAULT_FORCE_LAW)

    # Geometry-bearing children: any element except the <default> block. We read
    # the element tag as a hint but let an explicit "type" attribute win, so
    # both <contact_point .../> and <geometry type="ContactSphere" .../> work.
    entries = []  # (geometry, force_law_name, force_params_tuple)
    for el in root:
        if el.tag == "default":
            continue
        name = el.get("name")
        if name is None:
            raise ValueError("Each contact element must have a name.")

        def attr(key, fallback=None):
            return el.get(key, defaults.get(key, fallback))

        gtype = el.get("type") or _tag_to_geometry_type(el.tag)
        pos_str = attr("pos", "0 0 0")
        raw_geo = {
            "name": name,
            "type": gtype,
            "parent_body": attr("body"),
            "pos": [float(x) for x in pos_str.split()],
            "orientation": [float(x) for x in attr("orientation", "0 0 0").split()],
            "parameters": {"radius": attr("radius")} if attr("radius") else {},
            # NOTE: Other parameters (for different contact geoms) should be added here; 
            # it would be better to use Osim model to add contact geoms
        }
        geometry = _make_geometry(raw_geo)

        # A surface geometry carries no force; only features do.
        if isinstance(geometry, _SURFACE_CLASSES):
            entries.append((geometry, None, None))
            continue

        force_law = el.get("force_law", root_force_law)
        params = _read_xml_force_params(force_law, attr)
        entries.append((geometry, force_law, params))

    if not any(law is not None for _, law, _ in entries):
        raise ValueError("No feature contact geometries found in the contact file.")

    # Group features by (force_law, params); surfaces ride along in every group
    # so _make_pairs can pair against them (or default ground if none present).
    surfaces = [g for g, law, _ in entries if law is None]
    groups = []
    by_key = {}
    for geometry, law, params in entries:
        if law is None:
            continue
        by_key.setdefault((law, params), []).append(geometry)

    for (law, params), feats in by_key.items():
        pairs = _make_pairs(feats + surfaces)
        groups.append((law, pairs, dict(params)))
    return groups

# NOTE: could maybe change that if the yaml file names change to ContactPoint
def _tag_to_geometry_type(tag):
    """Map a legacy xml element tag to a geometry type name. 'contact_point' is
    the historical tag; anything else is assumed already a type name."""
    return "ContactPoint" if tag == "contact_point" else tag

def _read_xml_force_params(force_law, attr):
    """Read a feature's force parameters from xml attributes, as a hashable
    tuple of (name, value) pairs (so groups can be keyed on it). Each force law
    declares which attributes it reads.
    NOTE: when a new force law becomes selectable from xml, add its reader here."""
    if force_law == "springdamper":
        return (
            ("k", float(attr("k", 1.0))),
            ("b", float(attr("b", 0.0))),
            ("mu", float(attr("mu", 0.0))),
        )
    if force_law == "huntcrossley":
        return (
            ("stiffness", float(attr("stiffness", attr("k", 1.0)))),
            ("dissipation", float(attr("dissipation", 0.0))),
            ("static_friction", float(attr("static_friction", attr("mu", 0.0)))),
            ("dynamic_friction", float(attr("dynamic_friction", attr("mu", 0.0)))),
            ("viscous_friction", float(attr("viscous_friction", 0.0))),
            ("transition_velocity", float(attr("transition_velocity", 0.01))),
        )
    raise ValueError(
        f"Force law '{force_law}' is not readable from xml. Add a reader in "
        f"_read_xml_force_params (and register the law in _FORCE_LAWS)."
    )


# ----------------------------------------------------------------------
# OpenSim parsing. Each OpenSim Force type that represents contact gets a
# handler that knows that type's parameter layout and returns contact groups.
# Geometries with no force referencing them are inert (OpenSim semantics), so
# no model is invented for them -- only a diagnostic.
# NOTE: a new OpenSim contact-force type is supported by writing a handler and
# registering it in _OSIM_FORCE_HANDLERS. No other edit needed.
# ----------------------------------------------------------------------
def _resolve_geometries(geo_names, geo_objs, force_name, referenced):
    geos = []
    for gn in geo_names:
        if gn not in geo_objs:
            raise ValueError(f"{force_name} references unknown geometry '{gn}'.")
        geos.append(geo_objs[gn])
        referenced.add(gn)
    return geos


def _handle_huntcrossley(force, geo_objs, referenced):
    """Read an OpenSim HuntCrossleyForce into contact groups. This function is
    where the HuntCrossleyForce XML layout (its ContactParametersSet, the
    space-separated geometry list, the per-parameter-set coefficients) is
    known."""
    fparams = force.get("parameters", {})
    pset = fparams.get("HuntCrossleyForce_ContactParametersSet", {})
    transition_velocity = float(fparams.get("transition_velocity", 0.01))

    groups = []
    for grp in pset.get("objects", []):
        geos = _resolve_geometries(grp["geometry"].split(), geo_objs, force.get("name"), referenced)
        pairs = _make_pairs(geos)
        params = {
            "stiffness": float(grp["stiffness"]),
            "dissipation": float(grp["dissipation"]),
            "static_friction": float(grp["static_friction"]),
            "dynamic_friction": float(grp["dynamic_friction"]),
            "viscous_friction": float(grp["viscous_friction"]),
            "transition_velocity": transition_velocity,
        }
        groups.append(("huntcrossley", pairs, params))
    return groups


_OSIM_FORCE_HANDLERS = {
    "HuntCrossleyForce": _handle_huntcrossley,
}


def _parse_osim(contact_geometries, forces):
    geo_objs = {g["name"]: _make_geometry(g) for g in contact_geometries}

    groups = []
    referenced = set()
    for force in forces:
        handler = _OSIM_FORCE_HANDLERS.get(force.get("type"))
        if handler is None:
            continue  # not a contact force (actuators, etc.) -- ignore
        groups.extend(handler(force, geo_objs, referenced))

    # OpenSim semantics: geometries no force references are inert. Don't invent
    # a model; just report them.
    unreferenced = [
        n for n, g in geo_objs.items()
        if n not in referenced and not isinstance(g, _SURFACE_CLASSES)
    ]
    if unreferenced:
        print(
            "contact_parser: these contact geometries are not referenced by any "
            f"contact force and are inert (no force applied): {unreferenced}"
        )
    return groups


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------
def _assemble(groups):
    if not groups:
        raise ValueError("No contact models could be built from the provided data.")
    models = [_FORCE_LAWS[name](pairs, params) for name, pairs, params in groups]
    return models[0] if len(models) == 1 else multi_contact.MultiContact(models)
