"""
Builds active-actuator model(s) from any source, normalizing into a common form.

Type names are used as registry keys exactly as they appear in their source
("CoordinateActuator", "TorqueActuator", "hill_2d"). The only translation is the
genuine MuJoCo-vs-OSIM naming difference: MuJoCo's "motor"/"general" map onto
"CoordinateActuator". Everything else passes through unchanged.

Registries, all keyed on type:
  _OSIM_READERS      : type -> reader(force_dict) -> reader output
  _XML_READERS       : type -> (child element tag(s), reader(el) -> reader output)
  _ACTUATOR_BUILDERS : type -> builder(items, joints, joint_names, defaults) -> BaseActuator

A reader's output need only match what that type's builder/constructor expects;
it does not have to be uniform across types. The coordinate/torque readers
produce flat dicts; the hill_2d reader passes the raw <muscle> elements straight
through, so Hill2d's existing constructor (joint_names, muscle_elements,
defaults) is used unchanged.

NOTE:
Adding support for a new actuator type means: register a builder, and register a
reader for each source that can produce it. No control-flow edits.

Passive torques are NOT built here -- they are always-on, derived from the
joints, and constructed directly in model.py as PassiveTorques(joints).
"""

import xml.etree.ElementTree as ET

from biosym.model.actuators.actuator_models import *
from biosym.model.actuators.multi_actuator import MultiActuator


# ----------------------------------------------------------------------
# Foreign source type names -> canonical type names. Names not listed pass
# through unchanged (OSIM names, and "hill_2d", are used as-is). This grows only
# when a new source uses a different name for an existing type (rare); a
# brand-new actuator type does not touch this table.
# ----------------------------------------------------------------------
_CANONICAL_TYPE = {
    "general": "CoordinateActuator",
    "motor":   "CoordinateActuator",
}

def _canonical(type_name):
    return _CANONICAL_TYPE.get(type_name, type_name)


# ----------------------------------------------------------------------
# OSIM readers: type -> function(force_dict) -> reader output.
# Each reader owns the knowledge of that force type's OSIM parameter layout.
# ----------------------------------------------------------------------
def _read_osim_coordinate(force):
    p = force.get("parameters", {})
    return {
        "name": force.get("name"),
        "joint": p.get("coordinate"),
        "min": float(p.get("min_control", -1e4)),
        "max": float(p.get("max_control",  1e4)),
    }

def _read_osim_torque(force):
    p = force.get("parameters", {})
    return {
        "name": force.get("name"),
        "bodyA": p.get("bodyA"),
        "bodyB": p.get("bodyB"),
        "axis": _parse_axis(p.get("axis")),
        "torque_is_global": str(p.get("torque_is_global", "false")).lower() == "true",
        "min": float(p.get("min_control", -1e4)),
        "max": float(p.get("max_control",  1e4)),
    }

_OSIM_READERS = {
    "CoordinateActuator": _read_osim_coordinate,
    "TorqueActuator":     _read_osim_torque,
}


# ----------------------------------------------------------------------
# XML readers: type -> (child element tag(s), function(el) -> reader output).
# The child tag(s) say which elements under the actuator block belong to this
# type, so the XML loop stays type-agnostic.
# ----------------------------------------------------------------------
def _read_xml_coordinate(el):
    # MuJoCo <motor>/<general>: torque bounds are gear * ctrlrange when given.
    ctrlrange = el.get("ctrlrange")
    gear = float(el.get("gear", 1.0))
    if ctrlrange is not None:
        lo, hi = (float(x) for x in ctrlrange.split())
        cmin, cmax = lo * gear, hi * gear
    else:
        cmin, cmax = float(el.get("min", -1e4)), float(el.get("max", 1e4))
    return {"name": el.get("name"), "joint": el.get("joint"), "min": cmin, "max": cmax}

_XML_READERS = {
    "CoordinateActuator": (("motor", "general"), _read_xml_coordinate),
    # hill_2d's reader passes the raw <muscle> element straight through, since
    # Hill2d's unchanged constructor consumes the elements directly.
    "hill_2d":            (("muscle",),          lambda el: el),
}


# ----------------------------------------------------------------------
# Builders: type -> builder(items, joint_names, defaults) -> BaseActuator.
# The buildable set is exactly these keys. Coordinate/torque builders ignore
# joint_names/defaults; hill_2d uses all three.
# NOTE: register any new actuator implementation here.
# ----------------------------------------------------------------------
_ACTUATOR_BUILDERS = {
    "CoordinateActuator": lambda items, joints, jn, defaults: coordinate_actuator.CoordinateActuator(items),
    "TorqueActuator":     lambda items, joints, jn, defaults: torque_actuator.TorqueActuator(items, joints),
    "hill_2d":            lambda items, joints, jn, defaults: hill2d.Hill2d(jn, items, defaults),
}


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------
def get_from_xml(source, joints=None, joint_names=None):
    """Build actuator(s) from a MuJoCo/standalone XML actuator block (an Element)
    or a path to an actuator XML file. `joints` is the flattened joint list,
    needed by TorqueActuator for body-chain resolution."""
    root = ET.parse(source).getroot() if isinstance(source, str) else source
    return _assemble(_parse_xml(root), joints, joint_names)


def get_from_osim(forces, joints, joint_names=None):
    """Build actuator(s) from the OpenSim parsed force set. `forces` is
    parser.data["forces"]; `joints` is parser.data["joints"], passed to actuators
    (TorqueActuator) that resolve body chains from it."""
    groups = _parse_osim(forces)
    if not groups:
        return None
    return _assemble(groups, joints, joint_names)


# ----------------------------------------------------------------------
# Readers (both type-agnostic loops over their registries)
# ----------------------------------------------------------------------
def _parse_osim(forces):
    groups = {}  # type -> list of reader outputs
    for force in forces:
        ctype = _canonical(force.get("type"))
        if ctype not in _ACTUATOR_BUILDERS:
            print(f"actuator_parser: skipping force '{force.get('name')}' "
                  f"of type '{force.get('type')}' (not an actuator type we build).")
            continue
        reader = _OSIM_READERS.get(ctype)
        if reader is None:
            print(f"actuator_parser: actuator type '{ctype}' is buildable but has "
                  f"no OSIM reader yet; skipping force '{force.get('name')}'.")
            continue
        groups.setdefault(ctype, []).append(reader(force))
    # OSIM has no MuJoCo-style <default> block, so defaults is None per group.
    return [(ctype, items, None) for ctype, items in groups.items()]


def _parse_xml(root):
    defaults = root.find("default")   # root-level <default> block
    groups = {}  # type -> list of reader outputs

    # Build a reverse map: child tag -> (canonical type, reader). This lets us
    # read the type from each CHILD's tag (MuJoCo style: <general>/<motor>/
    # <muscle> directly under <actuator>, with no type on the wrapper).
    tag_to_type = {}
    for ctype, (tags, reader) in _XML_READERS.items():
        for tag in tags:
            tag_to_type[tag] = (ctype, reader)

    # If the ROOT declares a type (standalone-file style), we still support it:
    declared = _canonical(root.get("type") or "")
    for el in root:
        if el.tag == "default":
            continue
        # Prefer the child's own tag; fall back to the root-declared type.
        if el.tag in tag_to_type:
            ctype, reader = tag_to_type[el.tag]
        elif declared in _XML_READERS:
            ctype = declared
            _, reader = _XML_READERS[declared]
        else:
            raise ValueError(
                f"actuator_parser: actuator element <{el.tag}> has no XML reader. "
                f"Its tag isn't a known actuator tag and the root declares no "
                f"known type. Register the tag in _XML_READERS (and a builder in "
                f"_ACTUATOR_BUILDERS)."
            )
        groups.setdefault(ctype, []).append(reader(el))

    if not groups:
        raise ValueError(
            f"actuator_parser: no actuator elements found under <{root.tag}>."
        )

    # Carry the defaults block alongside each group so the builder can use it.
    return [(ctype, items, defaults) for ctype, items in groups.items()]


# ----------------------------------------------------------------------
# Helpers + assembly
# ----------------------------------------------------------------------
def _parse_axis(axis_val):
    """OSIM axis may be an 'x y z' string or already a list; normalize to a list."""
    if axis_val is None:
        return [0.0, 0.0, 1.0]
    if isinstance(axis_val, str):
        return [float(x) for x in axis_val.split()]
    return [float(x) for x in axis_val]


def _assemble(groups, joints, joint_names):
    """Build each (type, items, defaults) group into its actuator object. One
    group -> that object; several -> a MultiActuator wrapper."""
    if not groups:
        raise ValueError("No actuators could be built from the provided source.")
    models = [_ACTUATOR_BUILDERS[ctype](items, joints, joint_names, defaults)
              for ctype, items, defaults in groups]
    return models[0] if len(models) == 1 else MultiActuator(models)