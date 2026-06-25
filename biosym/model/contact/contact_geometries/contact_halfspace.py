
import numpy as np
from sympy import lambdify

from biosym.model.contact.base_geometry import BaseGeometry
from biosym.utils import useful_functions as uf


class ContactHalfSpace(BaseGeometry):
    """
    An infinite flat half-space (plane), rigidly attached to a body, used as
    the ground/terrain side of a contact pair. Maps directly onto OpenSim's
    ContactHalfSpace geometry, e.g.:

        {'name': 'ground_contact', 'type': 'ContactHalfSpace',
         'parent_body': 'ground_frame', 'location': [0.0, 0.0, 0.0],
         'orientation': [0.0, 0.0, -1.57], 'parameters': {}}

    Parameters
    ----------
    name : str
        Name of the half-space (e.g. "ground_contact" in OpenSim).
    parent_body : str
        Name of the body this half-space is rigidly attached to. Usually a
        static ground/world body, but doesn't have to be -- a half-space on a
        moving body (treadmill belt, tilting platform) is handled the same
        way, since its point and normal are both expressed via the body's
        own (possibly time-varying) reference frame.
    pos : array of float, optional
        A point on the plane, as an offset from the parent body's origin, in
        the body's frame. Defaults to [0, 0, 0]. Corresponds to OpenSim's
        "location".
    orientation : array of float, optional
        XYZ Euler angles (same convention as the rest of the parser -- see
        biosym.utils.useful_functions.rotation_matrix_xyz) describing the
        half-space's rotation relative to the parent body's frame. Defaults
        to [0, 0, 0]. Corresponds to OpenSim's "orientation".

    Attributes
    ----------
    normal_expr : sympy.Matrix or None
        Symbolic (3, 1) outward unit normal in the ground frame. Set by
        build_kinematics().
    normal_fn : Callable or None
        Lambdified outward unit normal. Set by build_kinematics().

    Notes
    -----
    OpenSim's convention is that a ContactHalfSpace's outward normal is its
    local +X axis before "orientation" is applied. "orientation" is what
    rotates that local +X into whatever direction is actually "up" for a
    given model. We reuse uf.rotation_matrix_xyz here -- the same utility
    osim_parser.py uses for joint/body orientations -- so this stays
    consistent with the rest of the codebase. If forces point the wrong way
    for some model, check here first: evaluate get_normal() at the default
    pose.

    A half-space owns no penetration rules: it's always the "surface" side
    of a pair (the `other` argument), never the feature that calls
    .penetration(). It just needs to expose normal_expr, which the shared
    _penetration_against_plane helper reads.
    """

    def __init__(self, name, parent_body, pos=None, orientation=None, **kwargs):
        super().__init__(name, parent_body, pos=pos, orientation=orientation, **kwargs)
        self.normal_expr = None
        self.normal_fn = None

    def build_kinematics(self, model):
        # Reference point on the plane -- identical math to any body-fixed point.
        self.pos_expr, self.vel_expr = self._point_kinematics(model)
        self._lambdify_kinematics(model, self.pos_expr, self.vel_expr)

        # OpenSim's convention is that a ContactHalfSpace's solid region is local
        # +X (points with x > 0 are inside), so the OUTWARD contact normal is local
        # -X. "orientation" rotates that outward normal into whatever direction is
        # actually "up" for a given model. We reuse uf.rotation_matrix_xyz here.
        local_normal = uf.rotation_matrix_xyz(self.orientation) @ np.array([-1.0, 0.0, 0.0])
        self.normal_expr = self._direction_kinematics(model, local_normal)
        self.normal_fn = lambdify(model._v, self.normal_expr, modules="jax", cse=True, docstring_limit=2)

    def get_parameters(self):
        """A half-space is infinite and has no shape parameters beyond its
        placement, which already lives on the base class (pos/orientation)."""
        return {}

    def get_normal(self, states, constants):
        """
        Evaluate the half-space's outward unit normal direction numerically,
        in the ground frame, for the current state.
        """
        if self.normal_fn is None:
            raise RuntimeError(
                f"Geometry '{self.name}' kinematics have not been built yet. "
                "build_kinematics(model) must be called before get_normal()."
            )
        return self.normal_fn(*states.model, *constants.model)
