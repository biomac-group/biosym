
from biosym.model.contact.base_geometry import BaseGeometry, register_penetration


class ContactPoint(BaseGeometry):
    """
    A single fixed point on a body, used as a contact-capable geometry.

    Direct geometry-side counterpart of the contact points used in the
    legacy yaml-based contact_points model: a point rigidly attached to a
    body at a fixed offset, with no further shape of its own.

    Parameters
    ----------
    name : str
        Name of the contact point.
    parent_body : str
        Name of the body this point is rigidly attached to.
    pos : array of float, optional
        Offset from the parent body's origin, in the body's frame.
        Defaults to [0, 0, 0].
    """

    def __init__(self, name, parent_body, pos=None, **kwargs):
        super().__init__(name, parent_body, pos=pos, **kwargs)

    def build_kinematics(self, model):
        self.pos_expr, self.vel_expr = self._point_kinematics(model)
        self._lambdify_kinematics(model, self.pos_expr, self.vel_expr)

    def get_parameters(self):
        """A point has no shape parameters beyond its position."""
        return {}


# ----------------------------------------------------------------------
# Penetration rules owned by ContactPoint, registered at import time.
# A point is the radius-0 case of the sphere.
# ----------------------------------------------------------------------
@register_penetration("ContactPoint", "ContactHalfSpace")
def _point_vs_plane(point, plane):
    return point._penetration_against_plane(plane, radius=0.0)
