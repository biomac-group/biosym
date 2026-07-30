
from biosym.model.contact.base_geometry import BaseGeometry, register_penetration


class ContactSphere(BaseGeometry):
    """
    A sphere of fixed radius, rigidly attached to a body, used as a
    contact-capable geometry. Maps directly onto OpenSim's ContactSphere
    geometry, parsed with OsimParser with the structure e.g.:

        {'name': 'foretoe_r', 'type': 'ContactSphere', 'parent_body': 'foretoe_r',
         'location': [0.028, -0.049, 0.002], 'orientation': [0.0, -1.57, -3.14], 
         'parameters': {'radius': '0.05'}}

    Parameters
    ----------
    name : str
        Name of the sphere.
    parent_body : str
        Name of the body this sphere is rigidly attached to.
    pos : array of float, optional
        Offset of the sphere's center from the parent body's origin, in the
        body's frame. Defaults to [0, 0, 0]. Corresponds to OpenSim's
        "location".
    radius : float
        Radius of the sphere. When constructed from osim_parser output,
        parameters["radius"] arrives as a string and must be cast to float
        by the caller (contact_parser.py) before being passed here.

    Notes
    -----
    A sphere's contact-relevant kinematics (position/velocity of its center)
    are identical to ContactPoint's -- the radius only enters the penetration
    calculation, where it shifts the effective surface inward by r compared
    to a bare point.

    "orientation" from the OpenSim data is intentionally unused: a sphere
    looks the same from every angle, so it has no bearing on contact
    behavior. We don't even store it, unlike pos/radius.

    The radius is currently a fixed float, not an optimization constant. If
    you ever want an optimizable radius, override get_n_constants/get_constants
    here (and have the force law read it from the state vector rather than
    from self.radius).
    """

    def __init__(self, name, parent_body, pos=None, radius=0.0, **kwargs):
        super().__init__(name, parent_body, pos=pos, **kwargs)
        self.radius = float(radius)

    def build_kinematics(self, model):
        # Contact-relevant reference point is the sphere's center -- same
        # math as ContactPoint, via the shared BaseGeometry helper. Radius
        # plays no role in position/velocity, only in penetration.
        self.pos_expr, self.vel_expr = self._point_kinematics(model)
        self._lambdify_kinematics(model, self.pos_expr, self.vel_expr)

    def get_parameters(self):
        return {"radius": self.radius}

# ----------------------------------------------------------------------
# Penetration rules owned by ContactSphere, registered at import time.
# ----------------------------------------------------------------------
@register_penetration("ContactSphere", "ContactHalfSpace")
def _sphere_vs_plane(sphere, plane):
    return sphere._penetration_against_plane(plane, radius=sphere.radius)


@register_penetration("ContactSphere", "ContactSphere")
def _sphere_vs_sphere(a, b):
    return a._penetration_against_sphere(b, a.radius, b.radius)
