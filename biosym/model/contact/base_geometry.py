"""
Abstract base class for contact geometries.

A "geometry" describes the shape and placement of a contact-capable
feature attached to a body (a point, sphere, half-space, mesh, ...). It is
solely responsible for:

- storing shape-specific parameters (radius, plane normal, etc.)
- building the symbolic kinematics (position & velocity of its reference
  point, in the model's ground frame) as a function of the model's
  generalized coordinates
- evaluating that position/velocity numerically given states/constants
- (optionally) reporting the geometric penetration/overlap against another
  geometry, when such a pairing is physically meaningful (e.g. a sphere
  against a half-space)

A geometry knows nothing about force laws. Stiffness, damping, and
friction coefficients, and the resulting force calculation, live in
BaseContact subclasses (biosym.model.contact.base_contact.BaseContact).

See Also
--------
    biosym.model.contact.contact_geometries.contact_point.ContactPoint
    biosym.model.contact.contact_geometries.contact_sphere.ContactSphere
    biosym.model.contact.contact_geometries.contact_halfspace.ContactHalfSpace
"""

from abc import ABC, abstractmethod

from sympy import Matrix, lambdify, sqrt
from sympy.physics.mechanics import Point

# Registry mapping an ordered (feature_type_name, surface_type_name) pair to
# the function that computes that pair's penetration. Populated by the geometry
# subclass modules at import time via @register_penetration -- so the base class
# never imports its own subclasses (no circular imports) and no geometry needs
# to know about any other. The pairwise math for each pair lives in exactly one
# place.
_PENETRATION_RULES = {}

def register_penetration(type_a, type_b):
    """
    Decorator registering a penetration rule for an ordered pair of geometry
    types, identified by class name. The decorated function takes (a, b) -- two
    BaseGeometry instances whose build_kinematics has already been called -- and
    returns the standard penetration dict.

    Lives in the geometry subclass modules, not here, e.g. in contact_sphere.py:

        @register_penetration("ContactSphere", "ContactHalfSpace")
        def _sphere_vs_plane(sphere, plane):
            return sphere._penetration_against_plane(plane, radius=sphere.radius)
    """
    def decorator(fn):
        _PENETRATION_RULES[(type_a, type_b)] = fn
        return fn
    return decorator

class BaseGeometry(ABC):
    """
    Abstract base class for contact geometries.

    Parameters
    ----------
    name : str
        Name of the geometry (used for bookkeeping/plotting/debugging).
    parent_body : str
        Name of the body this geometry is rigidly attached to.
    pos : array of float, optional
        Offset of the geometry's reference point from the parent body's
        origin, expressed in the parent body's frame. Defaults to [0, 0, 0].
    orientation : array of float, optional
        Orientation of the geometry relative to the parent body's frame
        (XYZ Euler angles, consistent with the rest of biosym). Mainly
        relevant for geometries with a meaningful normal/axis, such as
        ContactHalfSpace. Defaults to [0, 0, 0].

    Attributes
    ----------
    pos_expr : sympy.Matrix or None
        Symbolic (3, 1) position of the geometry's reference point in the
        ground frame. Set by build_kinematics().
    vel_expr : sympy.Matrix or None
        Symbolic (3, 1) velocity of the geometry's reference point in the
        ground frame. Set by build_kinematics().
    position_fn : Callable or None
        Lambdified function returning the geometry's reference point
        position in the ground frame. Set by build_kinematics().
    velocity_fn : Callable or None
        Lambdified function returning the geometry's reference point
        velocity in the ground frame. Set by build_kinematics().

    Notes
    -----
    Concrete subclasses live in biosym.model.contact.contact_geometries
    (e.g. ContactPoint, ContactSphere, ContactHalfSpace). Instantiation is
    the responsibility of contact_parser.py, which is also responsible for
    deciding which geometries are passed to which BaseContact subclass.
    """

    def __init__(self, name, parent_body, pos=None, orientation=None, **kwargs):
        self.name = name
        self.parent_body = parent_body
        self.pos = list(pos) if pos is not None else [0.0, 0.0, 0.0]
        self.orientation = list(orientation) if orientation is not None else [0.0, 0.0, 0.0]
        self.pos_expr = None
        self.vel_expr = None
        self.position_fn = None
        self.velocity_fn = None

    # ------------------------------------------------------------------
    # Shared kinematics helpers
    # ------------------------------------------------------------------
    def _resolve_frame_and_origin(self, model):
        """
        Resolve this geometry's parent reference frame and origin point.

        Handles the special "ground_frame" sentinel the same way model.py
        does elsewhere: the ground is model.ground_frame / model.origin,
        which are NOT stored in reference_frames / body_origins. Any
        geometry attached to ground (e.g. a ContactHalfSpace acting as the
        terrain) goes through here.
        """
        if self.parent_body == "ground_frame":
            return model.ground_frame, model.origin
        return model.reference_frames[self.parent_body], model.body_origins[self.parent_body]

    def _point_kinematics(self, model, offset=None):
        """
        Build symbolic position & velocity expressions for a single point
        fixed on self.parent_body, offset from the body's origin by
        `offset` (in the body's own frame). Shared for any
        geometry whose contact-relevant reference is a single fixed point
        on the body (e.g. ContactPoint, ContactSphere).

        Parameters
        ----------
        model : biosym.model.model.BiosymModel
        offset : array of float, optional
            Offset from the body origin, in the body's frame. Defaults to
            self.pos.

        Returns
        -------
        sympy.Matrix, sympy.Matrix
            Position and velocity, each a (3, 1) column matrix in the
            ground frame, already passed through model._replace_dyn.
        """
        offset = offset if offset is not None else self.pos
        ref_frame, origin = self._resolve_frame_and_origin(model)

        point = Point(f"cp_{self.name}")
        point.set_pos(
            origin,
            ref_frame.x * offset[0] + ref_frame.y * offset[1] + ref_frame.z * offset[2],
        )

        point.v2pt_theory(origin, model.ground_frame, ref_frame)

        pos_expr = Matrix([
            point.pos_from(model.origin).dot(frame_dim)
            for frame_dim in (model.ground_frame.x, model.ground_frame.y, model.ground_frame.z)
        ])
        vel_expr = Matrix([
            point.vel(model.ground_frame).dot(frame_dim)
            for frame_dim in (model.ground_frame.x, model.ground_frame.y, model.ground_frame.z)
        ])

        return model._replace_dyn(pos_expr), model._replace_dyn(vel_expr)

    def _direction_kinematics(self, model, local_direction):
        """
        Express a direction vector fixed in self.parent_body's frame (a
        plain numeric 3-vector in the body's frame) in the ground frame.
        See class docstring; returns a (3, 1) matrix through
        model._replace_dyn.
        """
        ref_frame, _ = self._resolve_frame_and_origin(model)
        direction = (
            local_direction[0] * ref_frame.x
            + local_direction[1] * ref_frame.y
            + local_direction[2] * ref_frame.z
        )
        dir_expr = Matrix([
            direction.dot(frame_dim)
            for frame_dim in (model.ground_frame.x, model.ground_frame.y, model.ground_frame.z)
        ])
        return model._replace_dyn(dir_expr)

    def _lambdify_kinematics(self, model, pos_expr, vel_expr):
        """
        Lambdify symbolic position/velocity expressions and store the
        resulting callables as self.position_fn / self.velocity_fn.
        """
        self.position_fn = lambdify(model._symbols, pos_expr, modules="jax", cse=True, docstring_limit=2)
        self.velocity_fn = lambdify(model._symbols, vel_expr, modules="jax", cse=True, docstring_limit=2)

    # ------------------------------------------------------------------
    # Abstract / overridable interface
    # ------------------------------------------------------------------
    @abstractmethod
    def build_kinematics(self, model):
        """
        Build the symbolic position & velocity of this geometry's reference
        point in the model's ground frame, store them on self.pos_expr /
        self.vel_expr, and lambdify them into self.position_fn /
        self.velocity_fn.

        Parameters
        ----------
        model : biosym.model.model.BiosymModel
            The model this geometry is attached to. Used to access
            reference frames, body origins, and the symbolic variable
            ordering needed for lambdify.

        Notes
        -----
        Called once per geometry, triggered by the owning BaseContact's
        process_eom().
        """

    @abstractmethod
    def get_parameters(self):
        """
        Get this geometry's shape-specific parameters.

        Returns
        -------
        dict
            Shape parameters, e.g. {'radius': 0.05} for a sphere,
            {} for a point or half-space.
        """

    def get_position(self, states, constants):
        """
        Evaluate the geometry's reference point position numerically.

        Parameters
        ----------
        states : object
            Current state values for the model.
        constants : object
            Current constant parameter values.

        Returns
        -------
        numpy.ndarray or jax.Array
            Position of the geometry's reference point in the ground frame.
        """
        if self.position_fn is None:
            raise RuntimeError(
                f"Geometry '{self.name}' kinematics have not been built yet. "
                "build_kinematics(model) must be called before get_position()."
            )
        return self.position_fn(*states.filter('model').flatten(), *constants.filter('model').flatten())

    def get_velocity(self, states, constants):
        """
        Evaluate the geometry's reference point velocity numerically.

        Parameters
        ----------
        states : object
            Current state values for the model.
        constants : object
            Current constant parameter values.

        Returns
        -------
        numpy.ndarray or jax.Array
            Velocity of the geometry's reference point in the ground frame.
        """
        if self.velocity_fn is None:
            raise RuntimeError(
                f"Geometry '{self.name}' kinematics have not been built yet. "
                "build_kinematics(model) must be called before get_velocity()."
            )
        return self.velocity_fn(*states.filter('model').flatten(), *constants.filter('model').flatten())

    def get_n_constants(self):
        """
        Get the number of additional optimization constants this geometry
        contributes (e.g. an optimizable radius). Defaults to 0; override
        in subclasses that expose shape parameters as constants.
        """
        return 0

    def get_constants(self):
        """
        Get the names of additional optimization constants this geometry
        contributes. Defaults to an empty list.
        """
        return []

    def _penetration_decompose(self, other, normal, depth):
        """
        Shared tail of every penetration rule. Given the contact `normal` (unit
        vector pointing the way `self` gets pushed) and the scalar `depth`
        (positive when penetrating), split the relative velocity into its normal
        and tangential parts and assemble the standard dict. Everything that
        differs between contact pairs is in computing `normal` and `depth`; this
        part is universal.
        """
        relative_vel = self.vel_expr - other.vel_expr
        normal_speed = relative_vel.dot(normal)        # rate of separation along normal
        normal_velocity = -normal_speed                # positive => penetration increasing
        tangential_velocity = relative_vel - normal_speed * normal

        return {
            "depth": depth,
            "normal_velocity": normal_velocity,
            "tangential_velocity": tangential_velocity,
            "normal": normal,
        }

    def _penetration_against_plane(self, other, radius=0.0):
        """
        Penetration of self's reference point -- inflated by `radius` (0 for a
        point, r for a sphere) -- against a plane-like geometry `other` that
        exposes a symbolic normal_expr (a ContactHalfSpace, in practice).
        """
        if getattr(other, "normal_expr", None) is None:
            raise NotImplementedError(
                f"_penetration_against_plane expects '{type(other).__name__}' to "
                f"expose a symbolic 'normal_expr' (e.g. a ContactHalfSpace, after "
                f"build_kinematics() has been called on it)."
            )
        relative_pos = self.pos_expr - other.pos_expr
        normal = other.normal_expr
        gap = relative_pos.dot(normal)     # signed distance plane -> self's center, along normal
        depth = radius - gap               # positive once self's surface crosses the plane
        return self._penetration_decompose(other, normal, depth)
    
    def _penetration_against_sphere(self, other, radius_self, radius_other):
        """
        Penetration of two spheres (a point being the radius-0 case), centered
        at self.pos_expr and other.pos_expr with the given radii.

        Unlike the plane case, the normal isn't read off a fixed surface -- it's
        the line joining the two centers, pointing from `other` toward `self`
        (i.e. the direction `self` is pushed), and it's configuration-dependent
        because both centers move.
        """
        relative_pos = self.pos_expr - other.pos_expr
        center_distance = sqrt(relative_pos.dot(relative_pos))
        normal = relative_pos / center_distance                 # unit, points other -> self
        depth = (radius_self + radius_other) - center_distance   # positive when overlapping
        return self._penetration_decompose(other, normal, depth)
    
    def penetration(self, other):
        """
        Compute the penetration between this geometry and `other` by looking up
        the registered rule for their (type, type) pair. The actual math lives
        in @register_penetration-decorated functions in the geometry modules, so
        the base class stays ignorant of which concrete geometries exist.

        Raises
        ------
        NotImplementedError
            If no rule is registered for this ordered pair of geometry types.
        """
        key = (type(self).__name__, type(other).__name__)
        rule = _PENETRATION_RULES.get(key)
        if rule is None:
            raise NotImplementedError(
                f"No penetration rule registered for {key}. Add one with "
                f"@register_penetration in the relevant geometry module."
            )
        return rule(self, other)
    
    def plot(self, ax, position, mode, **kwargs):
        """
        Draw this geometry's shape at a given position (e.g. a circle for a
        sphere, a plane patch for a half-space). Default is a no-op; the
        owning BaseContact is free to handle plotting itself if a simple
        marker is enough.
        """
        pass

    def __repr__(self):
        return f"<{type(self).__name__} '{self.name}' on body '{self.parent_body}'>"