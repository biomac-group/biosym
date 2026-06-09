

def get_rnea_equations(model):
    """
    Compute inverse dynamics using Recursive Newton-Euler Algorithm (RNEA).
    
    This function implements the RNEA algorithm to compute the joint torques/forces
    required to produce a given motion (positions, velocities, accelerations) of a
    multibody system. The algorithm consists of two passes:
    
    1. Forward pass: Compute velocities and accelerations of all bodies
    2. Backward pass: Compute forces, moments, and joint torques recursively from 
       leaves to root
    
    The implementation accounts for:
    - Gravitational forces
    - Inertial forces (linear and angular)
    - Gyroscopic effects (Coriolis and centrifugal)
    - External forces and moments applied to bodies
    - Both revolute (hinge) and prismatic (slide) joints
    
    Parameters
    ----------
    model : Model
        The biomechanical model containing:
        - Body origins, reference frames, and topology
        - Mass properties (masses, inertias, COM offsets)
        - Joint definitions (types, axes, parent-child relationships)
        - State variables (positions, velocities, accelerations)
        - External forces and moments
    
    Returns
    -------
    dict
        Dictionary mapping joint names to symbolic expressions for required
        joint torques (for revolute joints) or forces (for prismatic joints).
        Keys are joint names (str), values are SymPy expressions in terms of
        model state variables and parameters.
    
    Notes
    -----
    - All kinematics (velocities, accelerations) are computed in body frames for
      numerical efficiency, but represent motion relative to the ground frame
    - Forces and moments are propagated from child to parent bodies
    - External forces/moments are applied to bodies as specified in model.ext_forces
      and model.ext_torques
    - The algorithm assumes a tree topology (no kinematic loops)
    
    References
    ----------
    .. [1] Featherstone, R. (2008). Rigid Body Dynamics Algorithms. Springer.
    .. [2] Jain, A. (2010). Robot and Multibody Dynamics: Analysis and Algorithms.
    
    Examples
    --------
    >>> from biosym.model.model import load_model
    >>> model = load_model('path/to/model.yaml')
    >>> joint_torques = get_rnea_equations(model)
    >>> # joint_torques contains symbolic expressions for each joint
    """

    # Step 1.1: Forward Kinematics to get velocities and accelerations
    FK = []
    FK_dot = []
    Fk_ddot = []
    ang_ = []
    ang_vel_ = []
    ang_acc_ = []
    for body, point in model.body_origins.items():
        reference_frame = model.reference_frames[body]
        FK.append(
            [model._replace_dyn(
                point.pos_from(model.origin).dot(frame_dim))
                for frame_dim in [reference_frame.x, reference_frame.y, reference_frame.z]
            ]
        )
        FK_dot.append(
            [model._replace_dyn(
                point.vel(model.ground_frame).dot(frame_dim))
                for frame_dim in [reference_frame.x, reference_frame.y, reference_frame.z]
            ]
        )
        Fk_ddot.append(
            [model._replace_dyn(
                point.acc(model.ground_frame).dot(frame_dim), replace_d_q=True)
                for frame_dim in [reference_frame.x, reference_frame.y, reference_frame.z]
            ]
        )
        ang_vel_.append(
            [model._replace_dyn(
                reference_frame.ang_vel_in(model.ground_frame).dot(frame_dim))
                for frame_dim in [reference_frame.x, reference_frame.y, reference_frame.z]
            ]
        )
        ang_acc_.append(
            [model._replace_dyn(
                reference_frame.ang_acc_in(model.ground_frame).dot(frame_dim), replace_d_q=True)
                for frame_dim in [reference_frame.x, reference_frame.y, reference_frame.z]
            ]
        )


    # Step 1.2: RNEA to get forces and torques

    import sympy as sp

    # Initialize force and moment dictionaries
    body_forces = {}   # f_i: force on body i expressed in body i frame
    body_moments = {}  # n_i: moment on body i expressed in body i frame
    joint_torques = {} # tau_i: torque at joint i

    # Gravity vector in ground frame
    gravity_ground = sp.Matrix([
        model._v[model.g['idx'] + 0],
        model._v[model.g['idx'] + 1],
        model._v[model.g['idx'] + 2],
    ])  # Gravity vector as column vector



    def _inertia_from_vector(inertia_vec):
        """Convert inertia vector [Ixx, Iyy, Izz, Ixy, Iyz, Izx] to inertia tensor."""
        Ixx, Iyy, Izz, Ixy, Iyz, Izx = inertia_vec
        return sp.Matrix([[Ixx, Ixy, Izx],
                         [Ixy, Iyy, Iyz],
                         [Izx, Iyz, Izz]])

    def _find_in_topology_tree(body_name, tree):
        for curr_body in tree:
            if curr_body['name'] == body_name:
                return curr_body
            else:
                found = _find_in_topology_tree(body_name, curr_body.get('children', []))
                if found:
                    return found
        return None

    def _get_joints_for_body(body_name, model):
        """Get all joints where this body is the child."""
        list_of_joints = []
        for joint in model.dicts['joints']:
            if joint['child'] == body_name:
                list_of_joints.append(joint)
        return list_of_joints



    # Iterate in REVERSE topological order (leaves to root)
    for body_name in reversed(list(model.body_origins.keys())):

        body_idx = list(model.body_origins.keys()).index(body_name)

        # Get body properties
        mass = model._v[model.masses['idx']+body_idx]
        inertia = _inertia_from_vector([model._v[model.inertia['idx']+i+ 6* body_idx] for i in range(6)])  # Inertia tensor in body frame
        com_offset = [model._v[model.com['idx']+i+ 3* body_idx] for i in range(3)]  # CoM offset from body origin in body frame
        # Get kinematics from Step 1
        omega = sp.Matrix(ang_vel_[body_idx])  # Angular velocity
        alpha = sp.Matrix(ang_acc_[body_idx])  # Angular acceleration
        a_origin = sp.Matrix(Fk_ddot[body_idx])  # Linear acceleration of body origin

        # Compute acceleration of CoM: a_com = a_origin + alpha × r_com + omega × (omega × r_com)
        a_com = a_origin + sp.Matrix(alpha).cross(sp.Matrix(com_offset)) + sp.Matrix(omega).cross(sp.Matrix(omega).cross(sp.Matrix(com_offset)))

        # Transform gravity to body frame
        body_frame = model.reference_frames[body_name]
        R_bg = body_frame.dcm(model.ground_frame)   # R_body_ground
        gravity_body = model._replace_dyn(R_bg) * gravity_ground


        # Compute forces on body: f_i = mass * a_com - mass * gravity_body      
        f_i = mass * a_com - mass * gravity_body

        # Compute moments: n_i = I * alpha + omega × (I * omega) + r_com × (mass * a_com - mass * gravity_body)
        # The moment includes:
        # 1. Angular inertia effects: I * alpha + omega × (I * omega)
        # 2. Moment due to linear forces at CoM: r_com × f_com
        I_omega = inertia * omega 
        f_com = mass * a_com - mass * gravity_body  # Force at CoM
        n_i = inertia * alpha + sp.Matrix(omega).cross(sp.Matrix(I_omega)) + sp.Matrix(com_offset).cross(f_com)

        # Add forces from child bodies (if any)
        for child_body in _find_in_topology_tree(body_name, model.topology_tree).get('children', []):
            child_name = child_body['name']

            # Get child force and moment
            f_child = body_forces[child_name]
            n_child = body_moments[child_name]

            # Get rotation matrix from child frame to parent frame
            child_frame = model.reference_frames[child_name]
            parent_frame = model.reference_frames[body_name]

            # Build rotation matrix by projecting child axes onto parent axes
            # R_child_to_parent = model._replace_dyn(sp.Matrix([
            #     [child_frame.x.dot(parent_frame.x), child_frame.y.dot(parent_frame.x), child_frame.z.dot(parent_frame.x)],
            #     [child_frame.x.dot(parent_frame.y), child_frame.y.dot(parent_frame.y), child_frame.z.dot(parent_frame.y)],
            #     [child_frame.x.dot(parent_frame.z), child_frame.y.dot(parent_frame.z), child_frame.z.dot(parent_frame.z)]
            # ]))
            R_child_to_parent = model._replace_dyn(parent_frame.dcm(child_frame))


            # Get position of child joint (child origin) relative to parent origin, in parent frame
            child_origin = model.body_origins[child_name]
            parent_origin = model.body_origins[body_name]
            r_joint_vec = child_origin.pos_from(parent_origin)
            r_joint = model._replace_dyn(sp.Matrix([
                r_joint_vec.dot(parent_frame.x),
                r_joint_vec.dot(parent_frame.y),
                r_joint_vec.dot(parent_frame.z)
            ]))

            # Transform child force to parent frame and add
            f_child_in_parent = R_child_to_parent @ f_child
            f_i += f_child_in_parent

            # Add moment due to force offset + transformed child moment
            n_i += r_joint.cross(f_child_in_parent) + R_child_to_parent @ n_child

        # Add external forces and moments (only for root body - first in iteration)
        # Check for external moments
        if f"m_{body_name}_x" in model.ext_torques['names']:
            ext_moment_idx = model.ext_torques['names'].index(f"m_{body_name}_x")
            # External moments are 3D vectors
            ext_moment_vals = sp.Matrix([
                model._v[model.ext_torques['idx'] + ext_moment_idx + i]
                for i in range(3)
            ])
            # Transform from ground frame to body frame
            ext_moment_body = model._replace_dyn(R_bg) * ext_moment_vals
            n_i -= ext_moment_body  # Subtract because external moments oppose required torques

        # Check for external forces  
        if f"f_{body_name}_x" in model.ext_forces['names']:
            ext_force_idx = model.ext_forces['names'].index(f"f_{body_name}_x")
            # External forces are 3D vectors
            ext_force_vals = sp.Matrix([
                model._v[model.ext_forces['idx'] + ext_force_idx + i]
                for i in range(3)
            ])
            # Transform from ground frame to body frame
            ext_force_body = model._replace_dyn(R_bg) * ext_force_vals
            f_i -= ext_force_body  # Subtract because external forces oppose required forces

        # Store results
        body_forces[body_name] = f_i
        body_moments[body_name] = n_i

        # Compute joint torque: tau_i = n_i · joint_axis_i (for revolute) or f_i · joint_axis_i (for prismatic)
        # Get the joints where this body is the child
        joints = _get_joints_for_body(body_name, model)

        for i, joint in enumerate(joints):
            joint_name = joint['name']
            joint_type = joint['type']

            # Get joint axis in parent frame (as specified in the joint definition)
            joint_axis_parent = sp.Matrix(joint['axis'])

            # Transform joint axis to body frame
            if body_name in model._intermediate_frames:
                parent_frame = model._intermediate_frames[body_name][i]
            else:
                parent_frame = model.reference_frames[joint['parent']] if joint['parent'] != 'ground_frame' else model.ground_frame
            body_frame = model.reference_frames[body_name]
        

            # Project parent frame axis onto body frame
            joint_axis_body = model._replace_dyn(sp.Matrix([
                joint_axis_parent[0] * parent_frame.x.dot(body_frame.x) + 
                joint_axis_parent[1] * parent_frame.y.dot(body_frame.x) + 
                joint_axis_parent[2] * parent_frame.z.dot(body_frame.x),
                joint_axis_parent[0] * parent_frame.x.dot(body_frame.y) + 
                joint_axis_parent[1] * parent_frame.y.dot(body_frame.y) + 
                joint_axis_parent[2] * parent_frame.z.dot(body_frame.y),
                joint_axis_parent[0] * parent_frame.x.dot(body_frame.z) + 
                joint_axis_parent[1] * parent_frame.y.dot(body_frame.z) + 
                joint_axis_parent[2] * parent_frame.z.dot(body_frame.z)
            ]))


            # For revolute/hinge joints: project moment onto joint axis
            # For prismatic/slide joints: project force onto joint axis         
            if joint_type == 'hinge':
                tau = n_i.dot(joint_axis_body)
            elif joint_type == 'slide':
                tau = f_i.dot(joint_axis_body)
            else:
                raise ValueError(f"Unknown joint type: {joint_type}")

            if f"M_{joint_name}" in model.forces['names']:
                tau -= model._v[model.forces['idx'] + model.forces['names'].index(f"M_{joint_name}")]

            joint_torques[joint_name] = tau
    
    return joint_torques
