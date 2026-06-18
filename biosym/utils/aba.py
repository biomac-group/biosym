from functools import partial

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax
except ImportError:
    jax = None
    jnp = None
    jit = None
    lax = None

HINGE_JOINT = 1
SLIDE_JOINT = 2
DEFAULT_ABA_REGULARIZATION = 0 #1e-10


def get_aba_equations(model):
    """Symbolic ABA generation is no longer supported in this module."""
    raise NotImplementedError("aba.py no longer provides SymPy-based ABA equations; use get_aba_jax_model().")


def aba_step1_kinematics(model, compute_propagation=True):
    """Symbolic ABA helpers are no longer supported in this module."""
    raise NotImplementedError("aba.py no longer exposes SymPy-based ABA steps; use get_aba_jax_model().")


def aba_step2_inertia(model, kinematics):
    """Symbolic ABA helpers are no longer supported in this module."""
    raise NotImplementedError("aba.py no longer exposes SymPy-based ABA steps; use get_aba_jax_model().")


def aba_step3_accelerations(model, kinematics, dynamics):
    """Symbolic ABA helpers are no longer supported in this module."""
    raise NotImplementedError("aba.py no longer exposes SymPy-based ABA steps; use get_aba_jax_model().")


def _build_parent_index_map(bodies):
    body_indices = {body["name"]: index for index, body in enumerate(bodies)}
    parent_indices = []
    children = {index: [] for index in range(len(bodies))}
    roots = []

    for index, body in enumerate(bodies):
        parent_name = body["parent"]
        if parent_name in body_indices:
            parent_index = body_indices[parent_name]
            parent_indices.append(parent_index)
            children[parent_index].append(index)
        else:
            parent_indices.append(-1)
            roots.append(index)

    ordered = []
    queue = list(roots)
    while queue:
        index = queue.pop(0)
        ordered.append(index)
        queue.extend(children[index])

    if len(ordered) != len(bodies):
        ordered = list(range(len(bodies)))

    reorder = {old_index: new_index for new_index, old_index in enumerate(ordered)}
    sorted_bodies = [bodies[index] for index in ordered]
    sorted_parents = []
    for old_index in ordered:
        parent_index = parent_indices[old_index]
        sorted_parents.append(reorder[parent_index] if parent_index != -1 else -1)

    return sorted_bodies, sorted_parents


def _force_index_map(model):
    mapping = {}
    for index, name in enumerate(model.forces["names"]):
        if name.startswith("M_"):
            mapping[name[2:]] = index
    return mapping


def _state_index_map(names, prefix):
    mapping = {}
    for index, name in enumerate(names):
        if name.startswith(prefix):
            mapping[name[len(prefix) :]] = index
        else:
            mapping[name] = index
    return mapping


def _constant_offset(model, index):
    return index - model.n_states


def _normalize_joint_axis(axis, joint_name, body_name):
    axis_array = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis_array)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Joint '{joint_name}' on body '{body_name}' must define a finite, non-zero axis.")
    return axis_array / norm


def _build_runtime_model(model):
    original_bodies = list(model.dicts["bodies"])
    original_body_indices = {body["name"]: index for index, body in enumerate(original_bodies)}
    bodies, parent_indices = _build_parent_index_map(original_bodies)

    q_index_map = _state_index_map(model.coordinates["names"], "q_")
    qd_index_map = _state_index_map(model.speeds["names"], "qd_")
    tau_index_map = _force_index_map(model)

    const_mass_start = _constant_offset(model, model.masses["idx"])
    const_inertia_start = _constant_offset(model, model.inertia["idx"])
    const_com_start = _constant_offset(model, model.com["idx"])
    const_offset_start = _constant_offset(model, model.offset["idx"])
    const_g_start = _constant_offset(model, model.g["idx"])
    const_g_indices = np.arange(const_g_start, const_g_start + 3, dtype=np.int32)

    ext_force_body_map = {
        "_".join(name.split("_")[1:-1]): 3 * index for index, name in enumerate(model.ext_forces["names"][::3])
    }
    ext_torque_body_map = {
        "_".join(name.split("_")[1:-1]): 3 * index for index, name in enumerate(model.ext_torques["names"][::3])
    }

    body_specs = []
    for body_index, body in enumerate(bodies):
        original_body_index = original_body_indices[body["name"]]
        joints = body.get("joints", [])
        joint_types = []
        joint_axes = []
        q_indices = []
        qd_indices = []
        tau_indices = []

        for joint in joints:
            joint_type = joint.get("type")
            if joint_type in {"hinge", "revolute"}:
                joint_types.append(HINGE_JOINT)
            elif joint_type in {"slide", "prismatic"}:
                joint_types.append(SLIDE_JOINT)
            else:
                joint_types.append(0)

            raw_axis = joint.get("axis", [0.0, 0.0, 0.0])
            if joint_type in {"hinge", "revolute", "slide", "prismatic"}:
                joint_axis = _normalize_joint_axis(raw_axis, joint["name"], body["name"])
            else:
                joint_axis = np.asarray(raw_axis, dtype=np.float64)

            joint_axes.append(joint_axis)
            joint_name = joint["name"]
            q_indices.append(q_index_map[joint_name])
            qd_indices.append(qd_index_map[joint_name])
            tau_indices.append(tau_index_map.get(joint_name, -1))

        ext_force_start = ext_force_body_map.get(body["name"])
        ext_torque_start = ext_torque_body_map.get(body["name"])

        body_specs.append(
            {
                "name": body["name"],
                "parent": parent_indices[body_index],
                "body_index": body_index,
                "original_body_index": original_body_index,
                "dof": len(joints),
                "joint_types": tuple(joint_types),
                "joint_axes": np.asarray(joint_axes, dtype=np.float64),
                "q_indices": np.asarray(q_indices, dtype=np.int32),
                "qd_indices": np.asarray(qd_indices, dtype=np.int32),
                "tau_indices": np.asarray(tau_indices, dtype=np.int32),
                "mass_index": const_mass_start + original_body_index,
                "inertia_indices": np.arange(
                    const_inertia_start + 6 * original_body_index,
                    const_inertia_start + 6 * (original_body_index + 1),
                    dtype=np.int32,
                ),
                "com_indices": np.arange(
                    const_com_start + 3 * original_body_index,
                    const_com_start + 3 * (original_body_index + 1),
                    dtype=np.int32,
                ),
                "offset_indices": np.arange(
                    const_offset_start + 3 * original_body_index,
                    const_offset_start + 3 * (original_body_index + 1),
                    dtype=np.int32,
                ),
                "ext_force_indices": None
                if ext_force_start is None
                else np.arange(ext_force_start, ext_force_start + 3, dtype=np.int32),
                "ext_torque_indices": None
                if ext_torque_start is None
                else np.arange(
                    model.ext_forces["n"] + ext_torque_start,
                    model.ext_forces["n"] + ext_torque_start + 3,
                    dtype=np.int32,
                ),
            }
        )

    return {
        "body_specs": tuple(body_specs),
        "g_indices": const_g_indices,
        "n_bodies": len(body_specs),
        "n_q": model.coordinates["n"],
        "n_qd": model.speeds["n"],
        "n_tau": model.forces["n"],
        "n_ext": model.ext_forces["n"] + model.ext_torques["n"],
    }


def _pack_runtime_model_for_jax(runtime_model):
    body_specs = runtime_model["body_specs"]
    n_bodies = runtime_model["n_bodies"]
    n_tau = runtime_model["n_tau"]
    n_ext = runtime_model["n_ext"]
    max_dof = max((spec["dof"] for spec in body_specs), default=0)
    zero_tau_index = n_tau
    zero_ext_index = n_ext

    parent_indices = np.full((n_bodies,), -1, dtype=np.int32)
    dof = np.zeros((n_bodies,), dtype=np.int32)
    joint_types = np.zeros((n_bodies, max_dof), dtype=np.int32)
    joint_axes = np.zeros((n_bodies, max_dof, 3), dtype=np.float64)
    q_indices = np.zeros((n_bodies, max_dof), dtype=np.int32)
    qd_indices = np.zeros((n_bodies, max_dof), dtype=np.int32)
    tau_lookup_indices = np.full((n_bodies, max_dof), zero_tau_index, dtype=np.int32)
    mass_indices = np.zeros((n_bodies,), dtype=np.int32)
    inertia_indices = np.zeros((n_bodies, 6), dtype=np.int32)
    com_indices = np.zeros((n_bodies, 3), dtype=np.int32)
    offset_indices = np.zeros((n_bodies, 3), dtype=np.int32)
    ext_force_indices = np.full((n_bodies, 3), zero_ext_index, dtype=np.int32)
    ext_torque_indices = np.full((n_bodies, 3), zero_ext_index, dtype=np.int32)
    has_ext_force = np.zeros((n_bodies,), dtype=np.bool_)
    has_ext_torque = np.zeros((n_bodies,), dtype=np.bool_)

    for body_index, spec in enumerate(body_specs):
        current_dof = spec["dof"]
        parent_indices[body_index] = spec["parent"]
        dof[body_index] = current_dof
        mass_indices[body_index] = spec["mass_index"]
        inertia_indices[body_index] = spec["inertia_indices"]
        com_indices[body_index] = spec["com_indices"]
        offset_indices[body_index] = spec["offset_indices"]

        if current_dof > 0:
            joint_types[body_index, :current_dof] = spec["joint_types"]
            joint_axes[body_index, :current_dof, :] = spec["joint_axes"]
            q_indices[body_index, :current_dof] = spec["q_indices"]
            qd_indices[body_index, :current_dof] = spec["qd_indices"]
            tau_lookup_indices[body_index, :current_dof] = np.where(
                spec["tau_indices"] >= 0,
                spec["tau_indices"],
                zero_tau_index,
            )

        if spec["ext_force_indices"] is not None:
            has_ext_force[body_index] = True
            ext_force_indices[body_index] = spec["ext_force_indices"]

        if spec["ext_torque_indices"] is not None:
            has_ext_torque[body_index] = True
            ext_torque_indices[body_index] = spec["ext_torque_indices"]

    return {
        "n_bodies": n_bodies,
        "n_qd": runtime_model["n_qd"],
        "n_tau": n_tau,
        "n_ext": n_ext,
        "max_dof": max_dof,
        "g_indices": jnp.asarray(runtime_model["g_indices"], dtype=jnp.int32),
        "parent_indices": jnp.asarray(parent_indices, dtype=jnp.int32),
        "dof": jnp.asarray(dof, dtype=jnp.int32),
        "joint_types": jnp.asarray(joint_types, dtype=jnp.int32),
        "joint_axes": jnp.asarray(joint_axes),
        "q_indices": jnp.asarray(q_indices, dtype=jnp.int32),
        "qd_indices": jnp.asarray(qd_indices, dtype=jnp.int32),
        "tau_lookup_indices": jnp.asarray(tau_lookup_indices, dtype=jnp.int32),
        "mass_indices": jnp.asarray(mass_indices, dtype=jnp.int32),
        "inertia_indices": jnp.asarray(inertia_indices, dtype=jnp.int32),
        "com_indices": jnp.asarray(com_indices, dtype=jnp.int32),
        "offset_indices": jnp.asarray(offset_indices, dtype=jnp.int32),
        "ext_force_indices": jnp.asarray(ext_force_indices, dtype=jnp.int32),
        "ext_torque_indices": jnp.asarray(ext_torque_indices, dtype=jnp.int32),
        "has_ext_force": jnp.asarray(has_ext_force),
        "has_ext_torque": jnp.asarray(has_ext_torque),
    }


def _inertia_matrix_from_vector(inertia_vec):
    i_xx, i_yy, i_zz, i_xy, i_xz, i_yz = inertia_vec
    return jnp.array(
        [
            [i_xx, i_xy, i_xz],
            [i_xy, i_yy, i_yz],
            [i_xz, i_yz, i_zz],
        ],
        dtype=inertia_vec.dtype,
    )


def _skew3(vec):
    x, y, z = vec[0], vec[1], vec[2]
    return jnp.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=vec.dtype,
    )


def _passive_rotation(axis, angle, dtype):
    axis_skew = _skew3(axis)
    eye = jnp.eye(3, dtype=dtype)
    return eye - jnp.sin(angle) * axis_skew + (1.0 - jnp.cos(angle)) * (axis_skew @ axis_skew)


def _body_inertia_blocks(mass, com, inertia_cm):
    com_skew = _skew3(com)
    inertia_ww = inertia_cm - mass * com_skew @ com_skew
    inertia_wv = mass * com_skew
    inertia_vw = -mass * com_skew
    inertia_vv = mass * jnp.eye(3, dtype=inertia_cm.dtype)
    return inertia_ww, inertia_wv, inertia_vw, inertia_vv


def _transform_motion(rotation, translation, spatial_motion):
    angular = spatial_motion[0:3, 0]
    linear = spatial_motion[3:6, 0]
    angular_out = rotation @ angular
    linear_out = rotation @ (linear - jnp.cross(translation[:, 0], angular))
    return jnp.vstack([angular_out[:, None], linear_out[:, None]])


def _transform_force_transpose(rotation, translation, spatial_force):
    moment = spatial_force[0:3, 0]
    force = spatial_force[3:6, 0]
    force_out = rotation.T @ force
    moment_out = rotation.T @ moment + jnp.cross(translation[:, 0], force_out)
    return jnp.vstack([moment_out[:, None], force_out[:, None]])


def _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, spatial_motion):
    angular = spatial_motion[0:3, 0]
    linear = spatial_motion[3:6, 0]
    momentum_angular = i_ww @ angular + i_wv @ linear
    momentum_linear = i_vw @ angular + i_vv @ linear
    return jnp.vstack([momentum_angular[:, None], momentum_linear[:, None]])


def _inertia_times_subspace(i_ww, i_wv, i_vw, i_vv, subspace):
    subspace_angular = subspace[0:3, :]
    subspace_linear = subspace[3:6, :]
    u_angular = i_ww @ subspace_angular + i_wv @ subspace_linear
    u_linear = i_vw @ subspace_angular + i_vv @ subspace_linear
    return jnp.vstack([u_angular, u_linear])


def _subtract_rank_update(i_ww, i_wv, i_vw, i_vv, term_common, u_term):
    term_angular = term_common[0:3, :]
    term_linear = term_common[3:6, :]
    u_angular = u_term[0:3, :]
    u_linear = u_term[3:6, :]
    return (
        i_ww - term_angular @ u_angular.T,
        i_wv - term_angular @ u_linear.T,
        i_vw - term_linear @ u_angular.T,
        i_vv - term_linear @ u_linear.T,
    )


def _transform_inertia_blocks(rotation, translation, i_ww, i_wv, i_vw, i_vv):
    translation_skew = _skew3(translation[:, 0])
    lower_left = -rotation @ translation_skew
    lower_left_t = translation_skew @ rotation.T

    tmp_ww = i_ww @ rotation + i_wv @ lower_left
    tmp_wv = i_wv @ rotation
    tmp_vw = i_vw @ rotation + i_vv @ lower_left
    tmp_vv = i_vv @ rotation

    return (
        rotation.T @ tmp_ww + lower_left_t @ tmp_vw,
        rotation.T @ tmp_wv + lower_left_t @ tmp_vv,
        rotation.T @ tmp_vw,
        rotation.T @ tmp_vv,
    )


def _spatial_cross_motion(lhs, rhs):
    lhs_angular = lhs[0:3, 0]
    lhs_linear = lhs[3:6, 0]
    rhs_angular = rhs[0:3, 0]
    rhs_linear = rhs[3:6, 0]
    return jnp.vstack(
        [
            jnp.cross(lhs_angular, rhs_angular)[:, None],
            (jnp.cross(lhs_angular, rhs_linear) + jnp.cross(lhs_linear, rhs_angular))[:, None],
        ]
    )


def _spatial_cross_force(spatial_motion, spatial_force):
    angular = spatial_motion[0:3, 0]
    linear = spatial_motion[3:6, 0]
    moment = spatial_force[0:3, 0]
    force = spatial_force[3:6, 0]
    return jnp.vstack(
        [
            (jnp.cross(angular, moment) + jnp.cross(linear, force))[:, None],
            jnp.cross(angular, force)[:, None],
        ]
    )


def _invert_2x2(matrix, dtype):
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[1, 0]
    d = matrix[1, 1]
    determinant = a * d - b * c
    return (1.0 / determinant) * jnp.array([[d, -b], [-c, a]], dtype=dtype)


def _invert_3x3(matrix, dtype):
    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[1, 0]
    e = matrix[1, 1]
    f = matrix[1, 2]
    g = matrix[2, 0]
    h = matrix[2, 1]
    i = matrix[2, 2]

    cofactor00 = e * i - f * h
    cofactor01 = -(d * i - f * g)
    cofactor02 = d * h - e * g
    cofactor10 = -(b * i - c * h)
    cofactor11 = a * i - c * g
    cofactor12 = -(a * h - b * g)
    cofactor20 = b * f - c * e
    cofactor21 = -(a * f - c * d)
    cofactor22 = a * e - b * d

    determinant = a * cofactor00 + b * cofactor01 + c * cofactor02
    adjugate = jnp.array(
        [
            [cofactor00, cofactor10, cofactor20],
            [cofactor01, cofactor11, cofactor21],
            [cofactor02, cofactor12, cofactor22],
        ],
        dtype=dtype,
    )
    return (1.0 / determinant) * adjugate


def _solve_articulated_system(d_matrix, rhs, dof, regularization, dtype):
    if dof == 0:
        return jnp.zeros((0, 0), dtype=dtype), jnp.zeros((0, 1), dtype=dtype)

    identity = jnp.eye(dof, dtype=dtype)
    regularized = d_matrix + regularization * identity
    if dof == 1:
        d_inv = jnp.array([[1.0 / regularized[0, 0]]], dtype=dtype)
    elif dof == 2:
        d_inv = _invert_2x2(regularized, dtype)
    elif dof == 3:
        d_inv = _invert_3x3(regularized, dtype)
    else:
        d_inv = jnp.linalg.solve(regularized, identity)
    return d_inv, rhs


def _reduce_articulated_body_0dof(i_ww, i_wv, i_vw, i_vv, articulated_bias, zeta, dtype):
    articulated_bias_reduced = articulated_bias + _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, zeta)
    return {
        "rhs": jnp.zeros((0, 1), dtype=dtype),
        "d_inv": jnp.zeros((0, 0), dtype=dtype),
        "u_term": jnp.zeros((6, 0), dtype=dtype),
        "articulated_inertia": (i_ww, i_wv, i_vw, i_vv),
        "articulated_bias": articulated_bias_reduced,
    }


def _reduce_articulated_body_1dof(i_ww, i_wv, i_vw, i_vv, subspace, tau_body, articulated_bias, zeta, regularization, dtype):
    u_term = _inertia_times_subspace(i_ww, i_wv, i_vw, i_vv, subspace)
    d_scalar = jnp.dot(subspace[:, 0], u_term[:, 0]) + regularization
    rhs_scalar = tau_body[0, 0] - jnp.dot(subspace[:, 0], articulated_bias[:, 0])
    d_inv = jnp.array([[1.0 / d_scalar]], dtype=dtype)
    rhs = jnp.array([[rhs_scalar]], dtype=dtype)
    zeta_projection = jnp.dot(u_term[:, 0], zeta[:, 0])
    correction_scalar = (rhs_scalar - zeta_projection) / d_scalar
    rank_update = u_term * d_inv[0, 0]
    articulated_inertia_reduced = _subtract_rank_update(i_ww, i_wv, i_vw, i_vv, rank_update, u_term)
    articulated_bias_reduced = (
        articulated_bias
        + _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, zeta)
        + u_term * correction_scalar
    )
    return {
        "rhs": rhs,
        "d_inv": d_inv,
        "u_term": u_term,
        "articulated_inertia": articulated_inertia_reduced,
        "articulated_bias": articulated_bias_reduced,
    }


def _reduce_articulated_body_2dof(i_ww, i_wv, i_vw, i_vv, subspace, tau_body, articulated_bias, zeta, regularization, dtype):
    u_term = _inertia_times_subspace(i_ww, i_wv, i_vw, i_vv, subspace)
    d_matrix = subspace.T @ u_term + regularization * jnp.eye(2, dtype=dtype)
    rhs = tau_body - subspace.T @ articulated_bias
    d_inv = _invert_2x2(d_matrix, dtype)
    rank_update = u_term @ d_inv
    articulated_inertia_reduced = _subtract_rank_update(i_ww, i_wv, i_vw, i_vv, rank_update, u_term)
    articulated_bias_reduced = (
        articulated_bias
        + _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, zeta)
        + u_term @ (d_inv @ (rhs - u_term.T @ zeta))
    )
    return {
        "rhs": rhs,
        "d_inv": d_inv,
        "u_term": u_term,
        "articulated_inertia": articulated_inertia_reduced,
        "articulated_bias": articulated_bias_reduced,
    }


def _reduce_articulated_body_3dof(i_ww, i_wv, i_vw, i_vv, subspace, tau_body, articulated_bias, zeta, regularization, dtype):
    u_term = _inertia_times_subspace(i_ww, i_wv, i_vw, i_vv, subspace)
    d_matrix = subspace.T @ u_term + regularization * jnp.eye(3, dtype=dtype)
    rhs = tau_body - subspace.T @ articulated_bias
    d_inv = _invert_3x3(d_matrix, dtype)
    rank_update = u_term @ d_inv
    articulated_inertia_reduced = _subtract_rank_update(i_ww, i_wv, i_vw, i_vv, rank_update, u_term)
    articulated_bias_reduced = (
        articulated_bias
        + _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, zeta)
        + u_term @ (d_inv @ (rhs - u_term.T @ zeta))
    )
    return {
        "rhs": rhs,
        "d_inv": d_inv,
        "u_term": u_term,
        "articulated_inertia": articulated_inertia_reduced,
        "articulated_bias": articulated_bias_reduced,
    }


def _reduce_articulated_body_generic(i_ww, i_wv, i_vw, i_vv, subspace, tau_body, articulated_bias, zeta, dof, regularization, dtype):
    u_term = _inertia_times_subspace(i_ww, i_wv, i_vw, i_vv, subspace)
    d_matrix = subspace.T @ u_term
    rhs = tau_body - subspace.T @ articulated_bias
    d_inv, rhs = _solve_articulated_system(d_matrix, rhs, dof, regularization, dtype)
    rank_update = u_term @ d_inv
    articulated_inertia_reduced = _subtract_rank_update(i_ww, i_wv, i_vw, i_vv, rank_update, u_term)
    articulated_bias_reduced = (
        articulated_bias
        + _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, zeta)
        + u_term @ (d_inv @ (rhs - u_term.T @ zeta))
    )
    return {
        "rhs": rhs,
        "d_inv": d_inv,
        "u_term": u_term,
        "articulated_inertia": articulated_inertia_reduced,
        "articulated_bias": articulated_bias_reduced,
    }


def get_aba_jax_model(model, *, regularization=DEFAULT_ABA_REGULARIZATION, gravity_sign=-1.0):
    """Return a pure-JAX joint-acceleration kernel for the model."""
    if jax is None:
        raise ImportError("JAX is required for get_aba_jax_model")
    if regularization < 0.0:
        raise ValueError("regularization must be non-negative")

    runtime_model = _build_runtime_model(model)
    body_specs = runtime_model["body_specs"]
    g_indices = jnp.asarray(runtime_model["g_indices"], dtype=jnp.int32)
    n_qd = runtime_model["n_qd"]
    n_tau = runtime_model["n_tau"]
    n_ext = runtime_model["n_ext"]

    @jit
    def run_aba(q, qd, tau_ctrl, ext_forces_concatenated, constants):
        float_dtype = q.dtype
        zero_vec3 = jnp.zeros((3,), dtype=float_dtype)
        zero_spatial = jnp.zeros((6, 1), dtype=float_dtype)
        identity3 = jnp.eye(3, dtype=float_dtype)

        q_pad = jnp.concatenate([q, jnp.zeros((1,), dtype=float_dtype)])
        qd_pad = jnp.concatenate([qd, jnp.zeros((1,), dtype=float_dtype)])
        tau_pad = jnp.concatenate([tau_ctrl[:n_tau], jnp.zeros((1,), dtype=float_dtype)])
        ext_pad = jnp.concatenate([ext_forces_concatenated[:n_ext], jnp.zeros((1,), dtype=float_dtype)])

        v_list = []
        c_list = []
        p_list = []
        rotation_list = []
        translation_list = []
        subspace_list = []
        tau_list = []
        abs_rotation_list = []
        i_ww_list = []
        i_wv_list = []
        i_vw_list = []
        i_vv_list = []

        for body_index, body_spec in enumerate(body_specs):
            dof = body_spec["dof"]
            q_body = q_pad[jnp.asarray(body_spec["q_indices"], dtype=jnp.int32)]
            qd_body = qd_pad[jnp.asarray(body_spec["qd_indices"], dtype=jnp.int32)]
            tau_lookup_indices = np.where(body_spec["tau_indices"] >= 0, body_spec["tau_indices"], n_tau)
            tau_body = tau_pad[jnp.asarray(tau_lookup_indices, dtype=jnp.int32)][:, None]
            mass = constants[body_spec["mass_index"]]
            inertia_cm = _inertia_matrix_from_vector(constants[jnp.asarray(body_spec["inertia_indices"], dtype=jnp.int32)])
            com = constants[jnp.asarray(body_spec["com_indices"], dtype=jnp.int32)]
            translation_parent = constants[jnp.asarray(body_spec["offset_indices"], dtype=jnp.int32)]
            axes = jnp.asarray(body_spec["joint_axes"], dtype=float_dtype)
            joint_type_row = body_spec["joint_types"]

            local_rotation = identity3
            hinge_rotations = []
            for joint_index in range(dof):
                joint_type = joint_type_row[joint_index]
                axis = axes[joint_index]
                if joint_type == SLIDE_JOINT:
                    translation_parent = translation_parent + axis * q_body[joint_index]
                    hinge_rotations.append(identity3)
                elif joint_type == HINGE_JOINT:
                    passive_rotation = _passive_rotation(axis, q_body[joint_index], float_dtype)
                    local_rotation = passive_rotation @ local_rotation
                    hinge_rotations.append(passive_rotation)
                else:
                    hinge_rotations.append(identity3)

            hinge_axes_in_body = [zero_vec3 for _ in range(dof)]
            suffix_rotation = identity3
            for reverse_index in range(dof - 1, -1, -1):
                if joint_type_row[reverse_index] == HINGE_JOINT:
                    hinge_axes_in_body[reverse_index] = suffix_rotation @ axes[reverse_index]
                    suffix_rotation = suffix_rotation @ hinge_rotations[reverse_index]

            subspace_columns = []
            v_joint = zero_spatial
            c_joint = zero_spatial
            v_previous = zero_spatial
            for joint_index in range(dof):
                joint_type = joint_type_row[joint_index]
                if joint_type == HINGE_JOINT:
                    column = jnp.vstack([hinge_axes_in_body[joint_index][:, None], jnp.zeros((3, 1), dtype=float_dtype)])
                elif joint_type == SLIDE_JOINT:
                    axis_body = local_rotation @ axes[joint_index]
                    column = jnp.vstack([jnp.zeros((3, 1), dtype=float_dtype), axis_body[:, None]])
                else:
                    column = jnp.zeros((6, 1), dtype=float_dtype)
                subspace_columns.append(column)
                v_k = column * qd_body[joint_index]
                c_joint = c_joint + _spatial_cross_motion(v_previous, v_k)
                v_joint = v_joint + v_k
                v_previous = v_previous + v_k

            subspace = jnp.concatenate(subspace_columns, axis=1) if dof > 0 else jnp.zeros((6, 0), dtype=float_dtype)

            parent_index = body_spec["parent"]
            if parent_index == -1:
                v_parent = zero_spatial
                absolute_rotation = local_rotation
            else:
                v_parent = v_list[parent_index]
                absolute_rotation = local_rotation @ abs_rotation_list[parent_index]

            translation_col = translation_parent[:, None]
            v_body = _transform_motion(local_rotation, translation_col, v_parent) + v_joint
            c_body = c_joint + _spatial_cross_motion(v_body, v_joint)

            if body_spec["ext_force_indices"] is None:
                ext_force_ground = zero_vec3
            else:
                ext_force_ground = ext_pad[jnp.asarray(body_spec["ext_force_indices"], dtype=jnp.int32)]

            if body_spec["ext_torque_indices"] is None:
                ext_torque_ground = zero_vec3
            else:
                ext_torque_ground = ext_pad[jnp.asarray(body_spec["ext_torque_indices"], dtype=jnp.int32)]

            ext_force_body = absolute_rotation @ ext_force_ground
            ext_torque_body = absolute_rotation @ ext_torque_ground
            ext_spatial = jnp.vstack([ext_torque_body[:, None], ext_force_body[:, None]])

            i_ww, i_wv, i_vw, i_vv = _body_inertia_blocks(mass, com, inertia_cm)
            momentum = _apply_spatial_inertia(i_ww, i_wv, i_vw, i_vv, v_body)
            bias_force = _spatial_cross_force(v_body, momentum) - ext_spatial

            v_list.append(v_body)
            c_list.append(c_body)
            p_list.append(bias_force)
            rotation_list.append(local_rotation)
            translation_list.append(translation_col)
            subspace_list.append(subspace)
            tau_list.append(tau_body)
            abs_rotation_list.append(absolute_rotation)
            i_ww_list.append(i_ww)
            i_wv_list.append(i_wv)
            i_vw_list.append(i_vw)
            i_vv_list.append(i_vv)

        u_list = [None] * len(body_specs)
        d_inv_list = [None] * len(body_specs)
        u_term_list = [None] * len(body_specs)

        for reverse_index in range(len(body_specs) - 1, -1, -1):
            body_spec = body_specs[reverse_index]
            subspace = subspace_list[reverse_index]
            tau_body = tau_list[reverse_index]
            articulated_bias = p_list[reverse_index]
            zeta = c_list[reverse_index]

            i_ww = i_ww_list[reverse_index]
            i_wv = i_wv_list[reverse_index]
            i_vw = i_vw_list[reverse_index]
            i_vv = i_vv_list[reverse_index]

            dof = body_spec["dof"]
            if dof == 0:
                reduced = _reduce_articulated_body_0dof(i_ww, i_wv, i_vw, i_vv, articulated_bias, zeta, float_dtype)
            elif dof == 1:
                reduced = _reduce_articulated_body_1dof(
                    i_ww,
                    i_wv,
                    i_vw,
                    i_vv,
                    subspace,
                    tau_body,
                    articulated_bias,
                    zeta,
                    regularization,
                    float_dtype,
                )
            elif dof == 2:
                reduced = _reduce_articulated_body_2dof(
                    i_ww,
                    i_wv,
                    i_vw,
                    i_vv,
                    subspace,
                    tau_body,
                    articulated_bias,
                    zeta,
                    regularization,
                    float_dtype,
                )
            elif dof == 3:
                reduced = _reduce_articulated_body_3dof(
                    i_ww,
                    i_wv,
                    i_vw,
                    i_vv,
                    subspace,
                    tau_body,
                    articulated_bias,
                    zeta,
                    regularization,
                    float_dtype,
                )
            else:
                reduced = _reduce_articulated_body_generic(
                    i_ww,
                    i_wv,
                    i_vw,
                    i_vv,
                    subspace,
                    tau_body,
                    articulated_bias,
                    zeta,
                    dof,
                    regularization,
                    float_dtype,
                )

            rhs = reduced["rhs"]
            d_inv = reduced["d_inv"]
            u_term = reduced["u_term"]
            articulated_inertia_reduced = reduced["articulated_inertia"]
            articulated_bias_reduced = reduced["articulated_bias"]

            p_list[reverse_index] = articulated_bias_reduced
            u_list[reverse_index] = rhs
            d_inv_list[reverse_index] = d_inv
            u_term_list[reverse_index] = u_term
            i_ww_list[reverse_index] = articulated_inertia_reduced[0]
            i_wv_list[reverse_index] = articulated_inertia_reduced[1]
            i_vw_list[reverse_index] = articulated_inertia_reduced[2]
            i_vv_list[reverse_index] = articulated_inertia_reduced[3]

            parent_index = body_spec["parent"]
            if parent_index != -1:
                transformed_inertia = _transform_inertia_blocks(
                    rotation_list[reverse_index],
                    translation_list[reverse_index],
                    *articulated_inertia_reduced,
                )
                transformed_bias = _transform_force_transpose(
                    rotation_list[reverse_index],
                    translation_list[reverse_index],
                    articulated_bias_reduced,
                )
                p_list[parent_index] = p_list[parent_index] + transformed_bias
                i_ww_list[parent_index] = i_ww_list[parent_index] + transformed_inertia[0]
                i_wv_list[parent_index] = i_wv_list[parent_index] + transformed_inertia[1]
                i_vw_list[parent_index] = i_vw_list[parent_index] + transformed_inertia[2]
                i_vv_list[parent_index] = i_vv_list[parent_index] + transformed_inertia[3]

        a_ground = jnp.vstack([jnp.zeros((3, 1), dtype=float_dtype), gravity_sign * constants[g_indices][:, None]])
        accelerations = jnp.zeros((n_qd,), dtype=float_dtype)
        a_list = []
        for body_index, body_spec in enumerate(body_specs):
            parent_index = body_spec["parent"]
            parent_acceleration = a_ground if parent_index == -1 else a_list[parent_index]
            a_body = _transform_motion(rotation_list[body_index], translation_list[body_index], parent_acceleration) + c_list[body_index]
            qdd = d_inv_list[body_index] @ (u_list[body_index] - u_term_list[body_index].T @ a_body)
            a_body = a_body + subspace_list[body_index] @ qdd
            a_list.append(a_body)
            for joint_index, qd_index in enumerate(body_spec["qd_indices"]):
                accelerations = accelerations.at[qd_index].set(qdd[joint_index, 0])

        return accelerations

    return run_aba


def get_aba_biosym(model, *, regularization=DEFAULT_ABA_REGULARIZATION, gravity_sign=-1.0):
    """Return a biosym-compatible ABA function."""
    aba_fn = get_aba_jax_model(model, regularization=regularization, gravity_sign=gravity_sign)

    def aba_biosym(states, constants, model):
        q = states.model[model.coordinates["idx"] : model.coordinates["idx"] + model.coordinates["n"]]
        qd = states.model[model.speeds["idx"] : model.speeds["idx"] + model.speeds["n"]]
        tau = states.model[model.forces["idx"] : model.forces["idx"] + model.forces["n"]]
        ext_forces = states.model[
            model.ext_forces["idx"] : model.ext_forces["idx"] + model.ext_forces["n"] + model.ext_torques["n"]
        ]
        return aba_fn(q, qd, tau, ext_forces, constants.model)

    return partial(aba_biosym, model=model)


def qddot_step_biosym(model, *, regularization=DEFAULT_ABA_REGULARIZATION, gravity_sign=-1.0):
    """Return a biosym-compatible joint-acceleration function."""
    aba_fn = get_aba_jax_model(model, regularization=regularization, gravity_sign=gravity_sign)

    def qddot_biosym(states, constants):
        q = states.model[model.coordinates["idx"] : model.coordinates["idx"] + model.coordinates["n"]]
        qd = states.model[model.speeds["idx"] : model.speeds["idx"] + model.speeds["n"]]

        if "actuator_model" in model.run:
            tau = model.run["actuator_model"](states, constants).flatten()
        else:
            tau = jnp.zeros((model.forces["n"],), dtype=q.dtype)

        if "gc_model" in model.run:
            ext_forces, ext_moments = model.run["gc_model"](states, constants)
            ext_forces_concat = jnp.concatenate([ext_forces, ext_moments], axis=0).flatten()
        else:
            ext_forces_concat = jnp.zeros((model.ext_forces["n"] + model.ext_torques["n"],), dtype=q.dtype)

        return aba_fn(q, qd, tau, ext_forces_concat, constants.model)

    return qddot_biosym


def get_qddot_jacobian_manual(model):
    aba_model_jacobian = jax.jit(jax.jacobian(get_aba_jax_model(model), argnums=(0, 1, 2, 3)))

    def qddot_step_jacobian_manual(states, constants, model):
        q_idx = model.coordinates["idx"]
        q_n = model.coordinates["n"]
        qd_idx = model.speeds["idx"]
        qd_n = model.speeds["n"]

        q = states.model[q_idx : q_idx + q_n]
        qd = states.model[qd_idx : qd_idx + qd_n]

        if "actuator_model" in model.run:
            tau = model.run["actuator_model"](states, constants).flatten()
            n_tau = tau.shape[0]
            d_tau_d_states = model.run["actuator_model_jacobian"](states, constants)
        else:
            tau = jnp.zeros((model.forces["n"],), dtype=q.dtype)
            n_tau = tau.shape[0]
            d_tau_d_states = None

        if "gc_model" in model.run:
            ext_forces, ext_moments = model.run["gc_model"](states, constants)
            ext_forces_concat = jnp.concatenate([ext_forces, ext_moments], axis=0).flatten()
            d_ext_forces_d_states, d_ext_moments_d_states = model.run["gc_model_jacobian"](states, constants)

            n_force_rows = ext_forces.shape[0]
            n_moment_rows = ext_moments.shape[0]

            def combine_and_flatten(jac_force, jac_moment):
                if jac_force is None and jac_moment is None:
                    return None

                if jac_force is not None:
                    n_leaf = jac_force.shape[-1]
                elif jac_moment is not None:
                    n_leaf = jac_moment.shape[-1]
                else:
                    return None

                safe_force = jac_force if jac_force is not None else jnp.zeros((n_force_rows, 3, n_leaf))
                safe_moment = jac_moment if jac_moment is not None else jnp.zeros((n_moment_rows, 3, n_leaf))
                combined = jnp.concatenate([safe_force, safe_moment], axis=0)
                return combined.reshape(((n_force_rows + n_moment_rows) * 3, n_leaf))

            if d_ext_forces_d_states is None and d_ext_moments_d_states is None:
                d_ext_d_states = None
            elif d_ext_forces_d_states is None:
                d_ext_d_states = jax.tree_util.tree_map(
                    lambda jac_moment: combine_and_flatten(None, jac_moment),
                    d_ext_moments_d_states,
                )
            elif d_ext_moments_d_states is None:
                d_ext_d_states = jax.tree_util.tree_map(
                    lambda jac_force: combine_and_flatten(jac_force, None),
                    d_ext_forces_d_states,
                )
            else:
                d_ext_d_states = jax.tree_util.tree_map(
                    combine_and_flatten,
                    d_ext_forces_d_states,
                    d_ext_moments_d_states,
                )
        else:
            ext_forces_concat = jnp.zeros((model.ext_forces["n"] + model.ext_torques["n"],), dtype=q.dtype)
            d_ext_d_states = None

        d_aba_dq, d_aba_dqd, d_aba_dtau, d_aba_dext = aba_model_jacobian(q, qd, tau, ext_forces_concat, constants.model)

        n_dof = q_n

        def make_zero_jacobian(leaf):
            if leaf is None:
                return None
            return jnp.zeros((n_dof,) + leaf.shape)

        total_jacobian = jax.tree_util.tree_map(make_zero_jacobian, states)

        if total_jacobian.model is not None:
            jac_model = total_jacobian.model
            jac_model = jac_model.at[:, q_idx : q_idx + q_n].add(d_aba_dq)
            jac_model = jac_model.at[:, qd_idx : qd_idx + qd_n].add(d_aba_dqd)
            total_jacobian = total_jacobian.replace(model=jac_model)

        if d_tau_d_states is not None:
            def add_tau(current_jacobian, tau_jacobian):
                if current_jacobian is None or tau_jacobian is None:
                    return current_jacobian
                n_leaf = tau_jacobian.shape[-1]
                tau_flat = tau_jacobian.reshape((n_tau, n_leaf))
                return current_jacobian + d_aba_dtau @ tau_flat

            total_jacobian = jax.tree_util.tree_map(add_tau, total_jacobian, d_tau_d_states)

        if d_ext_d_states is not None:
            def add_external(current_jacobian, external_jacobian):
                if current_jacobian is None or external_jacobian is None:
                    return current_jacobian
                return current_jacobian + d_aba_dext @ external_jacobian

            total_jacobian = jax.tree_util.tree_map(add_external, total_jacobian, d_ext_d_states)

        return total_jacobian

    return partial(qddot_step_jacobian_manual, model=model)
