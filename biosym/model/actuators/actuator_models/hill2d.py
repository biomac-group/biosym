import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from biosym.model.actuators.base_actuator import BaseActuator

class Hill2d(BaseActuator):
    """
    A reimplementation of the 2D Hill muscle model as in gait2d.
    This model is purpose-built for 2D models with rotational joints.
    It is also purpose-built for direct collocation, and needs adjustments to work in a forward simulation setting,
    and might not perform well in DL scenarios.

    Why to not optimize for e as in BioMacSimToolbox:
    Activating e every 4th node (dt=0.01): c = 0.25 -> a = [0,1,0.66,0.33] - average: 0.5
    Activating e every 2nd node (dt=0.01): c = 0.5 -> a = [0.66,1,0.66,1] - average: 0.83
    Activating e continuously (dt=0.01), e = 0.5 -> c = 0.25 -> a = 0.5 - average: 0.5
    Activating e continuously (dt=0.01), e = 0.707 -> c = 0.5 -> a = 0.707 - average: 0.707 --> Lower than when jittering e
    Even worse:
    Activating e every 2nd node (dt=0.02): c = 0.5 -> a = [0, *2, 1.33, *0.67, 0.44,*1.67, 0.89, *1.33, 0.88] - average: 1.02
    Activating e strategically 1 (dt=0.02): c = 0.375 -> a = [0, *2, 1.33, 0.87, 0.58, *1.62, 1.07, 0.71, *1.29] - average: 1.05 --> This must be super bad for IPOPT

    So i think all we need to account for activation / deactivation dynamics is that the \\dot{a} is limited by [1/t_act, 1/t_deact]

    How to optimize for e then?

    Do not: https://www.biorxiv.org/content/10.1101/2025.01.30.635759v1.full.pdf
    But if you really want to: a[t+1] = e[t] + (a[t] - e[t]) * np.exp(-(e[t]/Tact+(1-e[t])/Tdeact)*t)

    Recommendation: Do not optimize for e at all, do not allow a>1, and limit \\dot{a} as stated here:
    a[t+1,max] = 1 + ( a[t] - 1 ) * exp(-dt/Tact) # Exponential decay to 1
    a[t+1,min] = (a[t]) * exp(-dt/Tdeact) # Exponential decay to 0

    So the constraint would be linear violation of this term
    """

    def __init__(self, joints_dict, muscles_dict, defaults):
        super().__init__(joints_dict)
        self.muscles_dict = muscles_dict

        # Grab the first muscle from the defaults if available
        if defaults is not None:
            defaults = defaults.findall("muscle")[0].attrib

        self.n_actuators = len(muscles_dict)
        self.actuators = {}

        self.names = [mi.get("name") for mi in muscles_dict]

        self.muscle_constants = {}
        for const in [
            "fmax",
            "lceopt",
            "width",
            "vmax",
            "umax",
            "Arel",
            "gmax",
            "kPEE",
            "PEEslack",
            "SEEslack",
            "Tact",
            "Tdeact",
            "L0"
        ]:
            self.muscle_constants[const] = jnp.array(
                [float(mi.get(const, defaults.get(const, 0.0))) for mi in muscles_dict]
            )[:, jnp.newaxis]
        # As defined in gait2d.c
        self.muscle_constants["kSEE"] = 1.0/((self.muscle_constants["umax"]**2)*(self.muscle_constants["SEEslack"]**2))

        self.moment_arm_matrix = jnp.zeros((self.n_actuators, len(joints_dict)))
        self.actuated_joints = set()
        for muscle, idx in enumerate(muscles_dict):
            for dof in idx.findall("dof"):
                joint_name = dof.get("name")
                joint_idx = joints_dict.index(joint_name)
                moment_arm = float(dof.get("momentarm"))
                self.moment_arm_matrix = self.moment_arm_matrix.at[
                    muscle, joint_idx
                ].set(moment_arm)
                self.actuated_joints.add(joint_name)

        self.joints = jnp.array(
            [float(mi.get("joint", defaults.get("joint", 0.0))) for mi in muscles_dict]
        )  # Joint angles
        self.state_vector = [f"Lce_{n}" for n in self.names] + [f"Lce_{n}_dot" for n in self.names] + [f"a_{n}" for n in self.names]
        self.idx = {
            "Lce": jnp.arange(0, self.n_actuators),
            "Lce_dot": jnp.arange(self.n_actuators, 2*self.n_actuators),
            "a": jnp.arange(2*self.n_actuators, 3*self.n_actuators),
        }
        self.bounds = {
            "states": {
                "min": jnp.concatenate(
                    (
                        1e-3
                        * jnp.ones(self.n_actuators),  # Lce, avoid dividing by zero
                        -self.muscle_constants["vmax"].squeeze(),  # Lce_dot
                        0 * jnp.ones(self.n_actuators),  # a
                    )
                ),
                "max": jnp.concatenate(
                    (
                        3 * jnp.ones(self.n_actuators),  # id
                        self.muscle_constants["vmax"].squeeze(),  # Lce_dot
                        1 * jnp.ones(self.n_actuators),  # a
                    )
                ),
            }
        }

    def get_actuators(self):
        return self.muscles_dict

    def get_n_actuators(self):
        """
        Returns the number of actuators in the model.
        """
        return self.n_actuators

    def reset(self):
        """
        Resets the actuator behaviour.
        """

    def get_actuated_joints(self):
        """
        Returns the list of actuated joints.
        """
        return self.actuated_joints

    def get_n_states(self):
        return len(self.state_vector)

    def get_n_constants(self):
        return 0

    def get_n_constraints(self, model, settings):
        return (
            self.n_actuators * (settings.get("nnodes_dur")-1) + self.n_actuators * settings.get("nnodes")
        )  # Each actuator has 2 constraints (activation dynamics and force equilibrium)
    
    def get_n_constraints_per_node(self):
        return self.n_actuators * 2

    def get_nnz(self, model, settings):
        nnodes = settings.get("nnodes")
        nnodes_dur = settings.get("nnodes_dur")
        # First constraint depends on model + actuator states, 2nd on globals and actuator states
        forces = nnodes * self.get_n_constraints_per_node()//2 * (self.get_n_states() + len(model.state_vector))
        activation = self.get_n_constraints_per_node()//2 * ((nnodes_dur-1) * 2 + 2 * (nnodes_dur-1))  # 2 globals: dur, speed
        return forces + activation

    def process_eom(self, model):
        """
        Build the muscle attachment and path computation functions using symbolic variables.
        This creates fast compiled functions for muscle visualization with proper reference frames.
        """
        from sympy import Matrix, lambdify
        from sympy.physics.mechanics import Point
        
        # Call parent process_eom
        super().process_eom(model)
        
        # Create symbolic functions for muscle attachment points and paths
        # Use dynamic body positions that change with joint angles (like FK system)
        attachment_points = []
        
        joint_names = [joint["name"] for joint in model.dicts["joints"]]
        body_names = [body["name"] for body in model.dicts["bodies"]]
        
        # Use FK_vis to get actual joint positions for muscle attachment calculations
        # We need to use the default states/constants to get representative joint positions
        # FK_vis returns (8,3) array with positions: [pelvis, hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l, extra]
        
        
        # Legacy joint_positions dict for compatibility with existing waypoint/insertion code
        joint_positions = {}
        for body_name in body_names:
            if body_name in model.body_origins:
                origin = model.body_origins[body_name]
                pos_vec = origin.pos_from(model.origin)
                joint_positions[body_name] = Matrix([
                    pos_vec.dot(model.ground_frame.x),
                    pos_vec.dot(model.ground_frame.y), 
                    pos_vec.dot(model.ground_frame.z)
                ])
            else:
                joint_positions[body_name] = Matrix([0, 0, 0])
        
        # Dynamic muscle attachment calculation using symbolic joint positions
        # This uses the actual joint hierarchy to calculate muscle attachment points
        def calculate_muscle_attachments(muscle_name, actuated_joint_names):
            """Calculate anatomically correct muscle attachment points using dynamic joint positions"""
            
            def get_joint_body_by_name(joint_name):
                """Get the child body of a joint by name"""
                joint_idx = joint_names.index(joint_name)
                return model.dicts["joints"][joint_idx]['child']
            
            def get_parent_body_by_joint_name(joint_name):
                """Get the parent body of a joint by name"""
                joint_idx = joint_names.index(joint_name)
                return model.dicts["joints"][joint_idx]['parent']
            
            def get_child_of_body(body_name):
                """Get the child body of a given body"""
                parents_of_joints = [j['parent'] for j in model.dicts['joints']]
                try:
                    child_joint_idx = parents_of_joints.index(body_name)
                    return model.dicts['joints'][child_joint_idx]['child']
                except ValueError:
                    return None  # No child found (terminal body)
            
            def get_symbolic_body_position(body_name):
                """Get symbolic position of a body origin"""
                if body_name in model.body_origins:
                    origin = model.body_origins[body_name]
                    pos_vec = origin.pos_from(model.origin)
                    return Matrix([
                        pos_vec.dot(model.ground_frame.x),
                        pos_vec.dot(model.ground_frame.y), 
                        pos_vec.dot(model.ground_frame.z)
                    ])
                else:
                    return Matrix([0, 0, 0])
            
            # Generic muscle attachment calculation based on actuated joints
            if len(actuated_joint_names) == 1:
                # Single-joint muscle: origin is 66% down the parent segment of the actuated joint
                joint_name = actuated_joint_names[0]
                
                # Get the parent segment (where muscle originates)
                parent_body = get_parent_body_by_joint_name(joint_name)
                # Get the child segment (where muscle crosses)
                child_body = get_joint_body_by_name(joint_name)
                
                # For single-joint muscles, the origin is 66% down the parent segment
                # Parent segment runs from the parent's parent to the actuated joint
                
                # Get parent of parent (proximal end of parent segment)
                parent_parent_body = None
                for joint in model.dicts["joints"]:
                    if joint['child'] == parent_body:
                        parent_parent_body = joint['parent']
                        break
                
                if parent_parent_body:
                    # Origin: 66% down the parent segment (from parent's parent to actuated joint)
                    proximal_pos = get_symbolic_body_position(parent_body)
                    actuated_joint_pos = get_symbolic_body_position(child_body)  # This is the actuated joint position
                    origin = proximal_pos + 0.66 * (actuated_joint_pos - proximal_pos)
                else:
                    # Fallback: use the parent body position (no parent's parent found)
                    origin = get_symbolic_body_position(parent_body)
                
                return origin
            
        
        for i, muscle_name in enumerate(self.names):
            # Find which joints this muscle actuates by checking moment arm matrix
            actuated_joints = []
            for j in range(len(joint_names)):
                if abs(self.moment_arm_matrix[i, j]) > 1e-6:
                    actuated_joints.append(j)
            
            if len(actuated_joints) == 0:
                raise ValueError(f"Muscle {muscle_name} has no actuated joints")
            else: 
                # Single-joint muscle - origin is on the parent segment of the actuated joint
                joint_idx = actuated_joints[0]
                joint_info = model.dicts["joints"][joint_idx]
                
                # For single-joint muscle, the muscle spans the parent body segment
                # Origin body is the parent of the actuated joint
                origin_body = joint_info['parent']  
                insertion_joint_pos = joint_positions[joint_info['child']]
                

                complete_path = []
                
                # Calculate anatomically correct origin using FK positions
                actuated_joint_names = [joint_info['name']]
                complete_path.append(calculate_muscle_attachments(muscle_name, actuated_joint_names))
                for j_ in actuated_joints:
                    joint_info = model.dicts["joints"][j_]
                    actuated_joint_names = [joint_info['name']]


                    # Waypoint: at joint (midpoint) with moment arm offset in insertion body reference frame
                    moment_arm = float(self.moment_arm_matrix[i,j_])
                    insertion_frame = model.reference_frames[joint_info['child']]
                    insertion_origin = model.body_origins[joint_info['child']]

                    # Create waypoint in insertion body reference frame
                    waypoint_sym = Point(f"muscle_{i}_waypoint")
                    waypoint_sym.set_pos(
                        insertion_origin,
                        insertion_frame.x * moment_arm + insertion_frame.y * 0 + insertion_frame.z * 0
                    )

                    # Convert to global position vector
                    waypoint_pos = waypoint_sym.pos_from(model.origin)
                    waypoint = Matrix([
                        waypoint_pos.dot(model.ground_frame.x),
                        waypoint_pos.dot(model.ground_frame.y),
                        waypoint_pos.dot(model.ground_frame.z)
                    ])
                    complete_path.append(waypoint)
                
                # Find insertion body's child to get the correct segment length

                insertion_child_bodies = []
                for joint in model.dicts["joints"]:
                    if joint['parent'] == joint_info['child']:
                        insertion_child_bodies.append(joint['child'])
                
                # Check if insertion body has multiple child joints at different positions
                if len(insertion_child_bodies) > 1:
                    # Check if all child joints are at the same position
                    child_positions = []
                    for child_body in insertion_child_bodies:
                        if child_body in joint_positions:
                            child_positions.append(tuple(joint_positions[child_body]))
                    
                    if len(set(child_positions)) > 1:
                        raise ValueError(f"Body {insertion_child_body} has multiple child joints at different positions: {insertion_child_bodies}")
                
                insertion_child_body = insertion_child_bodies[0] if insertion_child_bodies else None
                insertion_joint_pos = joint_positions[joint_info['child']]
                # Insertion: create attachment point 66% down the insertion body segment or use moment arm
                if insertion_child_body and insertion_child_body in body_names:
                    insertion_child_joint_pos = joint_positions[insertion_child_body]
                    # 66% from insertion joint toward insertion child joint (66% down insertion segment)
                    insert_attach = insertion_joint_pos + 0.33 * (insertion_child_joint_pos - insertion_joint_pos)                
                    complete_path.append(insert_attach)
            
            
            attachment_points.append(complete_path)
        
        # Convert to matrix form and create compiled functions
        muscle_geometry = []
        for muscle_path in attachment_points:
            if len(muscle_path) >= 2:
                # Create path matrix: each row is [origin, waypoints..., insertion]
                path_matrix = Matrix([point.T for point in muscle_path])
                muscle_geometry.append(path_matrix)
            else:
                raise ValueError(f"Muscle path has insufficient points: {len(muscle_path)}")
        
        # Compile the muscle geometry functions
        self.muscle_geometry = []
        for i, geom in enumerate(muscle_geometry):
            # Replace dynamic symbols before lambdifying
            geom_replaced = model._replace_dyn(geom)
            compiled_func = lambdify(model._v, geom_replaced, modules="numpy", cse=True)
            self.muscle_geometry.append(compiled_func)
        
        return super().process_eom(model)

    def forward(self, states, constants, model):
        F_ce, F_see, F_pee = self.muscle_equations(states, constants, model)
        #F_see = F_ce + F_pee
        # Todo: Hill's equations in here
        # What is the force at every joint
        return (self.moment_arm_matrix.T @ F_see).T  # shape (n_samples, n_joints)

    def muscle_equations(self, states, constants, model):
        # Constraction dynamics
        F_max = self.muscle_constants["fmax"]
        L_ce_opt = self.muscle_constants["lceopt"]
        W = self.muscle_constants["width"]
        V_max = self.muscle_constants["vmax"]
        A = self.muscle_constants["Arel"]
        G_max = self.muscle_constants["gmax"]
        k_pee = self.muscle_constants["kPEE"]
        pee_slack = self.muscle_constants["PEEslack"]
        k_see = self.muscle_constants["kSEE"]
        see_slack = self.muscle_constants["SEEslack"]
        L0 = self.muscle_constants["L0"]

        if states.model.ndim < 2:
            L_ce = states.actuator_model[self.idx["Lce"]][:, jnp.newaxis]
            L_ce_dot = states.actuator_model[self.idx["Lce_dot"]][:, jnp.newaxis]
            a = states.actuator_model[self.idx["a"]][:, jnp.newaxis]
            q = states.model[
                model.coordinates["idx"] : model.coordinates["idx"]
                + model.coordinates["n"]
            ][:, jnp.newaxis]
        else:
            L_ce = states.actuator_model[:, self.idx["Lce"]].T
            a = states.actuator_model[:, self.idx["a"]].T
            q = states.model[
                :,
                model.coordinates["idx"] : model.coordinates["idx"]
                + model.coordinates["n"],
            ].T
            # L_ce_dot for the last state is always zero?
            L_ce_dot = states.actuator_model[:, self.idx["Lce_dot"]].T


        x = (
            L_ce - 1
        ) / W  # L_ce: Normalized contractile element length, W: Width of the force-length relationship
        # Force-length relationship
        F1 = jnp.exp(-x*x)
        # Force-velocity relationship
        c_3 = V_max * A * (G_max - 1) / (A + 1)
        F2 = jnp.where(
            L_ce_dot < 0,
            (V_max + L_ce_dot) / (V_max - L_ce_dot / A),
            (G_max * L_ce_dot + c_3) / (L_ce_dot + c_3),
        )
        F_damp = 1e-3 * L_ce_dot  # Damping term

        # F_pee
        # stiffness of the linear term is 0.01 Fmax/meter
        # elongation of PEE, in _ce_opt units
        x = L_ce - pee_slack
        F_pee = 0.01 * L_ce_opt * x  # linear term
        F_pee = jnp.where(x > 0, F_pee + k_pee * x**2, F_pee)

        # F_see
        # stiffness of the linear term is 0.01 Fmax/meter
        # Lm = Lm-MA[i]*ang[i]?? Lm is the current muscle length based on moment arm and joint angle
        # Moment arm is constant, therefore dLm/djoint_angle = -moment_arm
        # moment_arm_matrix: (a, b, t), q: (b, t)
        # L_ce_opt: (a, t)
        # We want: Lm: (a, t)
        Lm = L0 - self.moment_arm_matrix @ q

        x = Lm - L_ce * L_ce_opt - see_slack
        F_see = 0.01 * x  # Assuming k1 should be 0.01 * F_max
        F_see = jnp.where(x > 0, F_see + k_see * x**2, F_see)
     
        # F_ce
        F_ce = a * F1 * F2 + F_damp
        return F_max * F_ce, F_max * F_see, F_max * F_pee

    def constraints(self, states, constants, model, settings):
        states, globals = states
        F_ce, F_see, F_pee = self.muscle_equations(states[:settings.get("nnodes")], constants, model)
        F_max = self.muscle_constants["fmax"]
        c1 = (F_see - F_ce - F_pee) / F_max  # Normalized to F_max
        c1 = c1.T.reshape(-1)  # shape (n_actuators,)
        if (states.h is not None) and (states.h.size > 1):
            h = states.h[:-1]
        elif globals is not None: # Is the order of priority correct here?
            h = globals.dur / (settings.get('nnodes_dur') - 1)
        else:
            h = 0.01  # Default time step if not provided
        if globals is not None:
            a = states.actuator_model[
                1:, self.idx["a"]
            ]  # at idx of actuators I guess, all but the last state
            a_max = 1 + (states.actuator_model[:-1, self.idx['a']] - 1) * jnp.exp(
                -h / self.muscle_constants['Tact'].T
            )  # Exponential decay to 1
            a_min = (states.actuator_model[:-1, self.idx['a']]) * jnp.exp(
                -h / self.muscle_constants['Tdeact'].T
            )  # Exponential decay to 0
            c2 = (a-a_max)*0
            c1 = jnp.concatenate((c1, c2.reshape(-1)), axis=0)
        else:
            pass
        return c1


    def jacobian(self, states, constants, model, settings):
        states, globals = states
        ### First constraint: Force equilibrium
        def c1(states, constants):
            F_ce, F_see, F_pee = self.muscle_equations(states, constants, model)
            F_max = self.muscle_constants["fmax"]
            c1 = (F_see - F_ce - F_pee) / F_max  # Normalized to F_max
            c1 = c1.T.reshape(-1)  # shape (n_actuators,)
            return c1 
        c_fun = jax.jit(jax.vmap(jax.jacobian(c1, argnums=0), in_axes=(0, None), out_axes=0))
        jac = c_fun(states[:settings.get("nnodes")], constants)

        ncons = self.get_n_constraints_per_node()
        nnodes = settings.get("nnodes")
        nnodes_dur = settings.get("nnodes_dur")
        node_indices = jnp.arange(nnodes)
        nvpn_model = states[0].model.size
        nvpn_all = states[0].size()
        nact = self.get_n_states()

            
        # Model jacobian blocks, only take the 
        row_blocks_model = node_indices[:, None] * ncons // 2 + jnp.arange(ncons // 2)[None, :]
        col_blocks_model = node_indices[:, None] * nvpn_all + jnp.arange(nvpn_model)[None, :]

        rows_model = jnp.repeat(row_blocks_model, nvpn_model, axis=1)
        cols_model = jnp.tile(col_blocks_model, (1, ncons // 2))
        data_model = jac.model.reshape(nnodes, -1)

        # Actuator jacobian blocks - diagonal structure
        row_blocks_act = node_indices[:, None] * ncons // 2 + jnp.arange(ncons // 2)[None, :]
        col_blocks_act = (node_indices[:, None] + 1) * nvpn_all - nact + jnp.arange(nact)[None, :]

        rows_act = jnp.repeat(row_blocks_act, nact, axis=1)
        cols_act = jnp.tile(col_blocks_act, (1, ncons//2))
        data_act = jac.actuator_model.reshape(nnodes, -1)

        # Concatenate model and actuator parts
        rows_out = jnp.concatenate([rows_model.flatten(), rows_act.flatten()])
        cols_out = jnp.concatenate([cols_model.flatten(), cols_act.flatten()])
        data_out = jnp.concatenate([data_model.flatten(), data_act.flatten()])

        # Second constraint: Activation dynamics
        def c2(current_state, next_states, globals):
            if (current_state.h is not None) and (current_state.h.size > 1):
                h = current_state.h
            elif globals is not None: # Is the order of priority correct here?
                h = globals.dur / (nnodes_dur-1)
            a = next_states.actuator_model[self.idx["a"]][jnp.newaxis,:]  # at idx of actuators I guess, all but the last state
            a_max = 1 + (current_state.actuator_model[self.idx['a']] - 1) * jnp.exp(
                -h / self.muscle_constants['Tact'].T
            )  # Exponential decay to 1
            a_min = (current_state.actuator_model[self.idx['a']]) * jnp.exp(
                -h / self.muscle_constants['Tdeact'].T
            )  # Exponential decay to 0
            c2 = (a)*0
            # Debug: check if constraints are active            
            return c2.T.reshape(-1) # shape (n_actuators*(nnodes-1),)

        # Add the second constraint jacobian

        nnodes = len(states)
        node_indices = jnp.arange(nnodes)
        c2_fun = jax.jit(jax.vmap(jax.jacobian(c2, argnums=(0,1,2)), in_axes=(0,0,None), out_axes=0))

        # row start is at nnodes_dur-1 * ncons//2
        # Sparser solution: only a
        rows_current_state = (settings.get("nnodes")*ncons//2) + (node_indices[:-1, None] + jnp.arange(len(self.idx['a']))[None, :])  # shape (nnodes-1, n_actuators)
        cols_current_state = (node_indices[:-1, None] + 1) * nvpn_all - nact + self.idx['a'][None, :]  # shape (nnodes-1, n_actuators)
        rows_next_state = ((settings.get("nnodes")+1)*ncons//2) + (node_indices[1:, None] + jnp.arange(len(self.idx['a']))[None, :]) # shape (nnodes-1, n_actuators)
        cols_next_state = (node_indices[1:, None] + 1) * nvpn_all - nact + self.idx['a'][None, :]  # shape (nnodes-1, n_actuators)

        if nnodes > 1:
            jac2_first_state, jac2_next_state, jac_globals = c2_fun(states[:-1], states[1:], globals)
            jac_first_state = jac2_first_state.actuator_model[:, jnp.arange(ncons//2), self.idx['a']].reshape(nnodes-1, -1)
            jac_next_state = jac2_next_state.actuator_model[:, jnp.arange(ncons//2), self.idx['a']].reshape(nnodes-1, -1)

            data_out = jnp.concatenate([data_out, jac_first_state.flatten(), jac_next_state.flatten()])
            rows_out = jnp.concatenate([rows_out, rows_current_state.flatten(), rows_next_state.flatten()])
            cols_out = jnp.concatenate([cols_out, cols_current_state.flatten(), cols_next_state.flatten()])

            data_globals = jnp.stack((jac_globals.dur, jac_globals.speed), axis=0).T.reshape(2, -1)
            rows_globals = jnp.repeat(jnp.arange(data_globals.shape[1]), data_globals.shape[0]) + (ncons // 2)*(settings.get("nnodes"))
            cols_globals = jnp.tile(jnp.arange(data_globals.shape[0]), data_globals.shape[1]) + states.size()  # shape (224,)
            
            rows_out = jnp.concatenate([rows_out, rows_globals])
            cols_out = jnp.concatenate([cols_out, cols_globals])
            data_out = jnp.concatenate([data_out, data_globals.flatten()])

        return rows_out, cols_out, data_out
    
    

    def plot(self, states, model, mode, ax, **kwargs):
        """
        Plots the muscles in the model using precomputed EOM muscle geometry.
        Each muscle is drawn following its anatomically correct path with color
        indicating activation level: blue (unused) to red (fully activated).
        
        Parameters
        ----------
        states : object or list
            The state(s) of the model containing muscle activations and positions
        model : object
            The model object containing muscle and body definitions
        mode : str
            The mode of the plot "init" or "update"
        ax : matplotlib.axes.Axes
            The axes object to plot on
        **kwargs : dict
            Additional plotting parameters including:
            - case : str, "2D" or "3D" (default "3D")
            - non_zero_axes : list, required for 2D case
            - frame : int, required for update mode
            - plot_objects : tuple, required for update mode
        """
        import matplotlib.colors as mcolors
        
        if "case" in kwargs:
            case = kwargs["case"]
            if case not in ["2D", "3D"]:
                raise ValueError("Invalid case. Must be '2D' or '3D'.")
            if case == "2D":
                if "non_zero_axes" in kwargs:
                    non_zero_axes = kwargs["non_zero_axes"]
                else:
                    raise ValueError("2D case requires non_zero_axes as an input argument to the muscle model.")
        else:
            case = "3D"

        if mode == "init":
            self.muscle_lines = []
            
            # Get muscle data for all time points using EOM-based geometry
            muscle_paths_all_frames = []
            activations = []
            
            if isinstance(states, list):
                for i in range(len(states)):
                    # Get muscle paths from precomputed EOM geometry
                    _, _ = self._get_muscle_attachment_points(
                        states[i].states, states[i].constants, model
                    )
                    
                    # Store muscle paths for this frame
                    if hasattr(self, 'current_muscle_paths'):
                        muscle_paths_all_frames.append(self.current_muscle_paths.copy())
                    else:
                        muscle_paths_all_frames.append([])
                    
                    # Get muscle activations
                    act = self._get_activations(states[i].states)
                    activations.append(act)
                    
            elif len(states.states.model.shape) == 1:
                _, _ = self._get_muscle_attachment_points(
                    states.states, states.constants, model
                )
                
                if hasattr(self, 'current_muscle_paths'):
                    muscle_paths_all_frames.append(self.current_muscle_paths.copy())
                else:
                    muscle_paths_all_frames.append([])
                
                act = self._get_activations(states.states)
                activations.append(act)
            else:
                for i in range(len(states)):
                    _, _ = self._get_muscle_attachment_points(
                        states[i].states, states[i].constants, model
                    )
                    
                    if hasattr(self, 'current_muscle_paths'):
                        muscle_paths_all_frames.append(self.current_muscle_paths.copy())
                    else:
                        muscle_paths_all_frames.append([])
                    
                    act = self._get_activations(states[i].states)
                    activations.append(act)
            
            self.muscle_paths_all_frames = muscle_paths_all_frames
            self.activations = np.array(activations)
            
            # Plot each muscle using EOM-based geometry
            for i in range(self.n_actuators):
                # Get muscle activation (0 = light grey, 1 = red)
                activation_val = float(self.activations[0][i])
                activation_val = np.clip(activation_val, 0.0, 1.0)  # Ensure 0-1 range
                
                # Linear color interpolation: light grey (0) to red (1)
                # Light grey: [0.8, 0.8, 0.8], Red: [1.0, 0.0, 0.0]
                grey_val = 0.6 * (1 - activation_val)
                muscle_color = mcolors.to_hex([grey_val + activation_val, grey_val, grey_val])
                
                # Line width based on activation (minimum width for visibility)
                line_width = 1.0 + 2.0 * activation_val
                
                # Plot muscle path from EOM geometry
                if (hasattr(self, 'muscle_paths_all_frames') and len(self.muscle_paths_all_frames) > 0 
                    and len(self.muscle_paths_all_frames[0]) > i 
                    and self.muscle_paths_all_frames[0][i] is not None):
                    
                    muscle_path = self.muscle_paths_all_frames[0][i]
                    
                    if case == "2D":
                        (l,) = ax.plot(
                            muscle_path[:, non_zero_axes[0]], 
                            muscle_path[:, non_zero_axes[1]],
                            color=muscle_color,
                            linewidth=line_width,
                            solid_capstyle='round'
                        )
                    else:
                        (l,) = ax.plot(
                            muscle_path[:, 0],
                            muscle_path[:, 1], 
                            muscle_path[:, 2],
                            color=muscle_color,
                            linewidth=line_width,
                            solid_capstyle='round'
                        )
                
                self.muscle_lines.append(l)
            
            return self.muscle_lines

        elif mode == "update":
            frame = kwargs.get("frame")
            plot_objects = kwargs.get("plot_objects")
            muscle_lines = plot_objects
            
            # Update muscle lines
            for i, line in enumerate(muscle_lines):
                if i >= len(self.activations[frame]):
                    continue
                    
                # Get updated activation
                activation_val = float(self.activations[frame][i])
                activation_val = np.clip(activation_val, 0.0, 1.0)
                
                # Update color: light grey (0) to red (1)
                # Light grey: [0.8, 0.8, 0.8], Red: [1.0, 0.0, 0.0]
                grey_val = 0.8 * (1 - activation_val)
                muscle_color = mcolors.to_hex([grey_val + activation_val, grey_val, grey_val])
                line_width = 1.0 + 2.0 * activation_val
                
                line.set_color(muscle_color)
                line.set_linewidth(line_width)
                
                # Update muscle path if available
                if (hasattr(self, 'muscle_paths_all_frames') and frame < len(self.muscle_paths_all_frames) 
                    and len(self.muscle_paths_all_frames[frame]) > i
                    and self.muscle_paths_all_frames[frame][i] is not None):
                    
                    muscle_path = self.muscle_paths_all_frames[frame][i]
                    
                    if case == "2D":
                        line.set_data(muscle_path[:, non_zero_axes[0]], muscle_path[:, non_zero_axes[1]])
                    else:
                        line.set_data(muscle_path[:, 0], muscle_path[:, 1])
                        line.set_3d_properties(muscle_path[:, 2])
        
        else:
            raise ValueError("Invalid mode. Must be 'init' or 'update'.")

    def _get_muscle_attachment_points(self, states, constants, model):
        """Helper method to get muscle origin and insertion points using precomputed symbolic geometry."""
        
        if not hasattr(self, 'muscle_geometry'):
            raise ValueError("Muscle geometry not compiled. Call process_eom first.")
        
        n_muscles = self.n_actuators
        origins = np.zeros((n_muscles, 3))
        insertions = np.zeros((n_muscles, 3))
        paths = []
        
        for i in range(n_muscles):
            # Get the muscle path from precomputed symbolic geometry
            try:
                # Call the lambdified function with unpacked state and constant vectors
                muscle_path = self.muscle_geometry[i](*states.model, *constants.model)
                if muscle_path.shape[0] >= 2:
                    origins[i] = muscle_path[0]  # First point
                    insertions[i] = muscle_path[-1]  # Last point
                    paths.append(muscle_path)
                else:
                    raise ValueError(f"Muscle path for muscle {i} has insufficient points: {muscle_path.shape[0]}")
            except Exception as e:
                raise ValueError(f"Error computing muscle {i} geometry: {e}")
        
        # Store paths for use in plotting
        self.current_muscle_paths = paths
        return origins, insertions


    def _get_activations(self, states):
        """Helper method to extract muscle activations from states."""
        # Extract activation values from muscle states
        if states.actuator_model.ndim < 2:
            activations = states.actuator_model[self.idx["a"]]
        else:
            activations = states.actuator_model[:, self.idx["a"]]
        return np.array(activations).flatten()