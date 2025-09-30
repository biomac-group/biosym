import jax.numpy as jnp
import numpy as np

from biosym.model.actuators.base_actuator import BaseActuator

JOINT_RANGE_TOL = np.deg2rad(2)  # 2 degrees transition zone for joint limits


class Hill2d(BaseActuator):
    """
    A reimplementation of the 2D Hill muscle model as in gait2d.
    """

    def __init__(self, joints_dict, muscles_dict, defaults):
        super().__init__(joints_dict)
        self.muscles_dict = muscles_dict

        # Grab the first muscle from the defaults if available
        if defaults is not None:
            defaults = defaults.findall('muscle')[0].attrib
            print("Default muscle parameters:", defaults)

        self.n_actuators = len(muscles_dict)
        self.actuators = {}

        self.names = [mi.get("name") for mi in muscles_dict]

        self.muscle_constants = {}
        for const in ["fmax", "lceopt", "width", "vmax", "umax", "Arel", "gmax", "kPEE", "PEEslack", "kSEE", "SEEslack", "Tact", "Tdeact"]:
            self.muscle_constants[const] = jnp.array([float(mi.get(const, defaults.get(const, 0.0))) for mi in muscles_dict])
        
        moment_arm_matrix = jnp.zeros((self.n_actuators, len(joints_dict)))
        for muscle, idx in enumerate(muscles_dict):
            for dof in idx.findall("dof"):
                joint_name = dof.get("name")
                joint_idx = joints_dict.index(joint_name)
                moment_arm = float(dof.get("momentarm"))
                moment_arm_matrix = moment_arm_matrix.at[muscle, joint_idx].set(moment_arm)

        print(moment_arm_matrix)
        self.joints = jnp.array([float(mi.get("joint", defaults.get("joint", 0.0))) for mi in muscles_dict])  # Joint angles
        print("Muscles:", self.names)
        self.states = [f"Lce_{n}" for n in self.names] + [f"Lce_dot_{n}" for n in self.names] + [f"a_{n}" for n in self.names]
        self.idx = {
            "Lce": jnp.arange(0, self.n_actuators),
            "Lce_dot": jnp.arange(self.n_actuators, 2 * self.n_actuators),
            "a": jnp.arange(2 * self.n_actuators, 3 * self.n_actuators),
        }
        self.states_min = np.zeros(len(self.states))
        self.states_max = np.ones(len(self.states))

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

    def process_eom(self, model):
        return super().process_eom(model)

    def forward(self, states, constants, model):
        _, F_see, _ = self.muscle_equations(states, constants, model)
        # Todo: Hill's equations in here
        # What is the force at every joint?
        return self.moment_arm_matrix.T @ F_see
    
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

        # You'll need to extract these from states - adjust based on your state structure
        L_ce = states.actuator_states[self.idx["Lce"]]  # Assuming first column is L_ce
        L_ce_dot = states.actuator_states[self.idx["Lce_dot"]]  # Assuming second column is L_ce_dot
        a = states.actuator_states[self.idx["a"]]  # Assuming third column is activation


        x = (L_ce - 1) / W # L_ce: Normalized contractile element length, W: Width of the force-length relationship
        # Force-length relationship
        F1 = jnp.exp(-x**2)

        # Force-velocity relationship
        c_3 = V_max * A * (G_max - 1) / (A + 1)
        F2 = jnp.where(L_ce_dot < 0, (V_max + L_ce_dot)/(V_max - L_ce_dot/A), (G_max*L_ce_dot+c_3)/(L_ce_dot+c_3))

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
        x = Lm - L_ce*L_ce_opt - see_slack
        F_see = 0.01 * F_max * x  # Assuming k1 should be 0.01 * F_max
        F_see = jnp.where(x > 0, F_see + k_see * x**2, F_see)

        # F_ce
        F_ce = a * F1 * F2 + F_damp

        return F_max * F_ce, F_max * F_see, F_max * F_pee

    def constraints(self, states, constants, model):
        F_ce, F_see, F_pee = self.muscle_equations(states, constants, model)
        states, globals = states
        c1 = (F_see - F_ce - F_pee) / F_max  # Normalized to F_max

        if globals is not None:
            a = states.actuator_states[:,1:] # at idx of actuators I guess, all but the last state
            a_max = 1 + (states.actuator_states[:,-1] - 1) * jnp.exp(-globals.h/constants.T_act) # Exponential decay to 1
            a_min = (states.actuator_states[:,-1]) * jnp.exp(-globals.h/constants.T_deact) # Exponential decay to 0
            c2 = jnp.where(a > a_max, a_max - a, 0) + jnp.where(a < a_min, a - a_min, 0)
        


        # Activation dynamics
        ## I think that e doesn't really matter if we shouldn't optimize for it anyways
        """
            Why to not optimize for e as in BioMacSimToolbox:
            Activating e every 4th node (dt=0.01): c = 0.25 -> a = [0,1,0.66,0.33] - average: 0.5
            Activating e every 2nd node (dt=0.01): c = 0.5 -> a = [0.66,1,0.66,1] - average: 0.83
            Activating e continuously (dt=0.01), e = 0.5 -> c = 0.25 -> a = 0.5 - average: 0.5
            Activating e continuously (dt=0.01), e = 0.707 -> c = 0.5 -> a = 0.707 - average: 0.707 --> Lower than when jittering e
            Even worse:
            Activating e every 2nd node (dt=0.02): c = 0.5 -> a = [0, *2, 1.33, *0.67, 0.44,*1.67, 0.89, *1.33, 0.88] - average: 1.02
            Activating e strategically 1 (dt=0.02): c = 0.375 -> a = [0, *2, 1.33, 0.87, 0.58, *1.62, 1.07, 0.71, *1.29] - average: 1.05 --> This must be super bad for IPOPT

            So i think all we need to account for activation / deactivation dynamics is that the \dot{a} is limited by [1/t_act, 1/t_deact]

            How to optimize for e then?

            Do not: https://www.biorxiv.org/content/10.1101/2025.01.30.635759v1.full.pdf
            But if you really want to: a[t+1] = e[t] + (a[t] - e[t]) * np.exp(-(e[t]/Tact+(1-e[t])/Tdeact)*t)

            Recommendation: Do not optimize for e at all, do not allow a>1, and limit \dot{a} as stated here:
            a[t+1,max] = 1 + ( a[t] - 1 ) * exp(-dt/Tact) # Exponential decay to 1
            a[t+1,min] = (a[t]) * exp(-dt/Tdeact) # Exponential decay to 0

            So the constraint would be linear violation of this term
        """


        


