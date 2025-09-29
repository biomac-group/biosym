import jax.numpy as jnp
import numpy as np

from biosym.model.actuators.base_actuator import BaseActuator

JOINT_RANGE_TOL = np.deg2rad(2)  # 2 degrees transition zone for joint limits


class Hill2d(BaseActuator):
    """
    A reimplementation of the 2D Hill muscle model as in gait2d.
    """

    def __init__(self, joints_dict, muscles_dict):
        super().__init__(joints_dict)
        self.muscles_dict = muscles_dict

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
        moments = jnp.zeros((model.coordinates["n"],))
        # Todo: Hill's equations in here
        # What is the force at every joint?
        return moments
    
    def constraints(self, states, constants, model):
        # Constraction dynamics
        F_see, F_ce, F_pee = 0, 0, 0  # Placeholder
        c1 = F_see - F_ce - F_pee

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
            a[t+1,max] = 1 + ( a[t] - 1 ) * exp(-dt/Tdeact)
            a[t+1,min] = 0 + ( a[t] - 0 ) * exp(-dt/Tact)

            So the constraint would be linear violation of this term
        """


        


