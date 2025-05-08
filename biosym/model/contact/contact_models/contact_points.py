from biosym.model.contact.base_contact import BaseContact
import jax.numpy as jnp
import numpy as np
from sympy.physics.mechanics import Point
from sympy import Matrix, lambdify
import jax

class ContactPoints(BaseContact):
    def __init__(self, xml_root, body_weight=1):
        """
            Parses the contact model file and returns a list of contact points.
        """
        super().__init__(xml_root)
        # Get default values
        self.cp_defaults = xml_root.find("default/contact_point").attrib if xml_root.find("default/contact_point") is not None else {}
        cps = {}
        for cp in xml_root.findall("contact_point"):
            cp_name = cp.get("name")
            if cp_name is None:
                raise ValueError("Contact point must have a name")
            cps[cp_name] = {}
            for key, value in self.cp_defaults.items():
                if key == "pos":
                    cps[cp_name][key] = np.array([float(x) for x in cp.get(key, value).split()])
                else:
                    cps[cp_name][key] = cp.get(key, value)
        n_cps = len(cps)
        self.bodies = [cps[cp]['body'] for cp in cps]
        _, self.body_mapping = np.unique(self.bodies, return_inverse=True)
        self.body_mapping = np.tile(self.body_mapping, (3, 1))
        self.cps = cps

        # Get the contact point parameters
        self.k = [float(cps[cp]['k']) for cp in cps]
        self.b = [float(cps[cp]['b']) for cp in cps]
        self.mu = [float(cps[cp]['mu']) for cp in cps]
        self.p_cy_0 = [1e-3]*len(self.k) # Transition region size for position
        self.v_cx_0 = [1e-2]*len(self.k) # Transition region size for velocity

    def process_eom(self, model, body_weight=1):
        """ 
            Build the eom for the contact model with symbolic variables.
        """
        self.k = [k*body_weight for k in self.k]
        print(self.cps)
        cps_sympy = []
        pos_vector, vel_vector = [], []
        force_vector = []
        for i, cp in enumerate(self.cps):
            # Create a sympy point for the contact point
            cp_ = self.cps[cp]
            ref_frame = model.reference_frames[cp_['body']]
            origin = model.body_origins[cp_['body']]
            
            cp = Point(cp_['name'])
            cp.set_pos(origin, ref_frame.x*cp_['pos'][0] + ref_frame.y*cp_['pos'][1] + ref_frame.z*cp_['pos'][2])

            pos_vector.append([cp.pos_from(model.origin).dot(frame_dim) for frame_dim in [model.ground_frame.x, model.ground_frame.y, model.ground_frame.z]])
            vel_vector.append([cp.vel(model.ground_frame).dot(frame_dim) for frame_dim in [model.ground_frame.x, model.ground_frame.y, model.ground_frame.z]])
            pos_vector[-1] = model._replace_dyn(Matrix(pos_vector[-1])).T
            vel_vector[-1] = model._replace_dyn(Matrix(vel_vector[-1]))

            d = 0.5 * (pos_vector[-1][1]**2 + self.p_cy_0[i]**2) - pos_vector[-1][1]
            F_cy = self.k[i] * d * (1 - self.b[i] * vel_vector[-1][1])
            F_cx = -self.mu[i] * F_cy * vel_vector[-1][0] / (vel_vector[-1][0]**2 + self.v_cx_0[i]**2)**0.5
            F_cz = -self.mu[i] * F_cy * vel_vector[-1][2] / (vel_vector[-1][2]**2 + self.v_cx_0[i]**2)**0.5
            # Get F and M in the global frame
            force_vector.append([F_cx, F_cy, F_cz])
        force_vector = Matrix(force_vector)
        self.force_vector = lambdify(model._v, force_vector, modules='jax', cse=True, docstring_limit=2)
        pos_vector = Matrix(pos_vector)
        self.pos_vector = lambdify(model._v, pos_vector, modules='jax', cse=True, docstring_limit=2)

    def get_n_states(self):
        return 0
    
    def get_n_constants(self):
        # All constants are hardcoded in the process_eom sympy functions - can be changed in the future
        return 0
    
    def get_states(self):
        return []
    
    def get_constants(self):
        return []

    def get_bodies(self):
        """
            Returns the list of bodies that can be in contact.
        """
        return np.unique(self.bodies)
    
    def get_n_bodies(self):
        """
            Returns the number of bodies that can be in contact.
        """
        return len(np.unique(self.bodies))
    
    def get_cp_forces(self, states, constants, model):
        """
            Returns the contact forces for all bodies in the global frame.
            inputs:
                - states: The state of the model
                - model: The model object
            outputs:
                - contact_forces: The contact forces for all contact points in the global frame
        """
        cp_forces = self.force_vector(*states['model'], *constants['model'])
        return cp_forces
        
    def get_cp_moment_arms(self, states, constants, model):
        """
            Returns the moment arms for every contact point wrt to the body origin.
        """
        body_idx = np.array([list(model.rigid_bodies.keys()).index(p) for p in self.bodies])
        pos_bodies = model.run["FK"](states, constants)[body_idx]
        pos_cps = self.pos_vector(*states['model'], *constants['model'])
        return pos_cps - pos_bodies


    def forward(self, states, constants, model):
        """
            Returns the contact forces for all bodies in the global frame.
            inputs:
                - states: The state of the model
                - model: The model object
            outputs:
                - contact_forces: The contact forces for all bodies in the global frame
        """
        moment_arms = self.get_cp_moment_arms(states, constants, model)
        cp_forces = self.get_cp_forces(states, constants, model)

        
        # Get F and M in the global frame
        length = self.get_n_bodies()
        # Bincount is only 1D, therefore vmap it to 2D (note: vmap the whole model --> gpu version)
        def _bincount(arr, weights):
            return jnp.bincount(arr, weights=weights, length=length)
        foot_forces = jax.vmap(_bincount)(self.body_mapping, cp_forces.T).T

        moment_cps = jnp.cross(moment_arms, cp_forces)
        foot_moments = jax.vmap(_bincount)(self.body_mapping, moment_cps.T).T
        return foot_forces, foot_moments