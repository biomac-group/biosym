import os 
os.environ["JAX_COMPILATION_CACHE_DIR"] = "~/.biosym/jax_cache" # This needs to happen before importing jax
os.makedirs(os.path.expanduser("~/.biosym/jax_cache"), exist_ok=True)
from biosym.model.parsers import *
import sympy
from sympy import symbols, Matrix, lambdify, sqrt, Derivative, simplify
from sympy.physics.mechanics import KanesMethod, ReferenceFrame, Point, RigidBody, dynamicsymbols, Inertia
import jax.numpy as jnp
import numpy as np
import jax
import jax.export as export

class BiosymModel:
    """
        Biomechanical models are defined in this class.
        It will contain all functionality to load, save, and manipulate the model.
        Docstrings need to be added.
    """
    def __init__(self, definition_file, force_rebuild=False):
        # First, there should be a check if a pickled version of the model exists.
        if not force_rebuild:
            # Check if a pickled version of the model exists
            print('Warning: Loading from pickled model is not implemented yet.')
            # return the pickled file instead
            return # 

        # I think that it makes sense to force .yaml files at some point, because there are settings that are not represented in the model files
        if definition_file.endswith(".xml"):
            parser = mujoco_parser.MujocoParser(definition_file)
        elif definition_file.endswith(".osim"):
            raise NotImplementedError("OSIM models not supported yet.")
        elif definition_file.endswith(".yaml"):
            raise NotImplementedError("Loading models from yaml files is not supported yet.")
        else:
            raise ValueError("Model definition file must be in .xml, .osim, or .yaml format.")
        
        self.run = {}
    
        self._create_dictionaries(parser)
        self._create_sympy_model()
        # Future work: (These should be disabled or enabled by a flag in the config file); so that we don't need to compile everything every time
        self._create_eom()
        self._create_jax_eom()
        self._create_FK(True)
        # self._create_FK(parser)
        # self._create_IMU(parser) ....

    def _create_dictionaries(self, parser):
        """
            Create dictionaries for coordinates, speeds, accelerations, forces, joints, bodies, and external forces.
            Each contains:
            - list of names
            - start_index in the state vector
            - number of items
            These data is needed to build the state vector _v correctly and index on it
        """
        """
            ToDos: 
            - Treat constrained joints - they actually change the number of DOFs
            - Add muscles, they have a different number of states than torques
            - Mujoco: allow assignment of less bodies for external forces
            - All in all: work in progress, to be iterated over
            The current state, however, is good enough to define the OCP
        """

        self.dicts = {
            "bodies": parser.get_bodies(),
            "joints": parser.get_joints(),
        }

        # Create overviews of all states
        n_dof = parser.get_n_joints()
        self.coordinates = {
            'names': [f"q_{joint['name']}" for joint in parser.get_joints()],
            'idx': 0,
            'n': n_dof,
        }
        self.speeds = {
            'names': [f"qd_{joint['name']}" for joint in parser.get_joints()],
            'idx': n_dof,
            'n': n_dof,
        }
        self.accs = {
            'names': [f"qdd_{joint['name']}" for joint in parser.get_joints()],
            'idx': 2 * n_dof,
            'n': n_dof,
        }

        n_forces = parser.get_n_internal_forces()
        self.forces = {
            'names': [f"f_{force['name']}" for force in parser.get_internal_forces()],
            'idx': 3 * n_dof,
            'n': n_forces,
        }

        # The first representation of external forces is a list of bodies, where the forces can be applied
        n_ext_forces = parser.get_n_external_forces()
        self.ext_forces = {
            'names': [f"m_{force['name']}_{dim}" for force in parser.get_external_forces_bodies() for dim in ['x','y','z']],
            'idx': 3 * n_dof + n_forces,
            'n': n_ext_forces,
        } 
        
        # And torques
        self.ext_torques = {
            'names': [f"t_{force['name']}_{dim}" for force in parser.get_external_forces_bodies() for dim in ['x','y','z']],
            'idx': 3 * n_dof + n_forces + n_ext_forces,
            'n': n_ext_forces,
        }

        self.state_vector = (self.coordinates['names']+
                             self.speeds['names']+
                             self.accs['names']+
                             self.forces['names']+
                             self.ext_forces['names']+
                             self.ext_torques['names'])
        self.n_states = len(self.state_vector)

        # Create dictionaries for all constants
        i = len(self.state_vector)
        self.g = {
            'names': ['g_x', 'g_y', 'g_z'],
            'idx': i,
            'n': 3,
        }
        i += 3

        self.masses = {
            'names': [f"m_{body['name']}" for body in parser.get_bodies()],
            'idx': i,
            'n': len(parser.get_bodies()),
        }
        i += len(parser.get_bodies())

        inertia_tensor = ['Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']
        self.inertia = {
            'names': [f"I_{body['name']}_{dim}" for body in parser.get_bodies() for dim in inertia_tensor ],
            'idx': i,
            'n': len(parser.get_bodies()) * len(inertia_tensor),
        }
        i += len(parser.get_bodies()) * len(inertia_tensor)

        self.com = {
            'names': [f"com_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ['x', 'y', 'z']],
            'idx': i,
            'n': len(parser.get_bodies()) * 3,
        }
        i += len(parser.get_bodies()) * 3
        
        self.offset = {
            'names': [f"offset_{body['name']}_{dim}" for body in parser.get_bodies() for dim in ['x', 'y', 'z']],
            'idx': i,
            'n': len(parser.get_bodies()) * 3,
        }
        i += len(parser.get_bodies()) * 3

        self.constants = (self.g['names'] + 
                            self.masses['names'] +
                            self.inertia['names'] +
                            self.com['names'] +
                            self.offset['names'])
        self.n_constants = len(self.constants)

        self.gravity = parser.get_gravity()

    def _create_sympy_model(self):
        """
            Create the equations of motion (EOM) for the model.
            Every value should be stored in the vector self._v.
        """
        self._nv = self.n_states + self.n_constants
        # We set everything with the IndexedBase, so that it is vectorized
        # However, coordinates and speeds need to be dynamicsymbols, therefore we create a separate vector for them and merge them later
        self._v = sympy.IndexedBase('v', shape=(self._nv,))
        self._dynamic = Matrix(2*self.coordinates['n'], 1, lambda i, _: dynamicsymbols(f'dyn{i}'))
        # Maybe the IndexedBase needs to be initialized with its data types
        # e.g. [self._v[i] for i in :self.n_states] = [dynamicsymbols(name) for name in self.state_vector]; [self._v[i] for i in self.n_states:] = [symbols(name) for name in self.constants]
        self.ground_frame = ReferenceFrame('ground') # Fixed ground frame
        self.origin = Point('origin')
        self.origin.set_vel(self.ground_frame, 0) # For treadmill models, we could just set a velocity here
        self.body_origins = {}
        self.rigid_bodies = {}
        self.reference_frames = {}
        self.loads = []
        # kinematic differential equations: d coordinates - speeds = 0
        self.kd_eqs = [a-b for a,b in zip([self._dynamic[i].diff() for i in _slice(self.coordinates)], [self._dynamic[i] for i in _slice(self.speeds)])]
        
        # Create all bodies 
        # The slicing for bodies is not super clean - the idx are 
        self.default_values = np.zeros(self._nv)
        self.default_values[_slice(self.g)] = np.array(self.gravity)
        self.default_values[_slice(self.masses)] = np.array([body['mass'] for body in self.dicts['bodies']]).squeeze()
        def concat_defaults(value):
            """
                Concatenate all default values for the bodies
            """
            all_values = []
            for body in self.dicts['bodies']:
                all_values.append(body[value])
            # return a 
            return np.array(all_values).flatten()

        # Get all values that are stored as lists
        for value_dict, value in zip([self.com, self.offset, self.inertia],['com', 'body_offset', 'inertia']):
            self.default_values[_slice(value_dict)] = concat_defaults(value)

        # Get the model topology (A tree-like to help navigating the model)
        # This might change the indexing order of the bodies --> needs to be tested
        # Use body_idx as a substitute for now to be safe
        topology_tree = self._create_topology_tree()
        def build_reference_frames(topology, parent_frame=None, parent_origin=None):
            for idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body['name'] for body in self.dicts['bodies']].index(body_name)
                children = node["children"]

                # Get the current body data
                body = next((b for b in self.dicts['bodies'] if b['name'] == body_name), None)

                parent_frame = self.ground_frame if parent_frame is None else parent_frame
                parent_origin = self.origin if parent_origin is None else parent_origin

                body_origin = Point(f"{body_name}_origin")
                total_pos_offset = _to_sympy_vector([self._v[i] for i in range(self.offset['idx']+3*body_idx,self.offset['idx']+3*(body_idx+1))], parent_frame) 
                body_origin.set_pos(parent_origin, total_pos_offset)
                body_frame = ReferenceFrame(f"{body_name}_frame")

                # Accumulate joint axes, this is currently very messy and needs to be fixed
                # Yipeng's code is a bit cleaner, but has some double-setting values
                slide_axis = [0] * 3
                hinge_axis = [0] * 3
                slide_vel = [0] * 3
                hinge_vel = [0] * 3
                for joint in body['joints']:
                    joint_idx = self.coordinates['names'].index(f"q_{joint['name']}")
                    if joint['type'] == "slide":
                        for i in range(3):
                            slide_axis[i] += self._dynamic[self.coordinates['idx']+joint_idx]*joint['axis'][i]
                            slide_vel[i] += self._dynamic[self.speeds['idx']+joint_idx]*joint['axis'][i]
                    if joint['type'] == "hinge":
                        for i in range(3):
                            hinge_axis[i] += joint['axis'][i]#self._v[self.coordinates['idx']+joint_idx]*joint['axis'][i]
                            hinge_vel[i] += self._dynamic[self.speeds['idx']+joint_idx]*joint['axis'][i]
                slide_axis = _to_sympy_vector(slide_axis, parent_frame)
                hinge_axis = _to_sympy_vector(hinge_axis, parent_frame)
                slide_vel = _to_sympy_vector(slide_vel, parent_frame)
                hinge_vel = _to_sympy_vector(hinge_vel, parent_frame)
                body_origin.set_pos(parent_origin, slide_axis)
                body_origin.set_vel(parent_frame, slide_vel)
                self.body_origins[body_name] = body_origin
                print(f"Body {body_name} has slide axis {slide_axis} and hinge axis {hinge_axis}")
                if hinge_axis != 0:
                    body_frame.orient(parent_frame, 'Axis', (self._dynamic[self.coordinates['idx']+joint_idx]*joint['axis'][i], hinge_axis))
                    body_frame.set_ang_vel(parent_frame, hinge_vel)
                
                if slide_axis != 0:
                    print(slide_axis)
                    body_frame.set_ang_vel(parent_frame, body_origin.vel(parent_frame) + slide_vel * slide_axis)
                
                self.reference_frames[body_name] = body_frame
                build_reference_frames(children,body_frame, body_origin)
        build_reference_frames(topology_tree)

        def build_bodies(topology):
            for idx, node in enumerate(topology):
                body_name = node["name"]
                body_idx = [body['name'] for body in self.dicts['bodies']].index(body_name)
                children = node["children"]
                body_origin = self.origins[body_name]
                body_frame = self.reference_frames[body_name]

                # Set pos and lin_vel for the body
                mass_center_point = Point(f"{body_name}_mass_center")
                com_pos = _to_sympy_vector([self._v[i] for i in range(self.com['idx']+3*body_idx,self.com['idx']+3*(body_idx+1))], body_frame)
                mass_center_point.set_pos(body_origin, com_pos)
                mass_center_point.set_vel(body_origin, self.ground_frame, body_frame)
                self.mass_centers[body_name] = mass_center_point

                # set inertia tensor
                inertia_tensor = [[self._v[i] for i in self.inertia['idx']+6*body_idx+i] for i in range(6)]
                body_inertia = Inertia.from_inertia_scalars(mass_center_point, body_frame, *inertia_tensor)

                # Create the body
                body = RigidBody(body_name, mass_center_point, body_frame, [self._v[i] for i in self.masses['idx']+body_idx], body_inertia)
                self.rigid_bodies[body_name] = body

            build_bodies(children)
        
        # Add gravitational forces
        for bodyname, rigid_body in self.rigid_bodies.items():
            gravity_f = rigid_body.mass * (self.ground_frame.x * [self._v[i] for i in self.g['idx']] +
                                                self.ground_frame.y * [self._v[i] for i in self.g['idx']+1] +
                                                self.ground_frame.z * [self._v[i] for i in self.g['idx']+2])
            self.loads.append(gravity_f)

        # Add external forces - double check if this is correct
        # We are using body.origin to apply the force, but it could also be applied to the mass center or an arbitrary point - that is tbd
        for force in self.ext_forces['names']:
            body_name = force.split("_")[1]
            body_idx = [body['name'] for body in self.dicts['bodies']].index(body_name)
            force_idx = self.ext_forces['idx'] + 3 * body_idx
            force_vector = _to_sympy_vector([self._v[i] for i in range(force_idx,force_idx+3)], self.ground_frame)
            force_body = (self.body_origins[body_name], force_vector)
            self.loads.append(force_body)
        # Add external torques - also double check if this is correct
        for torque in self.ext_torques['names']:
            body_name = torque.split("_")[1]
            body_idx = [body['name'] for body in self.dicts['bodies']].index(body_name)
            torque_idx = self.ext_torques['idx'] + 3 * body_idx
            torque_vector = _to_sympy_vector([self._v[i] for i in range(torque_idx,torque_idx+3)], self.ground_frame)
            torque_body = (self.reference_frames[body_name], torque_vector)
            self.loads.append(torque_body)

    def _create_topology_tree(self):
        """
        Create a tree-like topology structure based on the hierarchical relationship
        of bodies defined in self.dicts['bodies']
        """
        # # First, create a dictionary mapping to quickly look up parent-child relationships
        # body_dict = {body['name']: body for body in self.dicts['bodies']}
        
        # Define a recursive function to build the tree structure
        def build_tree(parent_name):
            children = [
                body['name'] for body in self.dicts['bodies'] if body['parent'] == parent_name
            ]
            # Build the current node
            tree = {
                "name": parent_name,
                "children": [build_tree(child) for child in children]
            }
            return tree

        # Build topology starting from the root nodes
        root_bodies = [body['name'] for body in self.dicts['bodies'] if body['parent'] is None]
        topology_tree = [build_tree(root) for root in root_bodies if root != "ground"]  # Exclude ground
        return topology_tree
 
    def _create_eom(self):
        """
            Create the equations of motion (EOM) for the model.
            Every value should be stored in the vector self._v.
        """
        # Create the equations of motion using KanesMethod
        km = KanesMethod(self.ground_frame, 
                         q_ind = [self._dynamic[i] for i in _slice(self.coordinates)]
                            , u_ind = [self._dynamic[i] for i in _slice(self.speeds)]
                            , kd_eqs = self.kd_eqs)
        self.fr, self.frstar = km.kanes_equations(list(self.rigid_bodies.values()), self.loads)
        self.kane = km
        self.constants_sym = [self._v[i] for i in range(self.n_states, self._nv)]
        self.state_vector_sym = [self._v[i] for i in range(0,self.n_states)]
        self.eom = self.fr + self.frstar
        # replace the accelerations in the EOM with the v_ states
        print('Replacing accelerations in the EOM with the v_ states')
        in_ = [self._dynamic[i+self.speeds['idx']].diff() for i in range(self.speeds['n'])] 
        out_ = [self._v[i+self.accs['idx']] for i in range(self.accs['n'])] 
        self.eom = self.eom.xreplace(dict(zip(in_, out_)))
        print('Replacing speeds in the EOM with the v_ states')
        in_ = [self._dynamic[i+self.speeds['idx']] for i in range(self.speeds['n'])]
        out_ = [self._v[i+self.speeds['idx']] for i in range(self.speeds['n'])]
        self.eom = self.eom.xreplace(dict(zip(in_, out_)))
        print('Replacing coordinates in the EOM with the v_ states')
        in_ = [self._dynamic[i+self.coordinates['idx']] for i in range(self.coordinates['n'])]
        out_ = [self._v[i+self.coordinates['idx']] for i in range(self.coordinates['n'])]
        self.eom = self.eom.xreplace(dict(zip(in_, out_)))

    def _create_jax_eom(self):
        """
            Create the equations of motion (EOM) for the model using JAX.
            Every value should be stored in the vector self._v.
        """
        print('Lambdifying the EOM, this might take a while...')
        self.confun = lambdify(self._v, self.eom, modules='jax', cse=True, docstring_limit=2)
        self.jacobian = jax.jacobian(self.confun)
        self._precompile_fn(self.jacobian, self._nv, 'jacobian')
        self._precompile_fn(self.confun, self._nv, 'confun')
    
    def _create_FK(self, get_FK_dot=True):
        """
            Create the forward kinematics (FK) for the model.
            Every value should be stored in the vector self._v.  
            Currently, this just returns the positions of the body_origins in the global frame.    
            FK for markers etc. should be added in different functions --> max speed
            Currently, this takes in the     
        """
        self.positions = [body for body in self.body_origins.keys()]
        pos_vector = []
        for _, point in self.body_origins.items():
            pos_vector.append([point.pos_from(self.origin).dot(frame_dim) for frame_dim in [self.ground_frame.x, self.ground_frame.y, self.ground_frame.z]])
        pos_vector = Matrix(pos_vector)
        pos_vector = lambdify(self._v, pos_vector, modules='jax', cse=True, docstring_limit=2)
        self._precompile_fn(pos_vector, self._nv, 'FK')

        if get_FK_dot:
            vel_vector = []
            vel_vector.append([point.vel(self.ground_frame).dot(frame_dim) for frame_dim in [self.ground_frame.x, self.ground_frame.y, self.ground_frame.z]])
            vel_vector = Matrix(vel_vector)
            vel_vector = lambdify(self._v, vel_vector, modules='jax', cse=True, docstring_limit=2)
            self._precompile_fn(vel_vector, self._nv, 'FK_dot')

    def _precompile_fn(self, function, input_length, name):
        """
            Precompile a function using JAX's jit for faster execution.
            This is useful for functions that will be called multiple times with the same input shape.
            We use a hacky way: by serializing the function and then deserializing it again, the caching mechanism of jax doesn't miss parts of the function. 
            This actually doesn't even seem to be slower than the normal jax.jit
        """
        input_dummy = jax.ShapeDtypeStruct([input_length], np.float32)
        exported = jax.export.export(jax.jit(function))(input_dummy)
        re_ = jax.export.deserialize(exported.serialize(vjp_order=1))
        # Trigger the jit compilation
        re_.call(np.zeros(input_length, dtype=np.float32))
        # Add the function to the run dictionary
        self.run[name] = re_.call
        


def _slice(dictionary):
    """
        Slice the state vector according to the dictionary
        To make accesing states / v_ easier
    """
    return np.arange(dictionary['idx'], dictionary['idx'] + dictionary['n'])
    
def _to_sympy_vector(values, reference_frame):
    """
    Convert a list of positional or directional values to a sympy.physics.mechanics.Vector.
    
    Args:
        values (list or tuple): A list or tuple containing [x, y, z] values.
        reference_frame (ReferenceFrame): The reference frame to use for the vector conversion.
        
    Returns:
        Vector: A sympy Vector representation of the input values.
    """
    if len(values) != 3:
        raise ValueError("The input values must be a list or tuple of length 3.")
    return (
        values[0] * reference_frame.x +
        values[1] * reference_frame.y +
        values[2] * reference_frame.z
    )          

def load_model(model_file, force_rebuild=False):
    """
        Load a model from a file.
        The file can be in .xml, .osim, or .yaml format.
        The function will return a Model object.
    """
    # We should generate a hash of the config / or xml tree and save the cloudpickled model in the cache
    model = BiosymModel(model_file, force_rebuild)
    return model

# Small testing script
if __name__ == "__main__":
    model = load_model("tests/test_models/pendulum.xml",True)
    print(model.run['confun'](np.ones(model._nv)))
    print(model.run['jacobian'](np.ones(model._nv)))
    print(model.run['FK'](np.ones(model._nv)))
    print(model.run['FK_dot'](np.ones(model._nv)))
    print(model.fr+model.frstar)
    print(model.n_states, model.n_constants)
    print(model.state_vector)
    print(model.constants)