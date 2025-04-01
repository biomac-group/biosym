import xml.etree.ElementTree as ET
from biosym.model.parsers.base_parser import BaseParser

class MujocoParser(BaseParser):
    """
        Parser for Mujoco model files.
        Currently, this only builds the structure, but we also want to have the parameters
    """
    def __init__(self, model_file, verbose=False):
        super().__init__(model_file)
        self.xml_tree = ET.parse(model_file)
        self.root = self.xml_tree.getroot()
        self.data = {
            'bodies': [],
            'joints': [],
        }
        self._parse(verbose)


    def _parse(self, verbose=False):
        def parse_body(body_element, parent_name=None):
            # Parsing basic properties of body
            body_name = body_element.get("name")
            if verbose: print(f"Parsing body: {body_name}, Parent: {parent_name}") 

            # Extract body_offset from XML's "pos" attribute
            pos = body_element.get("pos", "0 0 0")  # Default to [0, 0, 0] if not provided
            body_offset = [float(x) for x in pos.split()]  # Convert to list of floats 

            # Extract mass, inertia, and com the same way
            body_mass = [float(body_element.get("mass", "0"))]
            body_inertia = [float(x) for x in body_element.get("inertia", "0 0 0 0 0 0").split()]
            body_com = [float(x) for x in body_element.get("com", "0 0 0").split()]

            # Parsing joint information in the current body
            body_joints = []
            for joint in body_element.findall("joint"):
                joint_name = joint.get("name")
                if not joint_name:
                    continue
                # print(f"Joint found in {body_name}: {joint_name}")
                joint_type = joint.get("type", None)
                joint_axis_values = [float(x) for x in joint.get("axis", "1 0 0").split()]
                if verbose: print(f"Joint axis values: {joint_axis_values}")
                joint_range_values = [float(x) for x in joint.get("range", "0 0").split()]

                    
                body_joints.append({
                    'name': joint_name,
                    'type': joint_type,
                    'axis': joint_axis_values,
                    'range': joint_range_values,
                    'parent': body_name,
                })

            self.data['joints'] += body_joints  # Add joints to the main list
            # Store the parsed body and its joints
            self.data['bodies'].append({
                'name': body_name,
                'parent': parent_name,
                'body_offset': body_offset,
                'mass': body_mass,
                'inertia': body_inertia,
                'com': body_com,
                'joints': body_joints,  # Joints under the current body
            })
            
            # Recursively parse the sub-body
            for child_body in body_element.findall("body"):
                parse_body(child_body, parent_name=body_name)

        # Start parsing from the root worldbody
        for body in self.root.findall(".//worldbody/body"):
            parse_body(body)

        # Mujoco does not have an explicit ground contact model stated, so we set all joints to possibly be in contact
        # For bigger models, it will be better to have fewer - so that they don't bother about compile time
        self.external_forces_bodies = [name for name in self.data['bodies'] if name['name']]

        # Get the default gravity vector - if <option gravity> is not set, it defaults to [0, -9.81, 0]
        gravity = self.root.find(".//option").get("gravity", "0 -9.81 0")
        self.gravity = [float(x) for x in gravity.split()]

        if verbose:
            # Print all parsed 'body_offset'
            print("\nParsed Body Offsets:")
            for body in self.data['bodies']:
                body_offset = body.get('body_offset')  
                print(f"  Body: {body['name']}, Offset: {body_offset}")

    def get_n_bodies(self):
        """
            Returns the number of bodies in the model.
        """
        return len(self.data['bodies'])

    def get_n_joints(self):
        """
            Returns the number of joints in the model.
        """
        return len(self.data['joints'])
    
    def get_bodies(self):
        """
            Returns the list of bodies in the model.
        """
        return self.data['bodies']
    
    def get_joints(self):
        """
            Returns the list of joints in the model.
        """
        return self.data['joints']
    
    def get_n_external_forces(self):
        """
            Returns the number of bodies, where external forces can be applied.
        """
        return len(self.external_forces_bodies) * 3
    
    def get_external_forces_bodies(self):
        """
            Returns the list of bodies, where external forces can be applied.
        """
        return self.external_forces_bodies
    
    def get_n_internal_forces(self):
        """
            Returns the number of internal forces in the model.
        """
        import warnings
        warnings.warn("Internal forces are not parsed yet.")
        print("Returning n_joints as n_internal forces")
        return len(self.data['joints'])
        raise NotImplementedError("Internal forces are not parsed yet.")
    
    def get_internal_forces(self):
        """
            Returns the list of internal forces in the model.
        """
        import warnings
        warnings.warn("Internal forces are not parsed yet.")
        print("Returning joints as internal forces")
        return self.data['joints']
        raise NotImplementedError("Internal forces are not parsed yet.")
    
    def get_gravity(self):
        """
            Returns the gravity vector in the model.
        """
        return self.gravity
    
    def get_bodies(self):
        """
            Returns the list of bodies in the model.
        """
        return self.data['bodies']
    
    def get_joints(self):
        """
            Returns the list of joints in the model.
        """
        return self.data['joints']
    