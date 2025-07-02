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
            "bodies": [],
            "joints": [],
            "sites": [],  # Sites are plotting elements in mujoco
        }
        self._parse(verbose)

    def _parse(self, verbose=False):
        def parse_body(body_element, parent_name=None):
            # Parsing basic properties of body
            body_name = body_element.get("name")
            if verbose:
                print(f"Parsing body: {body_name}, Parent: {parent_name}")

            # Extract body_offset from XML's "pos" attribute
            pos = body_element.get(
                "pos", "0 0 0"
            )  # Default to [0, 0, 0] if not provided
            body_offset = [float(x) for x in pos.split()]  # Convert to list of floats

            # Extract mass, inertia, and com the same way
            inertia_element = body_element.find("inertial")
            body_mass = [float(inertia_element.get("mass", "0"))]
            body_inertia = [
                float(x)
                for x in inertia_element.get("fullinertia", "0 0 0 0 0 0").split()
            ]
            body_com = [float(x) for x in inertia_element.get("pos", "0 0 0").split()]

            # Parsing joint information in the current body
            body_joints = []
            for joint in body_element.findall("joint"):
                joint_name = joint.get("name")
                if not joint_name:
                    continue
                # print(f"Joint found in {body_name}: {joint_name}")
                joint_type = joint.get("type", None)
                joint_axis_values = [
                    float(x) for x in joint.get("axis", "1 0 0").split()
                ]
                if verbose:
                    print(f"Joint axis values: {joint_axis_values}")
                joint_range_values = [
                    float(x) for x in joint.get("range", "0 0").split()
                ]

                body_joints.append(
                    {
                        "name": joint_name,
                        "type": joint_type,
                        "axis": joint_axis_values,
                        "range": joint_range_values,
                        "parent": body_name,
                    }
                )

            for site in body_element.findall("site"):
                site_name = site.get("name")
                if not site_name:
                    continue
                # print(f"Site found in {body_name}: {site_name}")
                site_pos = [float(x) for x in site.get("pos", "0 0 0").split()]
                self.data["sites"].append(
                    {
                        "name": site_name,
                        "pos": site_pos,
                        "parent": body_name,
                    }
                )

            self.data["joints"] += body_joints  # Add joints to the main list
            # Store the parsed body and its joints
            self.data["bodies"].append(
                {
                    "name": body_name,
                    "parent": parent_name,
                    "body_offset": body_offset,
                    "mass": body_mass,
                    "inertia": body_inertia,
                    "com": body_com,
                    "joints": body_joints,  # Joints under the current body
                }
            )

            # Recursively parse the sub-body
            for child_body in body_element.findall("body"):
                parse_body(child_body, parent_name=body_name)

        # Start parsing from the root worldbody
        for body in self.root.findall(".//worldbody/body"):
            parse_body(body)

        # Mujoco often does not have an explicit ground contact model stated, so we set all joints to possibly be in contact
        # The contact model should come from a yaml file.
        # Mujoco GC models are currently not supported - but the definition in the mujoco file is possible
        self.external_forces_bodies = []
        self.contact_elements = self.root.findall(".//contact")
        if self.contact_elements != []:
            assert (
                len(self.contact_elements) == 1
            ), "Only one contact element is allowed in the mujoco model file"
            self.contact_elements = self.contact_elements[0]
        else:
            self.contact_elements = None

        # Get the default gravity vector - if <option gravity> is not set, it defaults to [0, -9.81, 0]
        gravity = self.root.find(".//option").get("gravity", "0 -9.81 0")
        self.gravity = [float(x) for x in gravity.split()]

        # We currently only allow externally specified actuators
        # to be in the mujoco model file, so we set the actuator list to None
        self.actuator_elements = self.root.findall(".//actuator")
        if self.actuator_elements != []:
            assert (
                len(self.actuator_elements) == 1
            ), "Only one actuator element is allowed in the mujoco model file"
            self.actuator_elements = self.actuator_elements[0]
        else:
            self.actuator_elements = None

        if verbose:
            # Print all parsed 'body_offset'
            print("\nParsed Body Offsets:")
            for body in self.data["bodies"]:
                body_offset = body.get("body_offset")
                print(f"  Body: {body['name']}, Offset: {body_offset}")

    def get_n_bodies(self):
        """
        Returns the number of bodies in the model.
        """
        return len(self.data["bodies"])

    def get_n_joints(self):
        """
        Returns the number of joints in the model.
        """
        return len(self.data["joints"])

    def get_bodies(self):
        """
        Returns the list of bodies in the model.
        """
        return self.data["bodies"]

    def get_joints(self):
        """
        Returns the list of joints in the model.
        """
        return self.data["joints"]

    def get_n_sites(self):
        """
        Returns the number of sites in the model.
        """
        return len(self.data["sites"])

    def get_sites(self):
        """
        Returns the list of sites in the model.
        """
        return self.data["sites"]

    def get_n_external_forces(self):
        """
        Returns the number of bodies, where external forces can be applied.
        In mujoco, ground contact isn't explicitly stated, so that should be in a config somewhere
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
        return len(self.actuators) if self.actuators is not None else 0

    def get_internal_forces(self):
        """
        Returns the list of internal forces in the model.
        """
        return self.actuators

    def get_gravity(self):
        """
        Returns the gravity vector in the model.
        """
        return self.gravity

    def get_bodies(self):
        """
        Returns the list of bodies in the model.
        """
        return self.data["bodies"]

    def get_joints(self):
        """
        Returns the list of joints in the model.
        """
        return self.data["joints"]

    def has_actuators(self):
        """
        Returns True if the model has actuators, False otherwise.
        """
        return self.actuator_elements is not None

    def get_actuators(self):
        """
        Returns the xml entries for the actuators in the model.
        """
        return self.actuator_elements

    def has_contact_model(self):
        """
        Returns True if the model has a contact model, False otherwise.
        """
        return self.contact_elements is not None

    def get_contact_model(self):
        """
        Returns the xml entries for the contact model in the model.
        """
        return self.contact_elements
