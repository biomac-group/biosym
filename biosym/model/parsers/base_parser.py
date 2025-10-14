from abc import ABC, abstractmethod


class BaseParser(ABC):
    """
    Abstract base class for model parsers.
    """

    def __init__(self, model_file):
        self.model_file = model_file
        self.model = None

    @abstractmethod
    def get_n_bodies(self):
        """
        Returns the number of bodies in the model.
        """

    @abstractmethod
    def get_n_joints(self):
        """
        Returns the number of joints in the model.
        """

    @abstractmethod
    def get_bodies(self):
        """
        Returns the list of bodies in the model.
        Each body is a dictionary with the following keys:
            - name: The name of the body
            - mass: The mass of the body
            - inertia: The inertia of the body
            - body_offset: The offset of the body
            - com: The center of mass of the body
            - parent: The parent body of the body
            - joints: The joint associated with the body
        """

    @abstractmethod
    def get_joints(self):
        """
        Returns the list of joints in the model.
        Each joint is a dictionary with the following keys:
            - name: The name of the joint
            - type: The type of the joint (e.g. revolute, prismatic)
            - axis: The axis of the joint
            - parent: The parent body of the joint
            - child: The child body of the joint

        We should add range / stiffness / damping / etc. here to be mujoco-feature complete
        """

    @abstractmethod
    def get_n_external_forces(self):
        """
        Returns the number of bodies, where external forces can be applied.
        """

    @abstractmethod
    def get_external_forces_bodies(self):
        """
        Returns the list of bodies, where external forces can be applied.
        """

    @abstractmethod
    def get_n_internal_forces(self):
        """
        Returns the number of internal forces in the model.
        """

    @abstractmethod
    def get_internal_forces(self):
        """
        Returns the list of internal forces in the model.
        """

    @abstractmethod
    def get_gravity(self):
        """
        Returns the gravity vector in the model.
        """

    @abstractmethod
    def get_n_sites(self):
       """
       Returns the number of sites (e.g., MuJoCo <site> elements or marker points)
       defined in the model.
        """

    @abstractmethod
    def get_sites(self):
        """
        Returns the list of sites in the model.
        Each site should be represented as a dictionary with keys such as:
            - name: site name
            - pos: position (x, y, z) relative to parent body/frame
            - body: parent body name (optional)
            - size: visual size (optional)
            - rgba: color/alpha tuple (optional)
        """
