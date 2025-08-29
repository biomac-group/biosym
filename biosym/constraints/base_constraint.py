import abc

class BaseConstraint(abc.ABC):
    """
    Abstract base class for constraints in the biosym package.
    All constraints should inherit from this class and implement the required methods.
    """
    def __init__(self, model, settings, args):
        """
        Initialize the BaseConstraint class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the constraint.
        """
        self.model = model
        self.settings = settings

    def _get_info(self):
        """
        Get information about the constraint function.
        This method can be overridden in subclasses to provide specific information.
        """
        return {
            'name': self.__class__.__name__,
            'description': 'Base constraint class for biosym constraints.',
            'n_constraints': self.get_n_constraints(),
            'required_variables': None,
            'nnz': self.get_nnz()
        }

    @abc.abstractmethod
    def get_confun(self, *args, **kwargs):
        """
        Evaluate the constraint function.

        :param args: Positional arguments for evaluation.
        :param kwargs: Keyword arguments for evaluation.
        :return: The constraint function.
        """
        pass

    @abc.abstractmethod
    def get_jacobian(self, *args, **kwargs):
        """
        Compute the Jacobian of the constraint function.

        :param args: Positional arguments for Jacobian computation.
        :param kwargs: Keyword arguments for Jacobian computation.
        :return: The Jacobian of the constraint function.
        """
        pass

    @abc.abstractmethod
    def get_n_constraints(self):
        """
        Get the number of constraints defined by this constraint function.

        :return: The number of constraints.
        """
        pass

    @abc.abstractmethod
    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the constraint function.

        :return: The number of non-zero entries.
        """
        pass