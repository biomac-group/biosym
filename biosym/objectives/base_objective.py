import abc


class BaseObjective(abc.ABC):
    """
    Abstract base class for objectives in the biosym package.
    All objectives should inherit from this class and implement the required methods.
    """

    def __init__(self, model, settings):
        """
        Initialize the BaseObjective class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        """
        self.model = model
        self.settings = settings

    def _get_info(self):
        """
        Get information about the objective function.
        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": self.__class__.__name__,
            "description": "Base objective class for biosym objectives.",
            "required_variables": None,
        }

    @abc.abstractmethod
    def get_objfun(self, *args, **kwargs):
        """
        Evaluate the objective function.

        :param args: Positional arguments for evaluation.
        :param kwargs: Keyword arguments for evaluation.
        :return: The evaluated value of the objective function.
        """

    @abc.abstractmethod
    def get_gradient(self, *args, **kwargs):
        """
        Compute the gradient of the objective function.

        :param args: Positional arguments for gradient computation.
        :param kwargs: Keyword arguments for gradient computation.
        :return: The gradient of the objective function.
        """
