from abc import ABC, abstractmethod
from inspect import signature
import numpy as np

from lsd.generate.dim_reductions.configs import Projector as ProjectorConfig


class BaseProjector(ABC):
    """
    Base class for all projector classes.

    This abstract base class provides a common interface for all projector implementations.

    Attributes
    ----------
    config : Projector
        The configuration object containing parameters for the projection algorithm.

    Methods
    -------
    project(data)
        Abstract method to perform the projection on the provided data. Must be implemented by subclasses.

    _get_parameters(Operator, config) -> dict
        Static method to extract relevant parameters from the configuration for the given operator.
    """

    def __init__(self, config: ProjectorConfig):
        """
        Constructor method for BaseProjector class.

        Parameters
        ----------
        config : Projector
            The configuration object containing parameters for the projection.
        """
        self.config = config

    @abstractmethod
    def project(self, data) -> np.ndarray:
        """
        Abstract method to project the data into a lower-dimensional space.

        This method must be implemented by subclasses to perform the actual
        dimensionality reduction based on the specific algorithm.

        Parameters
        ----------
        data : array-like
            The input data to be projected.

        Returns
        -------
        np.ndarray
            The projected data.
        """
        raise NotImplementedError()

    @staticmethod
    def _get_parameters(Operator, config) -> dict:
        """
        Extracts the relevant parameters from the configuration for the given operator.

        This static method uses introspection to retrieve the parameters of the operator's
        `__init__` method and matches them with the corresponding values in the configuration.

        Parameters
        ----------
        Operator : class
            The projection operator class whose parameters are to be retrieved.
        config : Projector
            The configuration object containing parameters for the projection.

        Returns
        -------
        dict
            A dictionary of parameters for the operator, populated with values from the configuration.
        """
        params = signature(Operator.__init__).parameters
        args = {
            name: config.get(name, param.default)
            for name, param in params.items()
            if param.default != param.empty
        }
        return args


def initialize() -> BaseProjector:
    """
    Initializes and returns the BaseProjector class.

    This function is intended to provide a convenient way to
    get an instance of the BaseProjector class.

    Returns
    -------
    BaseProjector
        The BaseProjector class.
    """
    return BaseProjector
