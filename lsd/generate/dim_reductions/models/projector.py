from abc import ABC, abstractmethod
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
