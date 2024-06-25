" Base class for Multiverse Generators."

from abc import ABC, abstractmethod
from typing import Union
import omegaconf


class Base(ABC):
    """
    Abstract Base Class for Multiverse Generators.

    This class provides the foundational structure for creating various
    multiverse generators. It outlines the essential methods and attributes
    that each generator class must implement.

    So far, `Presto` supports the following types of generators:
    - Dimensionality Reduction in `lsd.generate.dim_reductions.dr`
    - Variational Autoencoder `lsd.generate.autoencoders.ae`

    Parameters
    ----------
    params : Union[dict, omegaconf.DictConfig]
        Configuration parameters for setting up the multiverse generator. This
        can be a dictionary or an `omegaconf.DictConfig` object.

    Attributes
    ----------
    params : Union[dict, omegaconf.DictConfig]
        Stores the configuration parameters provided during instantiation.

    Methods
    -------
    setup()
        Abstract method for setting up the generator. This should include
        configuration and initialization processes specific to the generator.
    train()
        Abstract method for training the generator. Implementations should
        define the training procedures for the generator.
    generate()
        Abstract method for generating output. This is where the main logic for
        generating results or representations should be implemented.

    Notes
    -----
    This is an abstract base class and cannot be instantiated directly. Child
    classes must implement all abstract methods.
    """

    def __init__(self, params: Union[dict, omegaconf.DictConfig]):
        self.params = params

    @abstractmethod
    def setup(self):
        """
        Set up the generator.

        This method must be overridden in child classes to perform
        the necessary setup tasks for the generator, such as loading
        configurations, preparing data, and initializing models.

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented in a subclass.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Train the generator.

        This method must be overridden in child classes to define the
        training procedures for the generator. It may include training
        machine learning models or other processes necessary for generating
        output.

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented in a subclass.
        """
        pass

    @abstractmethod
    def generate(self):
        """
        Generate latent spaces using a configured model.

        This method must be overridden in child classes to define the logic
        for generating results or representations. This could involve using
        trained models or other mechanisms to produce the desired output.

        Raises
        ------
        NotImplementedError
            This method is abstract and must be implemented in a subclass.
        """
        pass
