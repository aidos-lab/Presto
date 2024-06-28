from torch import optim
from lsd.utils import ConfigType, get_parameters
from torch.nn import Module


class BaseTorchOptimizer:
    """
    A Base class for handling PyTorch optimizers within LSD.

    This class provides a generic interface to initialize and use PyTorch optimizers based on a given configuration. It is designed to be flexible and allows for easy extension and integration with different optimization algorithms.

    Parameters
    ----------
    model_parameters : iterable
        Iterable of parameters to optimize or dicts defining parameter groups which is instantiated in `lsd.generate.autoencoders.gym`.
    config : ConfigType
        A configuration object that includes the name of the optimizer and its parameters.

    Attributes
    ----------
    optimizer_name : str
        The name of the optimizer to use, extracted from the configuration.
    optimizer : torch.optim.Optimizer
        The instantiated PyTorch optimizer.

    Methods
    -------
    setup(config, model_parameters) -> optim.Optimizer
        Configures and initializes the optimizer.
    step()
        Performs a single optimization step.
    zero_grad(**kwargs)
        Clears the gradients of all optimized parameters.
    validate_config(config) -> str
        Validates the configuration and retrieves the optimizer name.
    """

    def __init__(self, model_parameters, config: ConfigType):
        """
        Initialize the optimizer based on the given configuration and model parameters.

        Parameters
        ----------
        model_parameters : iterable
            Iterable of parameters to optimize or dicts defining parameter groups.
        config : ConfigType
            A configuration object that includes the name of the optimizer and its parameters.
        """
        self.optimizer_name = self.validate_config(config)
        self.optimizer = self.setup(config, model_parameters)

    def setup(
        self, config: ConfigType, model_parameters: Module.parameters
    ) -> optim.Optimizer:
        """
        Configures and initializes the optimizer based on the provided configuration and model parameters.

        Parameters
        ----------
        config : ConfigType
            A configuration object containing the optimizer parameters.
        model_parameters : Module.parameters
            The parameters of the model to be optimized.

        Returns
        -------
        optim.Optimizer
            An instance of the specified optimizer configured with the provided parameters.
        """
        optimizer = getattr(optim, self.optimizer_name)
        parameters = get_parameters(optimizer, config)
        parameters["params"] = model_parameters
        return optimizer(**parameters)

    def step(self):
        """
        Performs a single optimization step.

        This method updates the parameters based on the current gradients.
        """
        self.optimizer.step()

    def zero_grad(self, **kwargs):
        """
        Clears the gradients of all optimized parameters.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional arguments for `zero_grad` method.
        """
        self.optimizer.zero_grad(**kwargs)

    def validate_config(self, config: ConfigType) -> str:
        """
        Validates the configuration to ensure the specified optimizer exists.

        Parameters
        ----------
        config : ConfigType
            A configuration object that includes the optimizer name.

        Returns
        -------
        str
            The name of the optimizer.

        Raises
        ------
        AssertionError
            If the optimizer name specified in the configuration does not exist in `torch.optim`.
        """
        name = config.generators[-1]
        assert hasattr(
            optim, name
        ), f"Optimizer {name} not found in torch.optim, please check your configuration."
        return name


def initialize() -> BaseTorchOptimizer:
    """
    Factory function to create an instance of `BaseTorchOptimizer`.

    Returns
    -------
    BaseTorchOptimizer
        An instance of `BaseTorchOptimizer`.
    """
    return BaseTorchOptimizer
