from torch import optim
import omegaconf


class SGD:
    def __init__(
        self,
        model_parameters,
        config: omegaconf.DictConfig,
    ):
        """
        Initializes the SGD optimizer with given parameters.

        Parameters:
        - model_parameters: Iterable of parameters to optimize or dicts defining parameter groups.
        - learning_rate (float, optional): Learning rate (default: 0.01).
        - momentum (float, optional): Momentum factor (default: 0.9).
        - weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        - nesterov (bool, optional): Enables Nesterov momentum (default: False).
        """
        self.optimizer = optim.SGD(
            model_parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )

    def step(self):
        """Performs a single optimization step."""
        self.optimizer.step()

    def zero_grad(self, **kwargs):
        """Sets the gradients of all optimized torch.Tensor s to zero."""
        self.optimizer.zero_grad(**kwargs)


def initialize():
    return SGD
