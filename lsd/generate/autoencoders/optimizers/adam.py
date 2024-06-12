from torch import optim
import omegaconf


class Adam:
    def __init__(
        self,
        model_parameters,
        config: omegaconf.DictConfig,
    ):
        self.optimizer = optim.Adam(
            model_parameters,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    def step(self):
        self.optimizer.step()

    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)


def initialize():
    return Adam
