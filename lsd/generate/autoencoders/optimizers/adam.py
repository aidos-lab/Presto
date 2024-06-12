from torch import optim


class Adam:
    def __init__(
        self,
        model_parameters,
        learning_rate=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        self.optimizer = optim.Adam(
            model_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def initialize():
    return Adam
