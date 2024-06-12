import torch
from torch.nn import functional as F


from lsd.generate.autoencoders.models.vae import BaseVAE


class BetaVAE(BaseVAE):
    def __init__(
        self,
        config,
        **kwargs,
    ) -> None:
        self.num_iter = 0
        super(BetaVAE, self).__init__(config)

        # Beta VAE parameters
        self.beta = config.beta
        self.gamma = config.gamma
        self.loss_type = config.loss_type
        self.C_max = torch.Tensor([config.max_capacity])
        self.C_stop_iter = config.Capacity_max_iter

    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs[
            "M_N"
        ]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter,
                0,
                self.C_max.data[0],
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
        }


def initialize():
    return BetaVAE
