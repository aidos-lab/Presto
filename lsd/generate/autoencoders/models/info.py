import torch
from torch.nn import functional as F

from lsd.generate.autoencoders.models.vae import BaseVAE


class InfoVAE(BaseVAE):
    def __init__(
        self,
        config,
        **kwargs,
    ) -> None:
        self.num_iter = 0
        super(InfoVAE, self).__init__(config)

        self.alpha = config.alpha
        self.beta = config.beta
        self.kernel_type = config.kernel
        self.reg_weight = config.reg_weight
        self.z_var = config.z_var
        self.eps = config.eps

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        kld_weight = kwargs[
            "M_N"
        ]  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(
            z,
            self.kernel_type,
            self.reg_weight,
            self.z_var,
            self.eps,
        )
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = (
            self.beta * recons_loss
            + (1.0 - self.alpha) * kld_weight * kld_loss
            + (self.alpha + self.reg_weight - 1.0) / bias_corr * mmd_loss
        )
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "MMD": mmd_loss,
            "KLD": -kld_loss,
        }


def initialize():
    return InfoVAE
