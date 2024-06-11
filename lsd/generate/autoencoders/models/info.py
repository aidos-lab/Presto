import functools
import operator

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
        mmd_loss = self.compute_mmd(z)
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

    def compute_kernel(
        self, x1: torch.tensor, x2: torch.tensor
    ) -> torch.tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == "rbf":
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == "imq":
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError("Undefined kernel type.")

        return result

    def compute_rbf(
        self, x1: torch.tensor, x2: torch.tensor, eps: float = 1e-7
    ) -> torch.tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (torch.tensor)
        :param x2: (torch.tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(
        self, x1: torch.tensor, x2: torch.tensor, eps: float = 1e-7
    ) -> torch.tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (torch.tensor)
        :param x2: (torch.tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: torch.tensor) -> torch.tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = (
            prior_z__kernel.mean()
            + z__kernel.mean()
            - 2 * priorz_z__kernel.mean()
        )
        return mmd


def initialize():
    return InfoVAE
