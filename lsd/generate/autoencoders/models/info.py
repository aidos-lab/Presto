import torch
from torch.nn import functional as F

from lsd.generate.autoencoders.models.vae import BaseVAE


class InfoVAE(BaseVAE):
    """
    Information Maximizing Variational Autoencoder (InfoVAE).

    This class implements the InfoVAE as described in:
    - "Information Maximizing Variational Autoencoder" (https://arxiv.org/pdf/1706.02262.pdf).

    InfoVAE is designed to increase the mutual information between the observed
    data and the latent variables, encouraging disentangled representations.

    Parameters
    ----------
    config : Config
        Configuration object containing hyperparameters and model settings.

    Attributes
    ----------
    alpha : float
        Weight of the mutual information term in the loss function.
    beta : float
        Weight of the reconstruction term in the loss function.
    kernel_type : str
        Type of kernel used in Maximum Mean Discrepancy (MMD) computation.
    reg_weight : float
        Regularization weight for the MMD term.
    z_var : float
        Variance of the latent variable's Gaussian distribution.
    eps : float
        Small value to avoid division by zero in MMD computation.
    num_iter : int
        Number of iterations for training.

    Methods
    -------
    loss_function(*args, **kwargs) -> dict
        Computes the loss for InfoVAE.
    """

    def __init__(
        self,
        config,
    ) -> None:
        """
        Constructor for the InfoVAE class.

        Parameters
        ----------
        config : Config
            Configuration object containing hyperparameters and model settings.
        """
        self.num_iter = 0
        super(InfoVAE, self).__init__(config)

        self.alpha = config.alpha
        self.beta = config.beta
        self.kernel_type = config.kernel
        self.reg_weight = config.reg_weight
        self.z_var = config.z_var
        self.eps = config.eps

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the loss for InfoVAE.

        The loss function is a combination of the reconstruction loss,
        a Maximum Mean Discrepancy (MMD) term, and a Kullback-Leibler
        divergence term to encourage disentangled latent space representation.

        Parameters
        ----------
        *args : tuple
            Variable length argument list, containing:
            - recons : torch.Tensor
                The reconstructed input from the decoder.
            - input : torch.Tensor
                The original input.
            - z : torch.Tensor
                The latent codes.
            - mu : torch.Tensor
                The mean of the latent Gaussian distribution.
            - log_var : torch.Tensor
                The log variance of the latent Gaussian distribution.
        **kwargs : dict
            Arbitrary keyword arguments, containing:
            - M_N : float
                Scaling factor for the KLD loss based on minibatch size.

        Returns
        -------
        dict
            A dictionary containing the computed losses:
            - 'loss' : torch.Tensor
                The total loss, combining reconstruction, MMD, and KLD losses.
            - 'Reconstruction_Loss' : torch.Tensor
                The mean squared error between the input and the reconstruction.
            - 'MMD' : torch.Tensor
                The Maximum Mean Discrepancy loss.
            - 'KLD' : torch.Tensor
                The Kullback-Leibler divergence loss.
        """
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


def initialize() -> InfoVAE:
    """
    Initializes the InfoVAE model.

    Returns
    -------
    InfoVAE
        The InfoVAE model instance.
    """
    return InfoVAE
