import torch
from torch.nn import functional as F


from lsd.generate.autoencoders.models.vae import BaseVAE


class BetaVAE(BaseVAE):
    """
    Beta Variational Autoencoder (BetaVAE).

    This class implements the β-VAE as described in
    the papers:
    - "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
      (https://openreview.net/forum?id=Sy2fzU9gl)
    - "Understanding disentangling in β-VAE" (https://arxiv.org/pdf/1804.03599.pdf)

    This class extends the `BaseVAE` class with a loss function that compute's the β-VAE loss as described in https://openreview.net/forum?id=Sy2fzU9gl & https://arxiv.org/pdf/1804.03599.

    Parameters
    ----------
    config : Config
        Configuration object containing hyperparameters and model settings.

    Attributes
    ----------
    beta : float
        The weight of the KLD term in the loss function. Used in 'H' loss type.
    gamma : float
        The weight of the KLD term in the loss function. Used in 'B' loss type.
    loss_type : str
        Type of loss to use ('H' for heuristic, 'B' for β-VAE loss).
    C_max : torch.Tensor
        The maximum capacity for the latent space.
    C_stop_iter : int
        The iteration number at which capacity `C_max` is reached.
    num_iter : int
        The number of iterations for training.

    Methods
    -------
    loss_function(*args, **kwargs) -> dict
        Computes the loss for BetaVAE.
    """

    def __init__(
        self,
        config,
    ) -> None:
        """
        Constuctor for the BetaVAE class.

        Parameters
        ----------
        config : Config
            Configuration object containing hyperparameters and model settings.
        """
        self.num_iter = 0
        super(BetaVAE, self).__init__(config)

        # Beta VAE parameters
        self.beta = config.beta
        self.gamma = config.gamma
        self.loss_type = config.loss_type
        self.C_max = torch.Tensor([config.max_capacity])
        self.C_stop_iter = config.Capacity_max_iter

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the loss for BetaVAE.

        The loss function is a combination of the reconstruction loss and
        a regularization term that encourages the latent space to follow
        a Gaussian distribution.

        Parameters
        ----------
        *args : tuple
            Variable length argument list, containing:
            - recons : torch.Tensor
                The reconstructed input from the decoder.
            - input : torch.Tensor
                The original input.
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
                The total loss, combining reconstruction and KLD loss.
            - 'Reconstruction_Loss' : torch.Tensor
                The mean squared error between the input and the reconstruction.
            - 'KLD' : torch.Tensor
                The Kullback-Leibler divergence loss.
        """
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


def initialize() -> BetaVAE:
    """
    Initializes the BetaVAE model.

    Returns
    -------
    BetaVAE
        The BetaVAE model instance.
    """
    return BetaVAE
