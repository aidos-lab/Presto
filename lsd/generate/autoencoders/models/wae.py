from torch.nn import functional as F

from lsd.generate.autoencoders.models.vae import BaseVAE


class WAE(BaseVAE):
    """
    Wasserstein Autoencoder (WAE).

    This class extends the `BaseVAE` class with a loss function that computes the Wasserstein Autoencoder loss as described in https://arxiv.org/pdf/1711.01558.

    Parameters
    ----------
    config : Config
        Configuration object containing hyperparameters and model settings.

    Attributes
    ----------
    kernel_type : str
        Type of kernel used in Maximum Mean Discrepancy (MMD) computation.
    reg_weight : float
        Regularization weight for the MMD term.
    z_var : float
        Variance of the latent variable's Gaussian distribution.
    eps : float
        Small value to avoid division by zero in MMD computation.

    Methods
    -------
    loss_function(*args, **kwargs) -> dict
        Computes the loss for WAE.
    """

    def __init__(
        self,
        config,
    ) -> None:
        """
        Constructor for the WAE class.

        Parameters
        ----------
        config : Config
            Configuration object containing hyperparameters and model settings.
        """
        super(WAE, self).__init__(config)

        self.kernel_type = config.kernel
        self.reg_weight = config.reg_weight
        self.z_var = config.z_var
        self.eps = config.eps

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the loss for WAE.

        The loss function combines the reconstruction loss and
        a Maximum Mean Discrepancy (MMD) term to regularize the
        latent space.

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
        **kwargs : dict
            Arbitrary keyword arguments. Not used in this function.

        Returns
        -------
        dict
            A dictionary containing the computed losses:
            - 'loss' : torch.Tensor
                The total loss, combining reconstruction and MMD losses.
            - 'Reconstruction_Loss' : torch.Tensor
                The mean squared error between the input and the reconstruction.
            - 'MMD' : torch.Tensor
                The Maximum Mean Discrepancy loss.
        """
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss = F.mse_loss(recons, input)

        mmd_loss = self.compute_mmd(
            z,
            self.kernel_type,
            reg_weight,
            self.z_var,
            self.eps,
        )

        loss = recons_loss + mmd_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "MMD": mmd_loss,
        }


def initialize() -> WAE:
    """
    Initializes the WAE model.

    Returns
    -------
    WAE
        The WAE model instance.
    """
    return WAE
