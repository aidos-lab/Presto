import functools
import operator
from abc import abstractmethod

import torch
from torch import Tensor, nn
from typing import Any, List

from lsd.utils import ConfigType


class BaseVAE(nn.Module):
    """
    Base Variational Autoencoder (VAE) Class, inspired by AntixK's
    PyTorch-VAE implementations.

    This class implements the fundamental structure of a variational autoencoder (VAE), including both encoder and decoder networks, as well as essential VAE operations like reparameterization and loss function calculation.

    Attributes
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    img_size : int
        Size of the input image (assumed to be square).
    in_channels : int
        Number of input image channels (e.g., 1 for grayscale, 3 for RGB).
    hidden_dims : List[int]
        List containing the dimensions of hidden layers in the encoder and decoder.

    Abstract Methods
    ----------------
    loss_function(*inputs: Any, **kwargs) -> Tensor
        Abstract method for calculating the VAE loss. Must be implemented in a subclass.

    Methods
    -------
    encode(input: Tensor) -> List[Tensor]
        Encodes the input and returns the latent mean and log variance.
    decode(input: Tensor) -> Tensor
        Decodes the latent codes to the image space.
    sample(num_samples: int, current_device: int, **kwargs) -> Tensor
        Samples from the latent space and returns the corresponding images.
    generate(x: Tensor, **kwargs) -> Tensor
        Reconstructs the input image from the latent space.
    latent(input: Tensor, **kwargs) -> Tensor
        Encodes the input to the latent space.
    forward(input: Tensor, **kwargs) -> List[Tensor]
        Full forward pass through the VAE, from input to reconstruction.

    Static Methods
    --------------
    build_encoder(in_channels: int, img_size: int, hidden_dims: List[int], latent_dim: int)
        Builds the encoder network.
    build_decoder(latent_dim: int, num_features: int, hidden_dims: List[int], in_channels: int)
        Builds the decoder network.
    reparameterize(mu: Tensor, logvar: Tensor) -> Tensor
        Performs the reparameterization trick to sample from the latent space.
    compute_mmd(z: Tensor, kernel_type: str, reg_weight: float, z_var: float, eps: float) -> Tensor
        Computes the Maximum Mean Discrepancy (MMD) between the latent codes and the prior.
    compute_kernel(x1: Tensor, x2: Tensor, kernel_type: str, z_var: float, eps: float) -> Tensor
        Computes the specified kernel between two sets of latent codes.

    Helper Functions
    --------------
    _compute_rbf(x1: Tensor, x2: Tensor, z_var: float) -> Tensor
        Computes the Radial Basis Function (RBF) kernel.
    _compute_imq(x1: Tensor, x2: Tensor, z_var: float, eps: float) -> Tensor
        Computes the Inverse Multi-Quadratic (IMQ) kernel.
    """

    def __init__(self, config: ConfigType):
        """
        Initializes the BaseVAE class with the given configuration.

        Parameters
        ----------
        config : ConfigType
            Configuration object containing attributes such as `latent_dim`, `img_size`,
            `in_channels`, and `hidden_dims`.
        """
        super(BaseVAE, self).__init__()

        self.latent_dim = config.latent_dim
        self.img_size = config.img_size
        self.in_channels = config.in_channels
        self.hidden_dims = [config.in_channels] + config.hidden_dims

        assert len(self.hidden_dims) > 0, "No hidden layers specified"

        # Encoder
        (
            self.encoder,  # Encoder Layers
            self.fc_mu,  # Linear layer for mean
            self.fc_var,  # Linear layer for variance
            self.encoded_shape,  # Pre-latent shape
            self.num_features,  # Pre-latent dimension
        ) = self.build_encoder(
            latent_dim=self.latent_dim,
            img_size=self.img_size,
            hidden_dims=self.hidden_dims,
            in_channels=self.in_channels,
        )

        # Decoder
        (
            self.decoder,  # Decoder Layers
            self.fc_decoder_input,  # Linear layer latent -> decoder Input
            self.final_layer,  # Final layer
        ) = self.build_decoder(
            latent_dim=self.latent_dim,
            num_features=self.num_features,
            hidden_dims=self.hidden_dims,
            in_channels=self.in_channels,
        )

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Model Specific Methods                                   │
    #  ╰──────────────────────────────────────────────────────────╯
    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        """
        Abstract method for computing the VAE loss.

        Must be implemented in a subclass.

        Parameters
        ----------
        *inputs : Any
            Variable length argument list containing the necessary inputs for the loss computation.
        **kwargs : Any
            Additional keyword arguments for the loss computation.

        Returns
        -------
        Tensor
            Computed loss value.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ General VAE Methods                                      │
    #  ╰──────────────────────────────────────────────────────────╯
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing it through the encoder network
        and returns the latent mean and log variance.

        Parameters
        ----------
        input : Tensor
            Input tensor to the encoder with shape [BatchSize, Channels, Height, Width].

        Returns
        -------
        List[Tensor]
            A list containing the mean (mu) and log variance (log_var) of the latent Gaussian distribution.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, input: Tensor) -> Any:
        """
        Decodes the latent codes to the image space.

        Parameters
        ----------
        input : Tensor
            Latent codes with shape [BatchSize, LatentDim].

        Returns
        -------
        Tensor
            Decoded image with shape [BatchSize, Channels, Height, Width].
        """
        result = self.fc_decoder_input(input)
        result = result.view(-1, *self.encoded_shape)
        result = self.decoder(result)

        result = self.final_layer(result)
        return result

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding
        image space maps.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        current_device : int
            Device on which to perform the sampling.

        Returns
        -------
        Tensor
            Sampled images with shape [num_samples, Channels, Height, Width].
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.

        Parameters
        ----------
        x : Tensor
            Input image tensor with shape [BatchSize, Channels, Height, Width].

        Returns
        -------
        Tensor
            Reconstructed image with shape [BatchSize, Channels, Height, Width].
        """

        return self.forward(x)[0]

    def latent(self, input: Tensor, **kwargs) -> Tensor:
        """
        Encodes the input to the latent space.

        Parameters
        ----------
        input : Tensor
            Input tensor with shape [BatchSize, Channels, Height, Width].

        Returns
        -------
        Tensor
            Latent space representation with shape [BatchSize, LatentDim].
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Full forward pass through the VAE, from input to reconstruction.

        Parameters
        ----------
        input : Tensor
            Input tensor with shape [BatchSize, Channels, Height, Width].

        Returns
        -------
        List[Tensor]
            A list containing the reconstructed image, input, latent vector, mean, and log variance.
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, z, mu, log_var]

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Static Methods                                           │
    #  ╰──────────────────────────────────────────────────────────╯
    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        Parameters
        ----------
        mu : Tensor
            Mean of the latent Gaussian with shape [B, D].
        logvar : Tensor
            Log variance of the latent Gaussian with shape [B, D].

        Returns
        -------
        Tensor
            Sampled latent vector with shape [B, D].
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def build_encoder(
        in_channels,
        img_size,
        hidden_dims,
        latent_dim,
    ):
        """
        Flexibly builds an encoder network given different data & modeling choices!

        Parameters
        ----------
        in_channels : int
            Number of input image channels.
        img_size : int
            Size of the input image.
        hidden_dims : List[int]
            List of dimensions for hidden layers.
        latent_dim : int
            Dimensionality of the latent space.

        Returns
        -------
        encoder : nn.Sequential
            Encoder network.
        fc_mu : nn.Linear
            Linear layer for calculating the mean.
        fc_var : nn.Linear
            Linear layer for calculating the log variance.

        encoded_shape : torch.Size
            Shape of the encoded tensor.
        num_features : int
            Number of features in the encoded tensor.
        """
        modules = []
        for idx in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[idx],
                        hidden_dims[idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[idx + 1]),
                    nn.LeakyReLU(),
                )
            )

        encoder = nn.Sequential(*modules)

        # Tracking Encoder Shapes
        encoded_shape = encoder(
            torch.rand(1, in_channels, img_size, img_size)
        ).shape[1:]
        num_features = functools.reduce(
            operator.mul,
            list(encoded_shape),
        )

        # VAE Linear Layers
        fc_mu = nn.Linear(num_features, latent_dim)
        fc_var = nn.Linear(num_features, latent_dim)

        return encoder, fc_mu, fc_var, encoded_shape, num_features

    @staticmethod
    def build_decoder(
        latent_dim,
        num_features,
        hidden_dims,
        in_channels,
    ):
        """
        Flexibly builds a decoder network given different data & modeling choices!


        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent space.
        num_features : int
            Number of features in the flattened encoder output.
        hidden_dims : List[int]
            List of dimensions for hidden layers.
        in_channels : int
            Number of input image channels.

        Returns
        -------
        decoder : nn.Sequential
            Decoder network as a sequential model.
        fc_decoder_input : nn.Linear
            Linear layer for mapping latent space to the decoder input.
        final_layer : nn.Sequential
            Final layer for output reconstruction.
        """
        # Build Decoder
        modules = []

        fc_decoder_input = nn.Linear(latent_dim, num_features)

        rhidden_dims = hidden_dims[::-1]

        for i in range(len(rhidden_dims) - 1):
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    rhidden_dims[i],
                    rhidden_dims[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(rhidden_dims[i + 1]),
                nn.LeakyReLU(),
            )
            modules.append(layer)

        decoder = nn.Sequential(*modules)

        # FINAL LAYER
        final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                rhidden_dims[-1],
                rhidden_dims[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(rhidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                rhidden_dims[-1],
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

        return decoder, fc_decoder_input, final_layer

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ MMD & Kernel Methods                                     │
    #  ╰──────────────────────────────────────────────────────────╯

    @staticmethod
    def compute_mmd(
        z: Tensor,
        kernel_type: str,
        reg_weight: float,
        z_var: float,
        eps: float,
    ) -> Tensor:
        """
        Computes the Maximum Mean Discrepancy (MMD) between the latent codes and the prior.

        Parameters
        ----------
        z : Tensor
            Latent codes with shape [NumSamples, LatentDim].
        kernel_type : str
            Type of kernel to use ('rbf' or 'imq').
        reg_weight : float
            Regularization weight for MMD.
        z_var : float
            Variance of the latent space.
        eps : float
            Small constant to avoid division by zero in kernel computation.

        Returns
        -------
        Tensor
            Computed MMD value.
        """
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = BaseVAE.compute_kernel(
            prior_z,
            prior_z,
            kernel_type,
            z_var,
            eps,
        )
        z__kernel = BaseVAE.compute_kernel(
            z,
            z,
            kernel_type,
            z_var,
            eps,
        )
        priorz_z__kernel = BaseVAE.compute_kernel(
            prior_z,
            z,
            kernel_type,
            z_var,
            eps,
        )

        mmd = (
            reg_weight * prior_z__kernel.mean()
            + reg_weight * z__kernel.mean()
            - 2 * reg_weight * priorz_z__kernel.mean()
        )
        return mmd

    @staticmethod
    def compute_kernel(
        x1: Tensor,
        x2: Tensor,
        kernel_type: str,
        z_var: float,
        eps: float,
    ) -> Tensor:
        """
        Computes the specified kernel between two sets of latent codes.

        Parameters
        ----------
        x1 : Tensor
            First set of latent codes with shape [NumSamples, LatentDim].
        x2 : Tensor
            Second set of latent codes with shape [NumSamples, LatentDim].
        kernel_type : str
            Type of kernel to use ('rbf' or 'imq').
        z_var : float
            Variance of the latent space.
        eps : float
            Small constant to avoid division by zero in kernel computation.

        Returns
        -------
        result : Tensor
            Computed kernel matrix.
        """
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if kernel_type == "rbf":
            result = BaseVAE._compute_rbf(
                x1,
                x2,
                z_var,
            )
        elif kernel_type == "imq":
            result = BaseVAE._compute_imq(
                x1,
                x2,
                z_var,
                eps,
            )
        else:
            raise ValueError("Undefined kernel type.")

        return result

    @staticmethod
    def _compute_rbf(
        x1: Tensor,
        x2: Tensor,
        z_var: float,
    ) -> Tensor:
        """
        Computes the Radial Basis Function (RBF) kernel between x1 and x2.

        Parameters
        ----------
        x1 : Tensor
            First set of latent codes with shape [N, N, D].
        x2 : Tensor
            Second set of latent codes with shape [N, N, D].
        z_var : float
            Variance of the latent space.

        Returns
        -------
        Tensor
            Computed RBF kernel matrix.
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    @staticmethod
    def _compute_imq(
        x1: Tensor,
        x2: Tensor,
        z_var: float,
        eps: float,
    ) -> Tensor:
        """
        Computes the Inverse Multi-Quadratic (IMQ) kernel between x1 and x2.

        Given by k(x_1, x_2) = sum [C / (C + ||x_1 - x_2||^2)].

        Parameters
        ----------
        x1 : Tensor
            First set of latent codes with shape [N, N, D].
        x2 : Tensor
            Second set of latent codes with shape [N, N, D].
        z_var : float
            Variance of the latent space.
        eps : float
            Small constant to avoid division by zero.

        Returns
        -------
        Tensor
            Computed IMQ kernel matrix.
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()
        return result
