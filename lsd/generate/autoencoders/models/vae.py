"Base Variational Autoencoder Class. Inspired by AntixK's PyTorch-VAE implementations."

import functools
import operator
from abc import abstractmethod

import torch
from torch import Tensor, nn

from typing import Any, List


class BaseVAE(nn.Module):
    def __init__(self, config):
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
        pass

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ General VAE Methods                                      │
    #  ╰──────────────────────────────────────────────────────────╯
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
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
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.fc_decoder_input(input)
        result = result.view(-1, *self.encoded_shape)
        result = self.decoder(result)

        result = self.final_layer(result)
        return result

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def latent(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, z, mu, log_var]

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Static Methods                                           │
    #  ╰──────────────────────────────────────────────────────────╯
    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
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
        modules = []
        # Build Encoder Architechture
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
        z: torch.tensor,
        kernel_type: str,
        reg_weight: float,
        z_var: float,
        eps: float,
    ) -> torch.tensor:
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
        x1: torch.tensor,
        x2: torch.tensor,
        kernel_type: str,
        z_var: float,
        eps: float,
    ) -> torch.tensor:
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
        x1: torch.tensor,
        x2: torch.tensor,
        z_var: float,
    ) -> torch.tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (torch.tensor)
        :param x2: (torch.tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    @staticmethod
    def _compute_imq(
        x1: torch.tensor,
        x2: torch.tensor,
        z_var: float,
        eps: float,
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
        C = 2 * z_dim * z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()
        return result
