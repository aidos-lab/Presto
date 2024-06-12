from dataclasses import dataclass
from dataclasses import field

#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class celebA:
    name: str = "celebA"
    module: str = "lsd.generate.autoencoders.datasets.celebA"
    batch_size: int = 128
    train_test_split: list[float] = field(default_factory=list)
    in_channels: int = 3
    image_size: int = 64
    num_classes: int = 40


@dataclass
class MNIST:
    name: str = "MNIST"
    module: str = "lsd.generate.autoencoders.datasets.mnist"
    batch_size: int = 64
    train_test_split: list[float] = field(default_factory=list)
    in_channels: int = 1
    img_size: int = 28
    num_classes: int = 10


#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯
@dataclass
class betaVAE:
    name: str = "Beta Variational Autoencoder"
    module: str = "lsd.generate.autoencoders.models.beta"
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=list)
    beta: float = 2
    gamma: float = 1
    loss: str = "B"


@dataclass
class infoVAE:
    name: str = "Information Maximizing Variational Autoencoder"
    module: str = "lsd.generate.autoencoders.models.info"
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=list)
    alpha: float = 0.01
    beta: float = 100
    kernel: str = "rbf"


@dataclass
class WAE:
    name: str = "Wasserstein Autoencoder"
    module: str = "lsd.generate.autoencoders.models.info"
    latent_dim: int = 10
    hidden_dims: list[int] = field(default_factory=list)
    lambda_: float = 0.01
    kernel: str = "rbf"
    kernel_width: int = 100


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Adam:
    name: str = "Adam"
    module: str = "lsd.generate.autoencoders.optimizers.adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    epochs: int = 100
