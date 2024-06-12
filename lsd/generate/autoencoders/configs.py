from dataclasses import dataclass
from dataclasses import field
from typing import Protocol

#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class DataModule(Protocol):
    module: str
    num_workers: int = 4
    batch_size: int = 64
    pin_memory: bool = False
    sample_size: float = 1.0
    seed: int = 42


@dataclass
class celebA(DataModule):
    name: str = "celebA"
    module: str = "lsd.generate.autoencoders.datasets.celebA"
    batch_size: int = 128
    train_test_split: list[float] = field(default_factory=list)
    in_channels: int = 3
    image_size: int = 64
    num_classes: int = 40


@dataclass
class MNIST(DataModule):
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
class Architecture(Protocol):
    module: str
    hidden_dims: list[int] = field(default_factory=lambda: [8, 16])
    latent_dim: int = 10


@dataclass
class betaVAE(Architecture):
    name: str = "Beta Variational Autoencoder"
    module: str = "lsd.generate.autoencoders.models.beta"
    beta: float = 2
    gamma: float = 1
    loss_type: str = "B"
    max_capacity: int = 25
    Capacity_max_iter: float = 1e5


@dataclass
class infoVAE(Architecture):
    name: str = "Information Maximizing Variational Autoencoder"
    module: str = "lsd.generate.autoencoders.models.info"
    alpha: float = 0.01
    beta: float = 100
    kernel: str = "rbf"


@dataclass
class WAE:
    name: str = "Wasserstein Autoencoder"
    module: str = "lsd.generate.autoencoders.models.info"
    lambda_: float = 0.01
    kernel: str = "rbf"
    kernel_width: int = 100


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Trainer(Protocol):
    module: str
    lr: float = 0.001
    weight_decay: float = 0.0
    epochs: int = 100
    clip_max_norm: float = 1.0


@dataclass
class Adam(Trainer):
    name: str = "Adam"
    module: str = "lsd.generate.autoencoders.optimizers.adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SGD(Trainer):
    name: str = "SGD"
    module: str = "lsd.generate.autoencoders.optimizers.sgd"
    lr: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False