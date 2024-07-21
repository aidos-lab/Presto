from dataclasses import dataclass
from dataclasses import field
from typing import Protocol


#  ╭──────────────────────────────────────────────────────────╮
#  │ Model Configurations                                     │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Architecture(Protocol):
    """
    Base configuration protocol for autoencoder architectures in LSD.

    This protocol defines a standard structure for specifying the architecture of autoencoders, including essential attributes such as module paths, hidden dimensions, and latent dimensions. It serves as a template for all autoencoder configurations supported by the LSD framework.

    Attributes
    ----------
    module : str
        The module path to the autoencoder model.
    hidden_dims : list of int, optional
        A list specifying the number of neurons in each hidden layer.
        The default value is [8, 16], indicating two hidden layers with 8 and 16 neurons respectively.
    latent_dim : int, optional
        The dimensionality of the latent space. The default is 10, representing the number of dimensions
        in the latent representation of the data.

    Notes:
    -------
    LSD's base implementation of a VAE, found in `lsd.generate.autoencoders.models.vae`, requires a specific implementation of a `loss_function` method.
    """

    module: str
    hidden_dims: list[int] = field(default_factory=lambda: [8, 16])
    latent_dim: int = 10


@dataclass
class betaVAE(Architecture):
    """
    Configuration for the Beta Variational Autoencoder.

    This class extends the `Architecture` class and adds specific configurations for the Beta Variational Autoencoder developed by Higgins et al. (2017).

    Attributes
    ----------
    name : str
        The name of the autoencoder. Default is "Beta Variational Autoencoder".
    module : str
        The module path to the betaVAE model. Default points to our implementation in "lsd.generate.autoencoders.models.beta".
    beta : float, optional
        The weight of the KL divergence term. Higher values enforce a stricter bottleneck. Default is 2.
    gamma : float, optional
        Regularization term. Default is 1.
    loss_type : str, optional
        The type of loss to use. "B" typically stands for Beta-VAE loss. Default is "B".
    max_capacity : int, optional
        Maximum capacity of the latent space. This controls the amount of information passed through the bottleneck. Default is 25.
    Capacity_max_iter : float, optional
        The number of iterations over which capacity increases linearly. Default is 1e5.
    """

    name: str = "Beta Variational Autoencoder"
    module: str = "lsd.generate.autoencoders.models.beta"
    beta: float = 2
    gamma: float = 1
    loss_type: str = "B"
    max_capacity: int = 25
    Capacity_max_iter: float = 1e5


@dataclass
class infoVAE(Architecture):
    """
    Configuration for the Information Maximizing Variational Autoencoder.

    This class extends the `Architecture` class and adds specific configurations for the Information Maximizing Variational Autoencoder, developed by Zhao et al. (2017).

    Attributes
    ----------
    name : str
        The name of the autoencoder. Default is "Information Maximizing Variational Autoencoder".
    module : str
        The module path to the infoVAE model. Default points to our implementation in "lsd.generate.autoencoders.models.info".
    alpha : float, optional
        Weight of the adversarial loss term. Default is -0.5.
    beta : float, optional
        Weight of the mutual information term. Default is 5.0.
    kernel : str, optional
        The type of kernel to use in the Maximum Mean Discrepancy (MMD) term. Default is "imq" (Inverse Multiquadratic).
    reg_weight : float, optional
        Regularization weight for the MMD term. Default is 1.0.
    z_var : float, optional
        The variance of the latent space distribution. Default is 2.0.
    eps : float, optional
        Small epsilon value to prevent division by zero in computations. Default is 1e-7.
    """

    name: str = "Information Maximizing Variational Autoencoder"
    module: str = "lsd.generate.autoencoders.models.info"
    alpha: float = -0.5
    beta: float = 5.0
    kernel: str = "imq"
    reg_weight: float = 1.0
    z_var: float = 2.0
    eps: float = 1e-7


@dataclass
class WAE(Architecture):
    """
    Configuration for the Wasserstein Autoencoder.

    This class extends the `Architecture` class and adds specific configurations for the Wasserstein Autoencoder, developed by Tolstikhin et al. (2017).

    Attributes
    ----------
    name : str
        The name of the autoencoder. Default is "Wasserstein Autoencoder".
    module : str
        The module path to the WAE model. Default points to our implementation in "lsd.generate.autoencoders.models.wae".
    kernel : str, optional
        The type of kernel to use in the regularization term. Default is "imq" (Inverse Multiquadratic).
    reg_weight : float, optional
        Regularization weight for the kernel divergence term. Default is 100.
    z_var : float, optional
        The variance of the latent space distribution. Default is 2.0.
    eps : float, optional
        Small epsilon value to prevent division by zero in computations. Default is 1e-7.
    """

    name: str = "Wasserstein Autoencoder"
    module: str = "lsd.generate.autoencoders.models.wae"
    kernel: str = "imq"
    reg_weight: float = 100
    z_var: float = 2.0
    eps: float = 1e-7


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class DataModule(Protocol):
    """
    Base configuration protocol for data modules in LSD.

    This protocol provides a common structure for defining data modules, which handle data loading and preprocessing for various datasets used in autoencoder training.

    Attributes
    ----------
    module : str
        The module path to the dataset loader.
    num_workers : int, optional
        The number of subprocesses to use for data loading. The default is 4.
    batch_size : int, optional
        The number of samples per batch of data. The default is 64.
    pin_memory : bool, optional
        If True, the data loader will copy Tensors into CUDA pinned memory before returning them. The default is False.
    sample_size : float, optional
        Fraction of the dataset to use. The default is 1.0, indicating the entire dataset.
    train_test_split : list of float, optional
        A list defining the proportions for training and testing split. Default is an empty list, which uses a default split configuration.
    train_test_seed : int, optional
        Seed for determining train-test split. The default is 42.
    """

    module: str
    num_workers: int = 4
    batch_size: int = 64
    pin_memory: bool = False
    sample_size: float = 1.0
    train_test_split: list[float] = field(default_factory=list)
    train_test_seed: int = 42


@dataclass
class MNIST(DataModule):
    """
    Configuration for the MNIST dataset.

    This class extends `DataModule` to provide specific configurations for the MNIST dataset, which is commonly used for digit classification tasks.

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "MNIST".
    module : str
        The module path to the MNIST dataset loader. Default is "lsd.generate.autoencoders.datasets.mnist".
    in_channels : int, optional
        The number of channels in the input images. Default is 1, indicating grayscale images.
    img_size : int, optional
        The size of the input images (height and width). Default is 28 pixels.
    num_classes : int, optional
        The number of classes in the dataset. Default is 10, representing digits 0 through 9.
    """

    name: str = "MNIST"
    module: str = "lsd.generate.autoencoders.datasets.mnist"
    in_channels: int = 1
    img_size: int = 28
    num_classes: int = 10


@dataclass
class celebA(DataModule):
    """
    Configuration for the CelebA dataset.

    This class extends `DataModule` to provide specific configurations for the CelebA dataset, used for tasks such as image generation and face attribute recognition.

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "celebA".
    module : str
        The module path to the CelebA dataset loader. Default is "lsd.generate.autoencoders.datasets.celebA".
    in_channels : int, optional
        The number of channels in the input images. Default is 3, indicating RGB images.
    image_size : int, optional
        The size of the input images (height and width). Default is 64 pixels.
    num_classes : int, optional
        The number of attributes or classes in the dataset. Default is 40.
    """

    name: str = "celebA"
    module: str = "lsd.generate.autoencoders.datasets.celebA"
    in_channels: int = 3
    img_size: int = 64
    num_classes: int = 40


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Trainer(Protocol):
    """
    Base configuration protocol for training configurations in LSD.

    This protocol provides a common structure for defining training configurations for various optimizers used in training autoencoders. It includes parameters that are commonly required for training, such as learning rate and number of epochs.

    Attributes
    ----------
    module : str
        The module path to the base optimizer. The default is
        "lsd.generate.autoencoders.optimizers.base", which requires subclass implementation for specific optimizers.
    lr : float, optional
        Learning rate for the optimizer. The default is 0.001.
    weight_decay : float, optional
        Weight decay (L2 penalty) applied to the optimizer. The default is 0.0.
    epochs : int, optional
        Number of epochs for training. The default is 100.
    kld : float, optional
        The weight of the Kullback-Leibler divergence term in the loss function. The default is 0.0002.
    optimizer_idx : int, optional
        Index to identify the optimizer being used. The default is 0.
    clip_max_norm : float, optional
        Maximum norm for gradient clipping to prevent exploding gradients. The default is 1.0.
    seed : int, optional
        Seed for random number generation to ensure reproducibility. The default is 42.
    """

    module: str = "lsd.generate.autoencoders.optimizers.base"
    lr: float = 0.001
    weight_decay: float = 0.0
    epochs: int = 100
    kld: float = 0.0002
    optimizer_idx: int = 0
    clip_max_norm: float = 1.0
    seed: int = 42


@dataclass
class Adam(Trainer):
    """
    Configuration for the Adam optimizer.

    This class extends `Trainer` to provide specific configurations for the Adam optimizer, which is an adaptive learning rate optimization algorithm designed for training deep neural networks.

    Attributes
    ----------
    name : str
        The name of the optimizer. Default is "Adam". This much match the name of the optimizer in the PyTorch `torch.optim` module.
    module : str, optional
        The module path to the Adam optimizer. Default is
        "lsd.generate.autoencoders.optimizers.base".
    betas : tuple of float, optional
        Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
    eps : float, optional
        Term added to the denominator to improve numerical stability. Default is 1e-8.
    """

    name: str = "Adam"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SGD(Trainer):
    """
    Configuration for the Stochastic Gradient Descent (SGD) optimizer.

    This class extends `Trainer` to provide specific configurations for the SGD optimizer, which is a traditional optimization algorithm used extensively in training machine learning models.

    Attributes
    ----------
    name : str
        The name of the optimizer. Default is "SGD". This much match the name of the optimizer in the PyTorch `torch.optim` module.
    module : str, optional
        The module path to the SGD optimizer. Default is
        "lsd.generate.autoencoders.optimizers.base".
    momentum : float, optional
        Momentum factor for SGD. The default is 0.9.
    nesterov : bool, optional
        Whether to enable Nesterov momentum. The default is False.
    """

    name: str = "SGD"
    momentum: float = 0.9
    nesterov: bool = False
