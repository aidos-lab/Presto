from dataclasses import dataclass
from typing import Any, Protocol

#  ╭──────────────────────────────────────────────────────────╮
#  │ Multiverse Configuration                                 │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Multiverse(Protocol):
    """
    A protocol for defining a multiverse of configurations for latent space generation. This is a base protocol for configuring multiverses using Presto's latent space design (LSD) framework.

    Attributes
    ----------
    base : str
        Class name of the base generator. This parameter is used to unpack all relevant parameters under this identifier to create a multiverse of configurations.
    data_choices : str
        Path to a YAML file containing the data choices. This file specifies various data options that the multiverse can utilize for model training and evaluation.
    model_choices : str
        Path to a YAML file with model choices. This file includes different model architectures or configurations that the multiverse can explore.
    implementation_choices : str
        Path to a YAML file detailing implementation choices. This file lists various implementation options, such as different algorithms or techniques that can be employed within the multiverse framework.
    module : str
        Path to the module containing the generator class or functions. This module is used to instantiate and manage the different configurations defined by the multiverse.
    """

    base: str
    data_choices: str
    model_choices: str
    implementation_choices: str
    module: str


#  ╭──────────────────────────────────────────────────────────╮
#  │ Universe Configuration                                   │
#  ╰──────────────────────────────────────────────────────────╯
@dataclass(frozen=True)
class Universe:
    """
    Represents a single configuration or 'universe' within a multiverse of model
    configurations. Each universe generates a unique latent space, and multiple
    universes together form a multiverse, allowing for the exploration of various latent spaces.

    Attributes
    ----------
    data_choices : Any
        Choices or configurations related to the data used in this universe. This attribute defines the dataset or data-related parameters that impact the model's performance and behavior within this universe.
    model_choices : Any
        Model-related choices or configurations for this universe. This includes
        the architecture, hyperparameters, and other model-specific settings that are unique to this particular universe.
    implementation_choices : Any
        Implementation choices for this universe. This attribute specifies different algorithms, methods, or strategies used to implement the model within this universe, impacting how the model processes and learns from the data.

    Notes
    -----
    - A multiverse is composed of many such universes. Each universe explores a
      different configuration or combination of data, model, and implementation
      choices to produce a unique latent space.
    - By examining multiple latent spaces, the multiverse provides a comprehensive
      view of potential model behaviors and outcomes, facilitating a more robust
      analysis and comparison of different configurations.
    """

    data_choices: Any
    model_choices: Any
    implementation_choices: Any


#  ╭──────────────────────────────────────────────────────────╮
#  │ Presto Multiverses                                       │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class AutoencoderMultiverse(Multiverse):
    """
    A multiverse configuration class for Variational Autoencoder models. Attributes are inherited from the base `Multiverse` protocol.
    """

    base: str = "Autoencoder"
    module: str = "lsd.generate.autoencoders"
    data_choices: str = "lsd/design/ae.yaml"
    model_choices: str = "lsd/design/ae.yaml"
    implementation_choices: str = "/design/ae.yaml"


@dataclass
class DimReductionMultiverse(Multiverse):
    """
    A multiverse configuration class for dimensionality reduction models.
    Attributes are inherited from the base `Multiverse` protocol.
    """

    base: str = "DimReduction"
    module: str = "lsd.generate.dim_reductions"
    model_choices: str = "lsd/design/dr.yaml"
    data_choices: str = "lsd/design/dr.yaml"
    implementation_choices: str = "/design/dr.yaml"


@dataclass
class TransformerMultiverse(Multiverse):
    """
    A multiverse configuration class for transformer models. Attributes are inherited from the base `Multiverse` protocol.

    Notes
    -----
    This class is not yet implemented but will be coming soon. Future releases will include support for various transformer architectures and configurations.
    """

    base: str = "Transformer"
    module: str = "lsd.generate.transformers"
    model_choices: str = "lsd/design/tf.yaml"
    data_choices: str = "lsd/design/tf.yaml"
    implementation_choices: str = "/design/tf.yaml"
