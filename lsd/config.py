from dataclasses import dataclass
from typing import Any

#  ╭──────────────────────────────────────────────────────────╮
#  │ Multiverse Configurations                                │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass(frozen=True)
class Universe:
    data_choices: Any
    model_choices: Any
    implementation_choices: Any


@dataclass
class Guide:
    base: str = (
        "Class name of the base generator: unpacks all parameters under this ID to create multiverse."
    )
    data_choices: str = "Path to yaml file with data choices."
    model_choices: str = "Path to yaml file with model choices"
    implementation_choices: str = (
        "Path to yaml file with implementation choices"
    )
    module: str = "Path to module Generator Module"


@dataclass
class AutoencoderMultiverse:
    base: str = "Autoencoder"
    module: str = "lsd.generate.autoencoders"
    data_choices: str = "lsd/design/data/ae.yaml"
    model_choices: str = "lsd/design/model/ae.yaml"
    implementation_choices: str = "/design/implementation/ae.yaml"


@dataclass
class DimReductionMultiverse:
    base: str = "DimReduction"
    module: str = "lsd.generate.dim_reductions"
    model_choices: str = "lsd/design/model/dr.yaml"
    data_choices: str = "lsd/design/data/dr.yaml"
    implementation_choices: str = "/design/implementation/dr.yaml"


@dataclass
class TransformerMultiverse:
    base: str = "Transformer"
    module: str = "lsd.generate.transformers"
    model_choices: str = "lsd/design/model/tf.yaml"
    data_choices: str = "lsd/design/data/tf.yaml"
    implementation_choices: str = "/design/implementation/tf.yaml"


# Development Stub
@dataclass
class CustomMultiverse:
    base: str = "Custom"
    module: str = "lsd.generate.transformers"
    model_choices: str = str()
    data_choices: str = str()
    implementation_choices: str = str()
