from dataclasses import dataclass

#  ╭──────────────────────────────────────────────────────────╮
#  │ Multiverse Configurations                                │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Guide:
    id_: str = "Unpack all parameters under this ID to create multiverse"
    data_choices: str = "A path to a yaml file containing data choices."
    model_choices: str = "A path to yaml file containing model choices"
    implementation_choices: str = "/design/implementation/ae.yaml"
    base: str = "lsd.generate.autoencoders.ae"
    generators: str = "lsd.generate.autoencoders.models"


@dataclass
class AutoencoderMultiverse:
    id_: str = "Autoencoder"
    data_choices: str = "lsd/design/data/ae.yaml"
    model_choices: str = "lsd/design/model/ae.yaml"
    implementation_choices: str = "/design/implementation/ae.yaml"
    base: str = "lsd.generate.autoencoders.ae"
    generators: str = "lsd.generate.autoencoders.models"


@dataclass
class DimReductionMultiverse:
    id_: str = "DimReduction"
    model_choices: str = "lsd/design/model/dr.yaml"
    data_choices: str = "lsd/design/data/dr.yaml"
    implementation_choices: str = "/design/implementation/dr.yaml"
    base: str = "lsd.generate.dim_reduction.dr"
    generators: str = "lsd.generate.dim_reductions.models"


@dataclass
class TransformerMultiverse:
    id_: str = "Transformer"
    model_choices: str = "lsd/design/model/tf.yaml"
    data_choices: str = "lsd/design/data/tf.yaml"
    implementation_choices: str = "/design/implementation/tf.yaml"
    base: str = "lsd.generate.transformers.tf"
    generators: str = "lsd.generate.transformers.models"


# Development Stub
@dataclass
class CustomMultiverse:
    pass
