"Config Objects."

from dataclasses import dataclass, field
from typing import Any, Protocol

import utils


@dataclass(frozen=True)
class Config:
    meta: Any
    data_params: Any
    model_params: Any
    trainer_params: Any


#  ╭──────────────────────────────────────────────────────────╮
#  │ Meta Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Meta:
    name: str
    id: int
    description: str
    project: str = "Presto"
    tags: list[str] = field(default_factory=list)


#  ╭──────────────────────────────────────────────────────────╮
#  │ Generator Configurations                                 │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class DimReductionConfig:
    name: str = "Dim Reduction"
    module: str = "dim_reduction.dr"
    seed: int = 42


@dataclass
class AutoEncoderConfig:
    name: str = "Autoencoder"
    module: str = "autoencoders.ae"
    seed: int = 42


@dataclass
class TransformerConfig:
    name: str = "Transformer"
    module: str = "transformers.tf"
    seed: int = 42


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


# Defaults
@dataclass
class DataModuleConfig(Protocol):
    module: str
    data_dir: str = f"{utils.project_root_dir()}" + "data/"
    num_workers: int = 8
    batch_size: int = 64
    pin_memory: bool = False
    sample_size: float = None
    seed: int = 42
