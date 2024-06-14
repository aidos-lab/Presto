from dataclasses import dataclass
from typing import Any, Protocol

#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class GeneratedData(Protocol):
    generator: str
    num_samples: int = 1000
    seed: int = 42


@dataclass
class LocalData(Protocol):
    module: str
    seed: int = 42


@dataclass
class iris(GeneratedData):
    name: str = "Iris"
    generator: str = "load_iris"
    num_classes: int = 3


@dataclass
class digits(GeneratedData):
    name: str = "Iris"
    generator: str = "load_digits"
    num_classes: int = 3


@dataclass
class linnerud(GeneratedData):
    name: str = "Linnerud"
    generator: str = "load_linnerud"


@dataclass
class wine(GeneratedData):
    name: str = "Wine"
    generator: str = "load_wine"


@dataclass
class breast_cancer(GeneratedData):
    name: str = "Breast Cancer"
    generator: str = "load_breast_cancer"


@dataclass
class MNIST(LocalData):
    name: str = "MNIST"
    module: str = "lsd.generate.autoencoders.datasets.mnist"
    samples: int = 1000


#  ╭──────────────────────────────────────────────────────────╮
#  │ DimReduction Configurations                              │
#  ╰──────────────────────────────────────────────────────────╯
@dataclass
class Projector(Protocol):
    module: str = "lsd.generate.dim_reductions.models.projector"
    dim: int = 2
    metric: str = "euclidean"


@dataclass
class UMAP(Projector):
    name: str = "Uniform Manifold Approximation and Projection"
    module: str = "lsd.generate.dim_reductions.models.umap"
    nn: int = 15
    min_dist: float = 0.1
    init: str = "spectral"
    seed: int = 42


@dataclass
class tSNE(Projector):
    name: str = "t-Distributed Stochastic Neighbor Embedding"
    module: str = "lsd.generate.dim_reductions.models.tsne"
    perplexity: int = 30
    ee: float = 12.0
    seed: int = 42


@dataclass
class Phate(Projector):
    name: str = (
        "Potential of Heat-diffusion for Affinity-based Transition Embedding"
    )
    module: str = "lsd.generate.dim_reductions.models.phate"
    k: int = 5
    gamma: float = 1.0
    decay: float = 0.5
    t: Any = "auto"


@dataclass
class Isomap(Projector):
    name: str = "Isometric Mapping"
    module: str = "lsd.generate.dim_reductions.models.isomap"
    nn: int = 15


@dataclass
class LLE(Projector):
    name: str = "Locally Linear Embedding"
    module: str = "lsd.generate.dim_reductions.models.lle"
    nn: int = 5
    reg: float = 0.001
    eigen_solver: str = "auto"
    tol: float = 1e-6


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Thread:
    name: str = "Multi-threading"
    module: str = "lsd.generate.dim_reductions.implementations.thread"
    n_jobs: int = 1
