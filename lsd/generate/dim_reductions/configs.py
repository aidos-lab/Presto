from dataclasses import dataclass


@dataclass
class UMAP:
    name: str = "Uniform Manifold Approximation and Projection"
    module: str = "lsd.generate.dim_reductions.models.umap"
    nn: int = 15
    min_dist: float = 0.1


@dataclass
class tSNE:
    name: str = "t-Distributed Stochastic Neighbor Embedding"
    module: str = "lsd.generate.dim_reductions.models.tsne"
    perplexity: int = 30
    ee: float = 12.0


@dataclass
class Phate:
    name: str = (
        "Potential of Heat-diffusion for Affinity-based Transition Embedding"
    )
    module: str = "lsd.generate.dim_reductions.models.phate"
    k: int = 5
    gamma: float = 1.0
    knn_dist: str = "euclidean"


@dataclass
class MNIST:
    name: str = "MNIST"
    module: str = "lsd.generate.dim_reductions.datasets.mnist"
    samples: int = 1000
    in_channels: int = 1
    image_size: int = 28
    num_classes: int = 10


@dataclass
class Thread:
    name: str = "Multi-threading"
    module: str = "lsd.generate.dim_reductions.implementations.thread"
    n_jobs: int = 1
