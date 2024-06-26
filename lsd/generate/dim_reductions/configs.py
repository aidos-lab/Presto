from dataclasses import dataclass
from typing import Protocol, Optional, Union


#  ╭──────────────────────────────────────────────────────────╮
#  │ DimReduction Configurations                              │
#  ╰──────────────────────────────────────────────────────────╯
@dataclass
class Projector(Protocol):
    """
    Protocol Dimensionality Reduction Configurations.

    These attributes are common to all dimensionality reduction techniques.

    Attributes
    -----------
    module : str
        The module containing the projector implementation.
    n_components : int
        The target dimension of the projection. Also referenced as n_components in many libraries.
    metric: str
        The distance metric used for the projection.
    p: int or None
        The power parameter for the Minkowski metric.
    seed : int
        The random seed for reproducibility.
    max_ambient_dim: int or none
        The maximum ambient dimension of the data. If data has more dimensions than this, it will be reduced to this value using PCA.
    """

    module: str = "lsd.generate.dim_reductions.models.projector"
    n_components: int = 2
    metric: str = "euclidean"
    p: int = 1
    seed: int = 42
    max_ambient_dim: Optional[int] = 50


@dataclass
class UMAP(Projector):
    """
    Uniform Manifold Approximation and Projection (UMAP) Configuration.

    See https://umap-learn.readthedocs.io/en/latest/ for more information.

    Key Attributes
    -----------------
    n_neighbors : int
        The number of nearest neighbors used for the projection.
    min_dist : float
        The minimum distance between points in the latent space.
    init : str
        The initialization method for the initializing points in the latent space.

    Additional Attributes
    ---------------------
    n_epochs : int, optional (default: None)
        The number of training epochs to be used in optimizing the embedding.
    learning_rate : float, optional (default: 1.0)
        The initial learning rate for the embedding optimization.
    spread : float, optional (default: 1.0)
        The effective scale of embedded points.
    set_op_mix_ratio : float, optional (default: 1.0)
        Interpolate between (fuzzy) union and intersection set operations.
    local_connectivity : int, optional (default: 1)
        The local connectivity required.
    repulsion_strength : float, optional (default: 1.0)
        Weighting applied to negative samples in low dimensional embedding optimization.
    negative_sample_rate : int, optional (default: 5)
        The number of negative samples to select per positive sample in optimization.
    transform_queue_size : float, optional (default: 1.0)
        This scales the data transform queue size.
    a : float, optional (default: None)
        Early exaggeration parameter.
    b : float, optional (default: None)
        The final value of the early exaggeration parameter.
    random_state : int or RandomState, optional (default: None)
        Determines the random number generation for reproducibility.
    verbose : bool, optional (default: False)
        Controls verbosity of the output.

    """

    # Key Attributes
    name: str = "Uniform Manifold Approximation and Projection"
    module: str = "lsd.generate.dim_reductions.models.umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    init: str = "spectral"
    # Additional Attributes
    n_epochs: Optional[int] = None
    learning_rate: float = 1.0
    spread: float = 1.0
    set_op_mix_ratio: float = 1.0
    local_connectivity: int = 1
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    transform_queue_size: float = 1.0
    a: Optional[float] = None
    b: Optional[float] = None
    random_state: Optional[int] = None
    verbose: bool = False


@dataclass
class tSNE(Projector):
    """
    T-distributed Stochastic Neighbor Embedding (t-SNE) Configuration.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    for more information.

    Key Attributes
    -----------------
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that is used in other
        manifold learning algorithms. Larger datasets usually require a larger perplexity.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in the embedded space and
        how much space will be between them.
    init : str or ndarray of shape (n_samples, n_components), default='pca'
        Initialization of embedding.


    Additional Attributes
    ---------------------

    learning_rate : float or 'auto', default='auto'
        The learning rate for t-SNE optimization. Use 'auto' to set it automatically based on
        the sample size.
    max_iter : int, default=1000
        Maximum number of iterations for the optimization.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the optimization.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will be stopped.
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances in a feature array.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator.
    method : {'barnes_hut', 'exact'}, default='barnes_hut'
        The method used to approximate the gradient during optimization.
    angle : float, default=0.5
        Trade-off between speed and accuracy for Barnes-Hut T-SNE.
    verbose : int, default=0
        Verbosity level.

    """

    # Key Attributes
    name: str = "t-Distributed Stochastic Neighbor Embedding"
    module: str = "lsd.generate.dim_reductions.models.tsne"
    perplexity: float = 30.0
    early_exaggeration: float = 12.0
    init: str = "pca"
    # Additional Attributes
    learning_rate: Union[float, str] = "auto"
    max_iter: int = 1000
    n_iter_without_progress: int = 300
    min_grad_norm: float = 1e-7
    random_state: Optional[int] = None
    method: str = "barnes_hut"
    angle: float = 0.5
    verbose: int = 0


@dataclass
class Isomap(Projector):
    """
    Isometric Mapping (Isomap) Configuration.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    for more information.

    Key Attributes
    -----------------
    n_neighbors : int or None, default=5
        Number of neighbors to consider for each point.
    radius : float or None, default=None
        Limiting distance of neighbors to return.


    Additional Attributes
    ---------------------
    eigen_solver : str, default='auto'
        Solver to use for eigenvalue decomposition.
    tol : float, default=0
        Tolerance for eigenvalue decomposition.
    max_iter : int or None, default=None
        Maximum number of iterations for eigenvalue decomposition.
    path_method : str, default='auto'
        Method to use in finding shortest paths.
    neighbors_algorithm : str, default='auto'
        Algorithm to use for nearest neighbors search.
    n_jobs : int or None, default=None
        Number of parallel jobs to run for neighbors search.
    p : float, default=2
        Parameter for the Minkowski metric.
    metric_params : dict or None, default=None
        Additional keyword arguments for the metric function.

    """

    # Key Attributes
    name: str = "Isometric Mapping"
    module: str = "lsd.generate.dim_reductions.models.isomap"
    n_neighbors: Union[int, str] = 30

    # Unique Attributes
    radius: Optional[float] = None
    eigen_solver: str = "auto"
    tol: float = 0
    max_iter: Optional[int] = None
    path_method: str = "auto"
    neighbors_algorithm: str = "auto"
    n_jobs: Optional[int] = None


@dataclass
class LLE(Projector):
    """
    Locally Linear Embedding (LLE) Configuration.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html
    for more information.

    Key Attributes
    -----------------
    n_neighbors : int
        The number of nearest neighbors used for the projection.
    reg : float
        The regularization parameter for the projection.

    Parameters
    ----------
    eigen_solver : str or None
        Solver to use for eigenvalue decomposition.
    tol : float or None
        Tolerance for algorithm convergence.
    max_iter : int or None
        Maximum number of iterations for the optimization.
    method : str
        Method to use for LLE computation.
    hessian_tol : float
        Tolerance for Hessian eigenmapping method.
    modified_tol : float
        Tolerance for modified LLE method.
    neighbors_algorithm : str
        Algorithm to use for nearest neighbors search.
    random_state : int or None
        Determines random number generation for dataset shuffling and eigensolver.
    n_jobs : int or None
        Number of parallel jobs to run for neighbors search.
    """

    # Key Attributes
    name: str = "Locally Linear Embedding"
    module: str = "lsd.generate.dim_reductions.models.lle"
    n_neighbors: int = 5
    reg: float = 0.001
    # Unique Attributes
    eigen_solver: Optional[str] = "auto"
    tol: Optional[float] = 1e-6
    n_components: int = 2
    max_iter: Optional[int] = 100
    method: str = "standard"
    hessian_tol: float = 1e-4
    modified_tol: float = 1e-12
    neighbors_algorithm: str = "auto"
    random_state: Optional[int] = None
    n_jobs: Optional[int] = None


@dataclass
class Phate(Projector):
    """
    Potential of Heat-diffusion for Affinity-based Transition Embedding (PHATE) Configuration.

    See https://phate.readthedocs.io/en/stable/ for more information.

    NOTE: `PHATE` is not yet compatible with "arm64e" or "arm64" architechtures.

    Key Attributes
    -----------------
    knn : int
        Number of nearest neighbors on which to build kernel.
    decay : Optional[int]
        Decay rate of kernel tails used in alpha decay.
    gamma : float
        Informational distance constant between -1 and 1.
    t : int or "auto"
        Power to which the diffusion operator is powered.
        If 'auto', t is selected based on the knee point in the Von Neumann Entropy of the diffusion operator.

    Additional Attributes
    ---------------------
    n_landmark : int
        Number of landmarks to use in fast PHATE.
    n_pca : int
        Number of principal components to use for calculating neighborhoods.
        For extremely large datasets, using n_pca < 20 allows neighborhoods to be calculated in roughly log(n_samples) time.
    mds_solver : str
        Which solver to use for metric MDS.
    knn_max : Optional[int]
        Maximum number of neighbors for which alpha decaying kernel is computed for each point.
    mds_dist : str
        Distance metric for MDS.
    mds : str
        Selects which MDS algorithm is used for dimensionality reduction.
    n_jobs : int
        The number of jobs to use for the computation.
    random_state : int
        The generator used to initialize SMACOF (metric, nonmetric) MDS.
    verbose : int
        If True or > 0, print status messages.

    """

    # Key Attributes
    name: str = (
        "Potential of Heat-diffusion for Affinity-based Transition Embedding"
    )
    module: str = "lsd.generate.dim_reductions.models.phate"
    knn: int = 5
    decay: Optional[int] = 40
    gamma: float = 1.0
    t: Union[int, str] = "auto"
    # Additional Attributes
    n_landmark: int = 2000
    n_pca: int = 100
    mds_solver: str = "sgd"
    knn_max: Optional[int] = None
    mds_dist: str = "euclidean"
    mds: str = "metric"
    n_jobs: int = 1
    random_state: Optional[int] = None
    verbose: int = 1


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class GeneratedData(Protocol):
    generator: str
    num_samples: int = 1000
    seed: int = 42
    path: Optional[str] = None


@dataclass
class LocalData(Protocol):
    path: str
    module: str = "lsd.generate.dim_reductions.datasets.local"
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
class swiss_roll(GeneratedData):
    name: str = "Swiss Roll"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "swiss_roll"
    hole: bool = False
    noise: float = 0.0
    num_classes: int = 3


@dataclass
class barbell(GeneratedData):
    name: str = "Barbell"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "barbell"
    beta: float = 1.0
    num_classes: int = 3


@dataclass
class noisy_annulus(GeneratedData):
    name: str = "Noisy Annulus"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "noisy_annulus"
    inner_radius: float = 2.0
    outer_radius: float = 6.0
    noise: float = 0.01
    num_classes: int = 3


@dataclass
class blobs(GeneratedData):
    name: str = "Blobs"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "blobs"
    num_samples: int = 1000
    num_classes: int = 3


@dataclass
class moons(GeneratedData):
    name: str = "Moons"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "moons"
    num_samples: int = 1000
    num_classes: int = 3


@dataclass
class MNIST(LocalData):
    name: str = "MNIST"
    module: str = "lsd.generate.dim_reductions.datasets.local"
    generator: str = "mnist"
    num_samples: int = 1000
    path: Optional[str] = None


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Thread:
    name: str = "Multi-threading"
    module: str = "lsd.generate.dim_reductions.implementations.thread"
    n_jobs: int = 1
