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
    p: int, optional
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
    n_epochs :int, optional (default: None)
        The number of training epochs to be used in optimizing the embedding.
    learning_rate : float, optionalal (default: 1.0)
        The initial learning rate for the embedding optimization.
    spread : float, optionalal (default: 1.0)
        The effective scale of embedded points.
    set_op_mix_ratio : float, optionalal (default: 1.0)
        Interpolate between (fuzzy) union and intersection set operations.
    local_connectivity :int, optional (default: 1)
        The local connectivity required.
    repulsion_strength : float, optionalal (default: 1.0)
        Weighting applied to negative samples in low dimensional embedding optimization.
    negative_sample_rate :int, optional (default: 5)
        The number of negative samples to select per positive sample in optimization.
    transform_queue_size : float, optionalal (default: 1.0)
        This scales the data transform queue size.
    a : float, optionalal (default: None)
        Early exaggeration parameter.
    b : float, optionalal (default: None)
        The final value of the early exaggeration parameter.
    random_state : int or RandomState, optionalal (default: None)
        Determines the random number generation for reproducibility.
    verbose : bool, optionalal (default: False)
        Controls verbosity of the output.

    """

    # Key Attributes
    name: str = "Uniform Manifold Approximation and Projection"
    module: str = "lsd.generate.dim_reductions.models.umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    init: str = "spectral"


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
    random_state : int, RandomState instance, optional, default=None
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


@dataclass
class Isomap(Projector):
    """
    Isometric Mapping (Isomap) Configuration.

    See https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    for more information.

    Key Attributes
    -----------------
    n_neighbors : int, optional, default=5
        Number of neighbors to consider for each point.
    radius : float, optional, default=None
        Limiting distance of neighbors to return.


    Additional Attributes
    ---------------------
    eigen_solver : str, default='auto'
        Solver to use for eigenvalue decomposition.
    tol : float, default=0
        Tolerance for eigenvalue decomposition.
    max_iter : int, optional, default=None
        Maximum number of iterations for eigenvalue decomposition.
    path_method : str, default='auto'
        Method to use in finding shortest paths.
    neighbors_algorithm : str, default='auto'
        Algorithm to use for nearest neighbors search.
    n_jobs : int, optional, default=None
        Number of parallel jobs to run for neighbors search.
    p : float, default=2
        Parameter for the Minkowski metric.
    metric_params : dict, optional, default=None
        Additional keyword arguments for the metric function.

    """

    # Key Attributes
    name: str = "Isometric Mapping"
    module: str = "lsd.generate.dim_reductions.models.isomap"
    n_neighbors: Union[int, str] = 30
    radius: Optional[float] = None


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
    eigen_solver : str, optional
        Solver to use for eigenvalue decomposition.
    tol : float, optional
        Tolerance for algorithm convergence.
    max_iter : int, optional
        Maximum number of iterations for the optimization.
    method : str
        Method to use for LLE computation.
    hessian_tol : float
        Tolerance for Hessian eigenmapping method.
    modified_tol : float
        Tolerance for modified LLE method.
    neighbors_algorithm : str
        Algorithm to use for nearest neighbors search.
    random_state : int, optional
        Determines random number generation for dataset shuffling and eigensolver.
    n_jobs : int, optional
        Number of parallel jobs to run for neighbors search.
    """

    # Key Attributes
    name: str = "Locally Linear Embedding"
    module: str = "lsd.generate.dim_reductions.models.lle"
    n_neighbors: int = 5
    reg: float = 0.001


@dataclass
class Phate(Projector):
    """
    Potential of Heat-diffusion for Affinity-based Transition Embedding (PHATE) Configuration.

    See https://phate.readthedocs.io/en/stable/ for more information.

    NOTE: `PHATE` is not yet compatible with "arm64e" or "arm64" architechtures. We have intentionally left this out of the `Presto` virtual environments to this and other dependency issues.

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


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class GeneratedData(Protocol):
    """
    Base Protocol for generating synthetic datasets.

    At the moment, we support generating datasets from Scikit-Learn's datasets module and some custom manifolds (see `lsd.generate.dim_reductions.datasets.manifolds`).

    Sklearn datasets can be implemented with the name of the loading function as the `generator` attribute. When `module` attribute is specified, our implementation will look for the generator function in that module.

    Attributes
    ----------
    generator : str
        The name of the dataset generator function.
    num_samples :int, optional
        The number of samples to generate. Default is 1000.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    path : str, optional
        Path to save the generated dataset. Default is None.

    Examples
    --------
    >>> iris = GeneratedData(generator="load_iris", num_samples=1000, seed=42)
    >>> noisy_annulus = GeneratedData(
    name="Noisy Annulus",
    module="lsd.generate.dim_reductions.datasets.manifolds",
    generator="noisy_annulus",
    inner_radius=2.0,
    outer_radius=6.0,
    noise=0.01,
    num_classes=3,
    num_samples=1000,
    seed=42,
    path=None
    )
    """

    generator: str
    num_samples: int = 1000
    seed: int = 42
    path: Optional[str] = None


@dataclass
class LocalData(Protocol):
    """
    Base Protocol for loading local datasets.

    Attributes
    ----------
    path : str
        The file path to the local dataset.
    module : str, optional
        The module name for loading the dataset. Default is "lsd.generate.dim_reductions.datasets.local". Here you can add custom loading functions.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    """

    path: str
    module: str = "lsd.generate.dim_reductions.datasets.local"
    seed: int = 42


@dataclass
class iris(GeneratedData):
    """
    Class for generating the Iris dataset from Scikit-Learn.
    Inherits from the `GeneratedData` Protocol.

    Scikit-learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

    Attributes
    ----------
    name : str
        The name of the dataset. Set to "Iris".
    generator : str
        The generator function for the Iris dataset. Set to "load_iris".
    num_classes : int
        The number of classes in the dataset. Set to 3.
    """

    name: str = "Iris"
    generator: str = "load_iris"
    num_classes: int = 3


@dataclass
class digits(GeneratedData):
    """
    Class for generating the Digits dataset us.
    Inherits from the `GeneratedData` Protocol.

    Scikit-learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html


    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Digits".
    generator : str
        The generator function for the Digits dataset. Default is "load_digits".
    num_classes : int
        The number of classes in the dataset. Default is 10.
    """

    name: str = "Digits"
    generator: str = "load_digits"
    num_classes: int = 10


@dataclass
class linnerud(GeneratedData):
    """
    Class for generating the Linnerud dataset.

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Linnerud".
    generator : str
        The generator function for the Linnerud dataset. Default is "load_linnerud".
    """

    name: str = "Linnerud"
    generator: str = "load_linnerud"


@dataclass
class wine(GeneratedData):
    """
    Class for generating the Wine dataset.
    Inherits from the `GeneratedData` Protocol.

    Scikit-learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

    Attributes
    ----------
    name : str
        The name of the dataset. Set to "Wine".
    generator : str
        The generator function for the Wine dataset. Set to "load_wine".
    """

    name: str = "Wine"
    generator: str = "load_wine"


@dataclass
class breast_cancer(GeneratedData):
    """
    Class for generating the Breast Cancer dataset.
    Inherits from the `GeneratedData` Protocol.

    Scikit-learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

    Attributes
    ----------
    name : str
        The name of the dataset. Set to "Breast Cancer".
    generator : str
        The generator function for the Breast Cancer dataset. Set to "load_breast_cancer".
    """

    name: str = "Breast Cancer"
    generator: str = "load_breast_cancer"


@dataclass
class swiss_roll(GeneratedData):
    """
    Class for generating the Swiss Roll dataset, a Manifold dataset.
    Inherits from the `GeneratedData` Protocol, see `lsd.generate.dim_reductions.datasets.manifolds` for our implementation.

    Scikit-learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Swiss Roll".
    module : str
        The module name for generating the dataset. Set to "lsd.generate.dim_reductions.datasets.manifolds".
    generator : str
        The generator function for the Swiss Roll dataset. Set to "swiss_roll".
    hole : bool, optionalal
        Whether to include a hole in the Swiss Roll. Default is False.
    noise : float, optionalal
        The amount of noise to add to the dataset. Default is 0.0.
    num_classes : int
        The number of classes in the dataset. Default is 3.
    """

    name: str = "Swiss Roll"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "swiss_roll"
    hole: bool = False
    noise: float = 0.0
    num_classes: int = 3


@dataclass
class barbell(GeneratedData):
    """
    Class for generating the Barbell dataset.
    Inherits from the `GeneratedData` Protocol, see `lsd.generate.dim_reductions.datasets.manifolds` for our implementation.

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Barbell".
    module : str
        The module name for generating the dataset. Set to "lsd.generate.dim_reductions.datasets.manifolds".
    generator : str
        The generator function for the Barbell dataset. Set to "barbell".
    beta : float, optionalal
        The shape parameter for the Barbell. Default is 1.0.
    num_classes : int
        The number of classes in the dataset. Default is 3.
    """

    name: str = "Barbell"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "barbell"
    beta: float = 1.0
    num_classes: int = 3


@dataclass
class noisy_annulus(GeneratedData):
    """
    Class for generating the Noisy Annulus dataset.
    Inherits from the `GeneratedData` Protocol, see `lsd.generate.dim_reductions.datasets.manifolds` for our implementation.

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Noisy Annulus".
    module : str
        The module name for generating the dataset. Set to "lsd.generate.dim_reductions.datasets.manifolds".
    generator : str
        The generator function for the Noisy Annulus dataset. Set to "noisy_annulus".
    inner_radius : float, optionalal
        The inner radius of the annulus. Default is 2.0.
    outer_radius : float, optionalal
        The outer radius of the annulus. Default is 6.0.
    noise : float, optionalal
        The amount of noise to add to the dataset. Default is 0.01.
    num_classes : int
        The number of classes in the dataset. Default is 3.
    """

    name: str = "Noisy Annulus"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "noisy_annulus"
    inner_radius: float = 2.0
    outer_radius: float = 6.0
    noise: float = 0.01
    num_classes: int = 3


@dataclass
class blobs(GeneratedData):
    """
    Class for generating the Blobs dataset.
    Inherits from the `GeneratedData` Protocol, see `lsd.generate.dim_reductions.datasets.manifolds` for our implementation.

    Sci-kit Learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Blobs".
    module : str
        The module name for generating the dataset. Set to "lsd.generate.dim_reductions.datasets.manifolds".
    generator : str
        The generator function for the Blobs dataset. Set to "blobs".
    num_samples :int, optional
        The number of samples to generate. Default is 1000.
    seed :int, optional
        Random seed for reproducibility. Default is 42.
    path : Optional[str], optionalal
        Path to save the generated dataset. Default is None.
    num_classes : int
        The number of classes in the dataset. Default is 3.
    """

    name: str = "Blobs"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "blobs"
    num_samples: int = 1000
    num_classes: int = 3


@dataclass
class moons(GeneratedData):
    """
    Class for generating the Moons dataset.
    Inherits from the `GeneratedData` Protocol, see `lsd.generate.dim_reductions.datasets.manifolds` for our implementation.

    Sci-kit Learn Documentation:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

    Attributes
    ----------
    name : str
        The name of the dataset. Default is "Moons".
    module : str
        The module name for generating the dataset. Set to "lsd.generate.dim_reductions.datasets.manifolds".
    generator : str
        The generator function for the Moons dataset. Set to "moons".
    num_samples :int, optional
        The number of samples to generate. Default is 1000.
    seed :int, optional
        Random seed for reproducibility. Default is 42.
    path : Optional[str], optionalal
        Path to save the generated dataset. Default is None.
    num_classes : int
        The number of classes in the dataset. Default is 3.
    """

    name: str = "Moons"
    module: str = "lsd.generate.dim_reductions.datasets.manifolds"
    generator: str = "moons"
    num_samples: int = 1000
    num_classes: int = 3


@dataclass
class MNIST(LocalData):
    """
    Class for loading the MNIST dataset.
    Inherits from the `LocalData` Protocol, see `lsd.generate.dim_reductions.datasets.local` for our implementation.


    This example implementation uses `mnist.npz` taken from Kaggle:
    https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy



    Attributes
    ----------
    name : str
        The name of the dataset. Default is "MNIST".
    module : str
        The module name for loading the dataset. Default is "lsd.generate.dim_reductions.datasets.local".
    generator : str
        The generator function for the MNIST dataset. Default is "mnist".
    num_samples :int, optional
        The number of samples to load. Default is 1000.
    seed :int, optional
        Random seed for reproducibility. Default is 42.
    path : Optional[str], optionalal
        Path to the local dataset file. Default is None.
    """

    name: str = "MNIST"
    module: str = "lsd.generate.dim_reductions.datasets.local"
    generator: str = "mnist"
    num_samples: int = 1000
    path: Optional[str] = None
    name: str = "MNIST"
    module: str = "lsd.generate.dim_reductions.datasets.local"
    generator: str = "mnist"
    num_samples: int = 1000
    path: Optional[str] = None


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Trainer(Protocol):
    """
    Base Protocol for dimensionality reduction trainers.
    These are parameters shared between DimReduction implementations.

    Attributes
    ----------
    n_jobs : int
        Number of parallel jobs to run. Default is 1.
    random_state : Optional[int]
        Random seed for reproducibility. Default is 42.
    verbose : bool
        Controls verbosity of the output. Default is False.
    """

    n_jobs: int = 1
    random_state: Optional[int] = 42
    verbose: bool = 0


@dataclass
class UMAPTrainer(Trainer):
    """
    Trainer parameters for UMAP (Uniform Manifold Approximation and Projection).

    Attributes
    ----------
    n_epochs : Optional[int], optional
        Number of training epochs. Default is None (auto).
    learning_rate : float, optional
        The initial learning rate for the optimization. Default is 1.0.
    spread : float, optional
        Effective scale of embedded points. Default is 1.0.
    set_op_mix_ratio : float, optional
        Interpolation between fuzzy union and intersection. Default is 1.0.
    local_connectivity : int, optional
        Number of nearest neighbors with high connectivity. Default is 1.
    repulsion_strength : float, optional
        Strength of the repulsion force. Default is 1.0.
    negative_sample_rate : int, optional
        Number of negative samples per positive sample. Default is 5.
    transform_queue_size : float, optional
        Size of the queue for embedding new points. Default is 1.0.
    a : Optional[float], optional
        Parameter for the fuzzy simplicial set. Default is None.
    b : Optional[float], optional
        Parameter for the fuzzy simplicial set. Default is None.
    """

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


@dataclass
class tSNETrainer(Trainer):
    """
    Trainer parameters for t-SNE (t-distributed Stochastic Neighbor Embedding).

    Attributes
    ----------
    learning_rate : Union[float, str], optional
        The learning rate for the optimization. Default is "auto".
    max_iter : int, optional
        Maximum number of iterations for optimization. Default is 1000.
    n_iter_without_progress : int, optional
        Maximum number of iterations without progress before stopping. Default is 300.
    min_grad_norm : float, optional
        Minimum gradient norm for convergence. Default is 1e-7.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.
    method : str, optional
        The optimization method to use. Default is "barnes_hut".
    angle : float, optional
        Trade-off between speed and accuracy. Default is 0.5.
    """

    learning_rate: Union[float, str] = "auto"
    max_iter: int = 1000
    n_iter_without_progress: int = 300
    min_grad_norm: float = 1e-7
    random_state: Optional[int] = None
    method: str = "barnes_hut"
    angle: float = 0.5
    verbose: int = 0


@dataclass
class IsomapTrainer(Trainer):
    """
    Trainer parameters for Isomap (Isometric Mapping).

    Attributes
    ----------
    n_landmark : int, optional
        Number of landmarks to use for the embedding. Default is 2000.
    n_pca : int, optional
        Number of components for PCA preprocessing. Default is 100.
    """

    eigen_solver: str = "auto"
    tol: float = 0
    max_iter: Optional[int] = None
    path_method: str = "auto"
    neighbors_algorithm: str = "auto"
    n_jobs: Optional[int] = None


@dataclass
class LLETrainer(Trainer):
    """
    Trainer parameters for LLE (Locally Linear Embedding).

    Attributes
    ----------
    eigen_solver : str, optional
        The eigenvalue decomposition method to use. Default is "auto".
    tol : float, optional
        Tolerance for convergence of the eigenvalue decomposition. Default is 1e-6.
    max_iter : Optional[int], optional
        Maximum number of iterations for the solver. Default is 100.
    method : str, optional
        The LLE method to use. Default is "standard".
    hessian_tol : float, optional
        Tolerance for Hessian-based LLE. Default is 1e-4.
    modified_tol : float, optional
        Tolerance for modified LLE. Default is 1e-12.
    neighbors_algorithm : str, optional
        Algorithm to use for nearest neighbors search. Default is "auto".
    """

    eigen_solver: str = "auto"
    tol: float = 1e-6
    max_iter: Optional[int] = 100
    method: str = "standard"
    hessian_tol: float = 1e-4
    modified_tol: float = 1e-12
    neighbors_algorithm: str = "auto"


@dataclass
class PhateTrainer(Trainer):
    """
    Trainer parameters for PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding).
    Attributes
    ----------
    knn_max : Optional[int], optional
        Maximum number of nearest neighbors for the KNN graph. Default is None.
    mds_dist : str, optional
        The distance metric to use for MDS. Default is "euclidean".
    mds : str, optional
        Type of MDS to use. Default is "metric".
    n_jobs : int, optional
        Number of parallel jobs to run. Default is 1.
    random_state : Optional[int], optional
        Random seed for reproducibility. Default is None.
    verbose : int, optional
        Verbosity level. Default is 1.
    """

    n_landmark: int = 2000
    n_pca: int = 100
    mds_solver: str = "sgd"
    knn_max: Optional[int] = None
    mds_dist: str = "euclidean"
    mds: str = "metric"
