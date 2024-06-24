from typing import Dict, List, Union

import gudhi as gd
import numpy as np
from gudhi.representations import Landscape
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection as Gauss
from tqdm import tqdm


class Presto:
    """
    Presto is a class for computing topological similarity between embeddings.

    Presto stands for: _PR_ojected _E_mbedding _S_imilarity based on Topological _O_verlays.

    Presto computes the topological similarity between two embeddings X and Y
    by projecting them into a lower-dimensional space and computing the
    average persistence landscape across multiple random projections.


    Parameters:
    - projector : class, optional
        The type of projection method to use. Default is PCA.
    - max_homology_dim : int, optional
        The maximum homology dimension to consider. Default is 1.
    - resolution : int, optional
        The resolution parameter for computing persistence landscapes. Default is 1000.

    Attributes:
    - projectionsX : list
        List of random projections for X.
    - projectionsY : list
        List of random projections for Y.
    - all_landscapesX : list
        List of persistence landscapes for X.
    - all_landscapesY : list
        List of persistence landscapes for Y.
    - landscapeX : dict
        Average persistence landscape for X.
    - landscapeY : dict
        Average persistence landscape for Y.

    Methods:
    - fit(X, Y, n_components=2, normalize=False, n_projections=100, normalization_approx_iterations=1000, seed=42)
        Fit a topological descriptor to embeddings X & Y.
    - fit_transform(X, Y, score_type="aggregate", n_components=2, n_projections=100, normalize=False, normalization_approx_iterations=1000, seed=42)
        Fit a topological descriptor for spaces X & Y and compute the Presto score (distance).

    Examples:
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.random_projection import GaussianRandomProjection as Gauss
    >>> from presto import Presto

    >>> # Compute Presto Distance between two embeddings
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> Y = np.array([[9, 8, 7, 6], [6, 5, 4, 3], [4, 3, 2, 1]])
    >>> Presto().fit_transform(X, Y,normalize=True)
    0.8058140896444504

    >>> # Create an instance of Presto with custom parameters
    >>> presto = Presto(projector=Gauss, max_homology_dim=2, resolution=500)
    >>> presto.fit_transform(X,Y,normalize=True,score_type="aggregate",n_projections=100)
    0.540073093798171

    >>> # Build Topological Descriptors and compute Presto Scores
    >>> projectionsX = presto._generate_projections(Gauss, X, 2, 100)
    >>> projectionsY = presto._generate_projections(Gauss, Y, 3, 150)
    >>> landscapesX = presto._generate_landscapes(projectionsX)
    >>> landscapesY = presto._generate_landscapes(projectionsY)
    >>> descriptorX = presto._average_landscape(landscapesX)
    >>> descriptorY = presto._average_landscape(landscapesY)
    >>> presto.compute_presto_scores(descriptorX, descriptorY, score_type="average")
    38.74874477538413

    >>> # Compute Presto Sensitivity Scores
    >>> landscapes = [descriptorX, descriptorY]
    >>> Presto().compute_presto_variance(landscapes)
    3.2124896082406025

    """

    def __init__(
        self,
        projector=PCA,
        max_homology_dim: int = 1,
        resolution: int = 100,
    ) -> None:
        """
        Initialize `Presto` to efficiently compute the topological similarity of embeddings.


        Parameters:
        - projector : class, optional
            The type of projection method to use. Default is PCA.
        - max_homology_dim : int, optional
            The maximum homology dimension to consider. Default is 1.
        - resolution : int, optional
            The resolution parameter for computing persistence landscapes. Default is 1000.
        """
        self.P = projector

        # Set Topological parameters
        self.max_homology_dim = max_homology_dim
        self.homology_dims = list(range(0, max_homology_dim + 1))
        self.landscape_resolution = resolution
        self.LS = Landscape(
            resolution=self.landscape_resolution, keep_endpoints=False
        )

        self._projectionsX = None
        self._projectionsY = None
        self._all_landscapesX = None
        self._all_landscapesY = None
        self._landscapeX = None
        self._landscapeY = None

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Getters                                                  │
    #  ╰──────────────────────────────────────────────────────────╯
    @property
    def projectionsX(self):
        return self._projectionsX

    @property
    def projectionsY(self):
        return self._projectionsY

    @property
    def all_landscapesX(self):
        return self._all_landscapesX

    @property
    def all_landscapesY(self):
        return self._all_landscapesY

    @property
    def landscapeX(self):
        return self._landscapeX

    @property
    def landscapeY(self):
        return self._landscapeY

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Setters                                                  │
    #  ╰──────────────────────────────────────────────────────────╯

    @projectionsX.setter
    def projectionsX(self, projectionsX):
        self._projectionsX = projectionsX

    @projectionsY.setter
    def projectionsY(self, projectionsY):
        self._projectionsY = projectionsY

    @all_landscapesX.setter
    def all_landscapesX(self, landscapesX):
        self._all_landscapesX = landscapesX

    @all_landscapesY.setter
    def all_landscapesY(self, landscapesY):
        self._all_landscapesY = landscapesY

    @landscapeX.setter
    def landscapeX(self, landscapeX):
        self._landscapeX = landscapeX

    @landscapeY.setter
    def landscapeY(self, landscapeY):
        self._landscapeY = landscapeY

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Key Members: Fit & Transform                             │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(
        self,
        X,
        Y,
        n_components: int = 2,
        normalize: bool = False,
        n_projections=100,
        normalization_approx_iterations: int = 1000,
        seed: int = 42,
    ):
        """
        Fit a topological descriptor to embeddings X & Y.

        This function computes `n_projections` random projections of X and Y
        using `projector`. Each projection is transformed into
        a persistence landscape using the `gudhi` library.

        The fitted topological descriptor is the average persistence
        landscape, aggregated over each of the projections. Stored as:
            - self._landscapeX
            - self._landscapeY

        Parameters:
        - X : np.ndarray
            The first embedding to fit. Shape(n_samples,n_features)
        - Y : np.ndarray
            The second embedding to fit. NEED NOT share the same shape as X.
        - n_projections : int, optional
            The number of random projections. Default is 100.
        - n_components : int, optional
            The number of components for the random projection. Default is 2.
        - normalize : bool, optional
            Whether to normalize the space based on an approximate diameter. Default is False.

        - normalization_approx_iterations : int, optional
            The number of iterations for approximating the space diameter during normalization. Default is 1000.
        - seed : int, optional
            Seed for the random number generator. Default is 42.
        """
        # Normalize
        if normalize:
            X = Presto._normalize_space(
                X,
                diameter_iterations=normalization_approx_iterations,
                seed=seed,
            )
            Y = Presto._normalize_space(
                Y,
                diameter_iterations=normalization_approx_iterations,
                seed=seed,
            )

        # Project
        self._projectionsX = self._generate_projections(
            self.P,
            X,
            n_components,
            n_projections,
            seed=seed,
            tag="Space X",
        )
        self._projectionsY = self._generate_projections(
            self.P,
            Y,
            n_components,
            n_projections,
            seed=seed,
            tag="Space Y",
        )

        # Fit Landscapes
        self._all_landscapesX = self._generate_landscapes(
            self._projectionsX,
            self.LS,
            self.homology_dims,
            tag="Space X",
        )
        self._all_landscapesY = self._generate_landscapes(
            self._projectionsY,
            self.LS,
            self.homology_dims,
            tag="Space Y",
        )

        # Assign Descriptors
        self._landscapeX = Presto._average_landscape(
            self._all_landscapesX, tag="Space X"
        )
        self._landscapeY = Presto._average_landscape(
            self._all_landscapesY, tag="Space Y"
        )

    def fit_transform(
        self,
        X,
        Y,
        score_type: str = "aggregate",
        n_components: int = 2,
        n_projections: int = 100,
        normalize: bool = False,
        normalization_approx_iterations: int = 1000,
        seed: int = 42,
    ):
        """
        Fit a topological descriptor for spaces X & Y and compute the Presto score (distance).

        Parameters:
        - X : array-like or pd.DataFrame, shape (n_samples, n_features), default=None
            Ignored. Placeholder for compatibility.
        - Y : array-like or pd.DataFrame, shape (n_samples, n_features)
            The second set of embeddings.
        - n_projections : int, optional
            The number of random projections. Default is 100. Ignored when projector is PCA.
        - score_type: str, optional
            Which type of Presto score to return. Options are:
            - "aggregate" (sum normed distances across all dimensions)
            - "average" (average distance across al dimensions)
            - "separate" (list of distances by dimension)
            Default is "aggregate".

        Returns:
        - presto_score : float
            The computed Presto score representing the distance between the topological descriptors of X and Y.
        """
        N = n_projections if type(self.P) != PCA else 1
        # Fit Topological Descriptors
        self.fit(
            X,
            Y,
            n_components=n_components,
            normalize=normalize,
            n_projections=N,
            normalization_approx_iterations=normalization_approx_iterations,
            seed=seed,
        )

        return Presto.compute_presto_scores(
            self._landscapeX, self._landscapeY, score_type=score_type
        )

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Data Parameters                                          │
    #  ╰──────────────────────────────────────────────────────────╯
    @staticmethod
    def _normalize_space(
        X,
        diameter_iterations=1000,
        seed=42,
    ):
        """
        Normalize a space based on an approximate diameter.

        Parameters:
        - X : np.ndarray
            The input space to be normalized.
        - diameter_iterations : int, optional
            The number of iterations to approximate the space diameter. Default is 1000.
        - seed : int, optional
            Seed for the random number generator. Default is 42.

        Returns:
        - np.ndarray
            The normalized space based on approximate diameter.
        """
        rng = np.random.default_rng(seed)
        subset = [rng.choice(len(X))]
        for _ in range(diameter_iterations - 1):
            distances = cdist([X[subset[-1]]], X).ravel()
            new_point = np.argmax(distances)
            subset.append(new_point)
        pairwise_distances = cdist(X[subset], X[subset])
        diameter = np.max(pairwise_distances)
        return X / diameter

    @staticmethod
    def _generate_projections(
        P,
        X,
        dim,
        n_projections,
        seed=42,
        tag: str = None,
    ):
        """
        Generate projections of the input data using projection method P.

        Parameters:
        - P : class
            The projection method.
        - X : np.ndarray
            The input data.
        - dim : int
            The number of components for the random projection.
        - n_projections : int
            The number of random projections.
        - seed : int, optional
            Seed for the random number generator. Default is 42.
        - tag : str, optional
            A tag for users.

        Returns:
        - random_projections : list
            List of random projections.
        """
        desc = (
            f"Generating Projections for {tag}"
            if tag
            else "Generating Projections"
        )
        rng = np.random.default_rng(seed)
        random_projections = []
        seeds = rng.choice(2**16 - 1, n_projections, replace=False).tolist()
        for _ in tqdm(
            range(n_projections),
            total=n_projections,
            desc=desc,
        ):

            # Initialize Random Projector
            state = np.random.mtrand.RandomState(seeds.pop())
            projector = P(n_components=dim, random_state=state)
            random_projections.append(projector.fit_transform(X))
        return random_projections

    @staticmethod
    def _generate_landscapes(
        projections: List[np.array],
        LS: Landscape = Landscape(resolution=1000, keep_endpoints=False),
        homology_dims: List[int] = [0, 1],
        tag: str = None,
    ) -> List[Dict[int, np.array]]:
        """
        Generate persistence landscapes from a list of projections.

        Parameters:
        - projections : list
            List of projections.

        Returns:
        - landscapes : dict
            Dictionary containing persistence landscapes for each homology dimension.
        """
        desc = (
            f"Generating Landscapes for {tag}"
            if tag
            else "Generating Landscapes"
        )
        landscapes = []
        for X_ in tqdm(
            projections,
            total=len(projections),
            desc=desc,
        ):
            landscape = {dim: np.array([]) for dim in homology_dims}
            alpha_complex = gd.AlphaComplex(
                points=X_, precision="fast"
            ).create_simplex_tree()
            alpha_complex.persistence()
            for dim in homology_dims:
                persistence_pairs = Presto._mask_infinities(
                    alpha_complex.persistence_intervals_in_dimension(dim)
                )

                landscape[dim] = LS.fit_transform([persistence_pairs])
            landscapes.append(landscape)
        return landscapes

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Presto Similarity & Sensitivity Scores                   │
    #  ╰──────────────────────────────────────────────────────────╯

    @staticmethod
    def compute_presto_scores(
        landscapeX, landscapeY, score_type: str = "aggregate"
    ):
        prestos = Presto._compute_landscape_norm(
            Presto._subtract_landscapes(landscapeX, landscapeY),
            score_type=score_type,
        )
        return prestos

    @staticmethod
    def compute_local_presto_sensitivity(
        landscape_equivalence_classes: List[List[Dict[int, np.array]]]
    ) -> float:
        """
        PS^2_k(MM | i) := sqrt( 1/q_i * sum_{Q \in QQ_i} PV^2_k(LL[Q]) ),
        where q_i is the number equivalence classes of models in dimension i.
        NB: We expect the caller to have grouped the relevant landscapes corresponding to the analysis of interest.
        :param landscape_equivalence_classes:
        :return:
        """
        q_i = len(landscape_equivalence_classes)
        sum_of_variances = sum(
            Presto.compute_presto_variance(Q)
            for Q in landscape_equivalence_classes
        )
        return np.sqrt(sum_of_variances / q_i)

    @staticmethod
    def compute_global_presto_sensitivity(
        landscape_equivalence_classes_per_dim: List[
            List[List[Dict[int, np.array]]]
        ]
    ) -> float:
        """
        PS^2_k(MM) := sqrt( 1/c * sum_{i \in [c]} 1/q_i * sum_{Q \in QQ_i} PV^2_k(LL[Q]) ),
        where c is the dimensionality of models in MM
        NB: We expect the caller to have grouped the relevant landscapes corresponding to the analysis of interest.
        :param landscape_equivalence_classes_per_dim:
        :return:
        """
        c = len(landscape_equivalence_classes_per_dim)
        sum_of_local_sensitivities = sum(
            Presto.compute_local_presto_sensitivity(
                landscape_equivalence_classes
            )
            ** 2
            for landscape_equivalence_classes in landscape_equivalence_classes_per_dim
        )
        return np.sqrt(sum_of_local_sensitivities / c)

    @staticmethod
    def compute_individual_presto_sensitivity(
        landscapes: List[Dict[int, np.array]]
    ) -> float:
        return Presto.compute_presto_coordinate_sensitivity(landscapes)

    @staticmethod
    def compute_presto_coordinate_sensitivity(
        landscapes: List[Dict[int, np.array]]
    ) -> float:
        """
        PCS^2_k(theta | MM) := sqrt( PV^2_k(LL[theta^{\pm 1}]) )
        NB: We expect the caller to have selected the relevant landscapes corresponding to the coordinates of interest.
        :param landscapes:
        :return:
        """
        return np.sqrt(Presto.compute_presto_variance(landscapes))

    @staticmethod
    def compute_presto_variance(landscapes: List[Dict[int, np.array]]) -> float:
        """
        PV^2_k(LL) := 1/|LL| * sum_{x=0}^k sum_{L \in LL^x} (||L||_2 - mean_{||LL^x||_2})^2
        :param landscapes:
        :return:
        """
        N = len(landscapes)
        homology_dims = range(max(landscapes[0]))
        landscape_norm_means, landscape_norms = (
            Presto._compute_landscape_norm_means(landscapes, return_norms=True)
        )
        dim_sums = 0
        for dim in homology_dims:
            dim_sum = 0
            for L in landscape_norms:
                dim_sum += (L[dim] - landscape_norm_means[dim]) ** 2
            dim_sums += dim_sum
        return dim_sums / N

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Persistence Landscape Helper Functions                   │
    #  ╰──────────────────────────────────────────────────────────╯

    @staticmethod
    def _add_landscapes_abs(
        landscapes: List[Dict[int, np.array]]
    ) -> Dict[int, np.array]:
        res = dict()
        for i in landscapes[0].keys():
            res[i] = sum(np.abs(Li[i]) for Li in landscapes)
        return res

    @staticmethod
    def _subtract_landscapes(
        landscapeX: Dict[int, np.array], landscapeY: Dict[int, np.array]
    ) -> Dict[int, np.array]:
        res = dict()
        for i in landscapeX.keys():
            res[i] = landscapeX[i] - landscapeY[i]
        return res

    @staticmethod
    def _pivot_landscapes(
        landscapes: Dict[int, List[np.array]]
    ) -> List[Dict[int, np.array]]:
        dimensions = sorted(landscapes.keys())
        n_landscapes = len(landscapes[0])
        pivoted = list()
        for idx in range(n_landscapes):
            pivoted.append({dim: landscapes[dim][idx] for dim in dimensions})
        return pivoted

    @staticmethod
    def _compute_landscape_norm(
        landscape: Dict[int, np.array],
        score_type: str = "aggregate",
    ) -> Union[Dict[int, float], float]:
        norms = {k: np.linalg.norm(v) for k, v in landscape.items()}
        if score_type == "aggregate":
            return sum(norms.values())
        elif score_type == "average":
            return sum(norms.values()) / len(norms.values())
        elif score_type == "separate":
            return norms
        else:
            raise NotImplementedError(score_type)

    @staticmethod
    def _compute_landscape_norm_means(
        landscapes: List[Dict[int, np.array]], return_norms: bool = False
    ):
        """
        We expect each landscape in the input list to be of the shape returned by _average_landscape, i.e., Dict[int, np.array].
        :param landscapes:
        :return:
        """
        N = len(landscapes)
        max_homology_dimension = max(landscapes[0].keys())
        landscape_norms = [
            Presto._compute_landscape_norm(L, score_type="separate")
            for L in landscapes
        ]
        landscape_norm_means = {
            i: sum(L[i] for L in landscape_norms) / N
            for i in range(max_homology_dimension + 1)
        }
        if return_norms:
            return landscape_norm_means, landscape_norms
        else:
            return landscape_norm_means

    @staticmethod
    def _average_landscape(
        L: List[Dict[int, np.array]],
        tag: str = None,
    ) -> Dict[int, np.array]:
        """
        Compute the average persistence landscape of a list of landscapes.

        Parameters:
        - L : List
            List of landscape dictionaries.

        Returns:
        - avg : dict
            Dictionary containing the average persistence landscape for each homology dimension.
        """
        desc = (
            f"Averaging Landscapes for {tag}" if tag else "Averaging Landscapes"
        )
        avg = {}
        for landscape in tqdm(L, desc=desc, total=len(L)):
            for dim in landscape.keys():
                if dim not in avg:
                    avg[dim] = np.zeros_like(landscape[dim])
                avg[dim] += landscape[dim]

        for dim in avg.keys():
            avg[dim] /= len(L)
        return avg

    @staticmethod
    def _mask_infinities(array):
        return array[array[:, 1] < np.inf]
