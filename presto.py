import gudhi as gd
import numpy as np
from gudhi.representations import Landscape
from scipy.spatial.distance import cdist
from sklearn.random_projection import GaussianRandomProjection as Gauss
from typing import Dict, List


class Presto:
    def __init__(
            self,
            projector=Gauss,
            n_components: int = 2,
            normalize: bool = False,
            max_homology_dim: int = 1,
            resolution: int = 100,
            normalization_approx_iterations: int = 1000,
            seed: int = 42
    ) -> None:
        """
        Initialize `Presto` to efficiently compute the topological similarity of embeddings.


        Parameters:
        - projector : class, optional
            The random projection class used for embedding. Default is GaussianRandomProjection.
        - n_components : int, optional
            The number of components for the random projection. Default is 2.
        - normalize : bool, optional
            Whether to normalize the space based on an approximate diameter. Default is False.
        - max_homology_dim : int, optional
            The maximum homology dimension to consider. Default is 1.
        - resolution : int, optional
            The resolution parameter for computing persistence landscapes. Default is 100.
        - normalization_approx_iterations : int, optional
            The number of iterations for approximating the space diameter during normalization. Default is 1000.
        - seed : int, optional
            Seed for the random number generator. Default is 42.
        """
        # Create random number generator
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # Initialize Projector
        self.projection_dimension = n_components
        self.P = projector(n_components=self.projection_dimension, random_state=np.random.RandomState(self.seed))

        # Set Normalization parameters
        self.normalize = normalize
        self.diameter_iterations = normalization_approx_iterations

        # Set Topological parameters
        self.max_homology_dim = max_homology_dim
        self.homology_dims = list(range(0, max_homology_dim + 1))
        self.landscape_resolution = resolution
        self.LS = Landscape(resolution=self.landscape_resolution, keep_endpoints=False)
        self._projectionsX = None
        self._projectionsY = None
        self._all_landscapesX = None
        self._all_landscapesY = None
        self._landscapeX = None
        self._landscapeY = None

    def fit(
            self,
            X,
            Y,
            n_projections=100,
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
        """

        if self.normalize:
            X, Y = self.normalize_space(X), self.normalize_space(Y)

        # Project
        self._projectionsX = self.generate_projections(X, n_projections)
        self._projectionsY = self.generate_projections(Y, n_projections)

        # Fit Landscapes
        self._all_landscapesX = self.generate_landscapes(self._projectionsX)
        self._all_landscapesY = self.generate_landscapes(self._projectionsY)

        # Average Landscapes
        self._landscapeX = Presto.average_landscape(self._all_landscapesX)
        self._landscapeY = Presto.average_landscape(self._all_landscapesY)

    def fit_transform(self, X, Y, n_projections: int = 15, score_type: str = "aggregate"):
        """
        Fit a topological descriptor and compute the Presto score.

        Parameters:
        - X : array-like or pd.DataFrame, shape (n_samples, n_features), default=None
            Ignored. Placeholder for compatibility.
        - Y : array-like or pd.DataFrame, shape (n_samples, n_features)
            The second set of embeddings.
        - n_projections : int, optional
            The number of random projections. Default is 100.
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
        self.fit(X, Y, n_projections)
        presto_scores = self.compute_presto_scores(self._landscapeX, self._landscapeY)
        if score_type == "aggregate":
            return sum(presto_scores.values())
        elif score_type == "average":
            return sum(presto_scores.values()) / len(presto_scores.values())
        elif score_type == "separate":
            return presto_scores
        else:
            raise NotImplementedError(score_type)

    def compute_presto_scores(self, landscapeX, landscapeY) -> Dict[int, float]:
        prestos = dict()
        for dim in self.homology_dims:
            lambdaX = landscapeX[dim]
            lambdaY = landscapeY[dim]
            if not np.isnan(lambdaX - lambdaY).any():
                prestos[dim] = np.linalg.norm(lambdaX - lambdaY)
        return prestos

    def compute_local_presto_sensitivity(self, landscape_equivalence_classes: List[List[Dict[int, np.array]]]) -> float:
        """
        PS^2_k(MM | i) := sqrt( 1/q_i * sum_{Q \in QQ_i} PV^2_k(LL[Q]) ),
        where q_i is the number equivalence classes of models in dimension i.
        NB: We expect the caller to have grouped the relevant landscapes corresponding to the analysis of interest.
        :param landscape_equivalence_classes:
        :return:
        """
        q_i = len(landscape_equivalence_classes)
        sum_of_variances = sum(self.compute_presto_variance(Q) for Q in landscape_equivalence_classes)
        return np.sqrt(sum_of_variances / q_i)

    def compute_global_presto_sensitivity(self,
                                          landscape_equivalence_classes_per_dim: List[
                                              List[List[Dict[int, np.array]]]]) -> float:
        """
        PS^2_k(MM) := sqrt( 1/c * sum_{i \in [c]} PS^2_k(MM | i) ),
        where c is the dimensionality of models in MM
        NB: We expect the caller to have grouped the relevant landscapes corresponding to the analysis of interest.
        :param landscape_equivalence_classes_per_dim:
        :return:
        """
        c = len(landscape_equivalence_classes_per_dim)
        sum_of_local_sensitivities = sum(
            self.compute_local_presto_sensitivity(landscape_equivalence_classes) for landscape_equivalence_classes in
            landscape_equivalence_classes_per_dim)
        return np.sqrt(sum_of_local_sensitivities / c)

    def compute_presto_coordinate_sensitivity(self, landscapes: List[Dict[int, np.array]]) -> float:
        """
        PCS^2_k(theta | MM) := sqrt( PV^2_k(LL[theta^{\pm 1}]) )
        NB: We expect the caller to have selected the relevant landscapes corresponding to the coordinates of interest.
        :param landscapes:
        :return:
        """
        return np.sqrt(self.compute_presto_variance(landscapes))

    def compute_presto_variance(self, landscapes: List[Dict[int, np.array]]) -> float:
        """
        PV^2_k(LL) := 1/|LL| * sum_{x=0}^k sum_{L \in LL^x} (||L||_2 - mean_{||LL^x||})^2
        :param landscapes:
        :return:
        """
        N = len(landscapes)
        landscape_norm_means, landscape_norms = Presto._compute_landscape_norm_means(landscapes, return_norms=True)
        dim_sums = 0
        for dim in self.homology_dims:
            dim_sum = 0
            for L in landscape_norms:
                dim_sum += (L[dim] - landscape_norm_means[dim]) ** 2
            dim_sums += dim_sum
        return dim_sums / N

    @staticmethod
    def _compute_landscape_norm_means(landscapes: List[Dict[int, np.array]], return_norms: bool = False):
        """
        We expect each landscape in the input list to be of the shape returned by average_landscape, i.e., Dict[int, np.array].
        :param landscapes:
        :return:
        """
        N = len(landscapes)
        max_homology_dimension = max(landscapes[0].keys())
        landscape_norms = list(map(Presto._compute_landscape_norm, landscapes))
        landscape_norm_means = {i: sum(L[i] for L in landscape_norms) / N for i in range(max_homology_dimension + 1)}
        if return_norms:
            return landscape_norm_means, landscape_norms
        else:
            return landscape_norm_means

    @staticmethod
    def _compute_landscape_norm(landscape: Dict[int, np.array]) -> Dict[int, float]:
        return {
            k: np.linalg.norm(v) for k, v in landscape.items()
        }

    def normalize_space(self, X):
        """
        Normalize a space based on an approximate diameter.

        Parameters:
        - X : np.ndarray
            The input space to be normalized.

        Returns:
        - normalized_X : np.ndarray
            The normalized space.
        """
        subset = [self.rng.choice(len(X))]
        for _ in range(self.diameter_iterations - 1):
            distances = cdist([X[subset[-1]]], X).ravel()
            new_point = np.argmax(distances)
            subset.append(new_point)
        pairwise_distances = cdist(X[subset], X[subset])
        diameter = np.max(pairwise_distances)
        return X / diameter

    def generate_projections(self, X, n_projections):
        """
        Generate random projections of the input data.

        Parameters:
        - X : np.ndarray
            The input data.
        - n_projections : int
            The number of random projections.

        Returns:
        - random_projections : list
            List of random projections.
        """
        random_projections = []
        for _ in range(n_projections):
            P_X = self.P.fit_transform(X)
            random_projections.append(P_X)
        return random_projections

    def generate_landscapes(self, projections: List[np.array]) -> Dict[int, List[np.array]]:
        """
        Generate persistence landscapes from a list of projections.

        Parameters:
        - projections : list
            List of projections.

        Returns:
        - landscapes : dict
            Dictionary containing persistence landscapes for each homology dimension.
        """
        landscapes = {dim: list() for dim in self.homology_dims}
        for X_ in projections:
            alpha_complex = gd.AlphaComplex(points=X_).create_simplex_tree()
            # Compute Peristence
            alpha_complex.persistence()
            for dim in self.homology_dims:
                persistence_pairs = mask_infinities(alpha_complex.persistence_intervals_in_dimension(
                    dim
                ))
                landscapes[dim].append(self.LS.fit_transform([persistence_pairs]))
        return landscapes

    def set_projections(self, projectionsX, projectionsY):
        self._projectionsX = projectionsX
        self._projectionsY = projectionsY

    def set_landscapes(self, landscapeX, landscapeY):
        self._landscapeX = landscapeX
        self._landscapeY = landscapeY

    def set_all_landscapes(self, all_landscapesX, all_landscapesY):
        self._all_landscapesX = all_landscapesX
        self._all_landscapesY = all_landscapesY

    @staticmethod
    def average_landscape(L: Dict[int, List[np.array]]) -> Dict[int, np.array]:
        """
        Average persistence landscapes over multiple projections.

        Parameters:
        - L : dict
            Dictionary containing persistence landscapes for each homology dimension.

        Returns:
        - avg : dict
            Dictionary containing the average persistence landscape for each homology dimension.
        """
        avg = {}
        for dim, landscapes in L.items():
            sum_ = np.zeros_like(landscapes[0])
            N = len(landscapes)
            for l in landscapes:
                sum_ += l
            avg[dim] = sum_.__truediv__(N)
        return avg


def mask_infinities(array):
    return array[array[:, 1] < np.inf]
