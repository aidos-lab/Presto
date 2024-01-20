"PESTO: Pairwise Embedding Score using Topological Overlays"

import gudhi as gd
import numpy as np
from gudhi.representations import Landscape
from scipy.spatial.distance import cdist
from sklearn.random_projection import GaussianRandomProjection as Gauss


class PESTO:
    def __init__(
            self,
            projector=Gauss,
            n_components=2,
            normalize: bool = False,
            max_homology_dim: int = 1,
            resolution: int = 100,
            normalization_approx_iterations: int = 1000,
            seed: int = 42
    ) -> None:
        """
        Initialize `PESTO` an object for efficienctly computing the structural similarity of embeddings.


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
        self.P = projector(n_components=self.projection_dimension, random_state=self.seed)

        # Set Normalization parameters
        self.normalize = normalize
        self.diameter_iterations = normalization_approx_iterations

        # Set Topological parameters
        self.homology_dims = list(range(0, max_homology_dim + 1))
        self.landscape_resolution = resolution
        self.LS = Landscape(resolution=self.landscape_resolution, keep_endpoints=False)
        self._landscapeX = None
        self._landscapeY = None
        self._pestos = None

    def fit(
            self,
            X,
            Y,
            N=15,
    ):
        """
        Fit a topological descriptor to embeddings X & Y.

        This function computes `N` random projections of X and Y
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
        """

        if self.normalize:
            X, Y = self._normalize_space(X), self._normalize_space(Y)

        # Project
        self._projectionsX = self._generate_projections(X, N)
        self._projectionsY = self._generate_projections(Y, N)

        # Fit Landscapes
        self._all_landscapesX = self._generate_landscapes(self._projectionsX)
        self._all_landscapesY = self._generate_landscapes(self._projectionsY)

        # Average Landscapes
        self._landscapeX = self._average_landscape(self._all_landscapesX)
        self._landscapeY = self._average_landscape(self._all_landscapesY)

    def fit_transform(self, X, Y, N: int = 15, score_type: str = "aggregate"):
        """
        Fit a topological descriptor and compute the PESTO score.

        Parameters:
        - X : array-like or pd.DataFrame, shape (n_samples, n_features), default=None
            Ignored. Placeholder for compatibility.
        - Y : array-like or pd.DataFrame, shape (n_samples, n_features)
            The second set of embeddings.
        - N : int, optional
            The number of random projections. Default is 100.
        - score_type: str, optional
            Which type of PESTO score to return. Options are:
            - "aggregate" (sum normed distances across all dimensions)
            - "average" (average distance across al dimensions)
            - "separate" (list of distances by dimension)
            Default is "aggregate".

        Returns:
        - pesto_score : float
            The computed PESTO score representing the distance between the topological descriptors of X and Y.
        """
        self._set_pestos(X, Y, N)
        if score_type == "aggregate":
            return sum(self._pestos.values())
        elif score_type == "average":
            return sum(self._pestos.values()) / len(self._pestos.values())
        elif score_type == "separate":
            return self._pestos
        else:
            raise NotImplementedError(score_type)

    def _set_pestos(self, X, Y, N: int = 15):
        if self._landscapeX is None or self._landscapeY is None:
            self.fit(X, Y, N)

        assert self._landscapeX is not None
        pestos = dict()
        for dim in self.homology_dims:
            lambdaX = self._landscapeX[dim]
            lambdaY = self._landscapeY[dim]
            if not np.isnan(lambdaX - lambdaY).any():
                pestos[dim] = np.linalg.norm(lambdaX - lambdaY)
        self._pestos = pestos

    def _normalize_space(self, X):
        """
        Normalize a space based on an approximate diameter.

        Parameters:
        - X : np.ndarray
            The input space to be normalized.

        Returns:
        - normalized_X : np.ndarray
            The normalized space.
        """
        subset = [self.rng.random.choice(len(X))]
        for _ in range(self.diameter_iterations - 1):
            distances = cdist([X[subset[-1]]], X).ravel()
            new_point = np.argmax(distances)
            subset.append(new_point)
        pairwise_distances = cdist(X[subset], X[subset])
        diameter = np.max(pairwise_distances)
        return X / diameter

    def _generate_projections(self, X, N):
        """
        Generate random projections of the input data.

        Parameters:
        - X : np.ndarray
            The input data.
        - N : int
            The number of random projections.

        Returns:
        - random_projections : list
            List of random projections.
        """
        random_projections = []
        for _ in range(N):
            P_X = self.P.fit_transform(X)
            random_projections.append(P_X)
        return random_projections

    def _generate_landscapes(self, projections: list):
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
        # all_persistence_pairs = {dim: list() for dim in self.homology_dims}
        for X_ in projections:
            alpha_complex = gd.AlphaComplex(points=X_).create_simplex_tree()
            # Compute Peristence
            alpha_complex.persistence()
            for dim in self.homology_dims:
                persistence_pairs = mask_infinities(alpha_complex.persistence_intervals_in_dimension(
                    dim
                ))
                # all_persistence_pairs[dim].append(persistence_pairs)
                landscapes[dim].append(self.LS.fit_transform([persistence_pairs]))
        return landscapes

    def _average_landscape(self, L: dict):
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
            sum_ = landscapes[0]
            N = len(landscapes)
            for l in landscapes[1:]:
                sum_ += l
            avg[dim] = sum_.__truediv__(N)
        return avg


def mask_infinities(array):
    return array[array[:, 1] < np.inf]
