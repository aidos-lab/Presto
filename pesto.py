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
        """
        # Initialize Projector
        self.projection_dimension = n_components
        self.P = projector(n_components=self.projection_dimension)

        # Set Normalization parameters
        self.normalize = normalize
        self.diameter_iterations = normalization_approx_iterations

        # Set Topological parameters
        self.homology_dims = range(0, max_homology_dim + 1)
        self.landscape_resolution = resolution
        self.LS = Landscape(resolution=self.landscape_resolution)
        self._landscapeX = None
        self._landscapeY = None

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

    def fit_transform(self, X, Y, N=15):
        """
        Fit a topological descriptor and compute the PESTO score.

        Parameters:
        - X : array-like or pd.DataFrame, shape (n_samples, n_features), default=None
            Ignored. Placeholder for compatibility.
        - Y : array-like or pd.DataFrame, shape (n_samples, n_features)
            The second set of embeddings.
        - N : int, optional
            The number of random projections. Default is 100.

        Returns:
        - pesto_score : float
            The computed PESTO score representing the distance between the topological descriptors of X and Y.
        """
        if not self._landscapeX:
            self.fit(X, Y, N)

        assert self._landscapeX is not None
        pesto_score = 0
        for dim in self.homology_dims:
            lambdaX = self._landscapeX[dim]
            lambdaY = self._landscapeY[dim]
            if not np.isnan(lambdaX - lambdaY).any():
                pesto_score += np.linalg.norm(lambdaX - lambdaY)
        return pesto_score

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
        subset = [np.random.choice(len(X))]
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
        landscapes = {}
        for dim in self.homology_dims:
            for X_ in projections:
                alpha_complex = gd.AlphaComplex(points=X_).create_simplex_tree()
                # Compute Peristence
                alpha_complex.persistence()
                persistence_pairs = alpha_complex.persistence_intervals_in_dimension(
                    dim
                )
                # TODO debug it looks like you're overwriting, rather than appending to a list – that could explain the volatility I observed
                # Also, I get only nans for 0th dimension and only 0s for 2nd dimension, so only 1st dimension appears to "work"/be meaningful?
                landscapes[dim] = self.LS.fit_transform([persistence_pairs])
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
