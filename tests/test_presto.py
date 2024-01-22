from unittest import TestCase
from presto import Presto
from sklearn.random_projection import GaussianRandomProjection as Gauss
import numpy as np


class PrestoTest(TestCase):
    def setUp(self) -> None:
        self.projector = Gauss
        self.n_components = 3
        self.normalize = False
        self.max_homology_dim = 2
        self.resolution = 100
        self.normalization_approx_iterations = 1000
        self.seed = 42
        self.presto_ = Presto(projector=self.projector, n_components=self.n_components, normalize=self.normalize,
                              max_homology_dim=self.max_homology_dim, resolution=self.resolution,
                              normalization_approx_iterations=self.normalization_approx_iterations, seed=self.seed)
        self.n_projections = 15
        self.X = self.presto_.rng.random(size=(100, 1000))
        self._projectionsX = self.presto_.generate_projections(self.X, self.n_projections)
        self._landscapesX = self.presto_.generate_landscapes(self._projectionsX)
        self._landscapeX = self.presto_.average_landscape(self._landscapesX)
        self.Y = self.presto_.rng.random(size=(200, 500))
        self._projectionsY = self.presto_.generate_projections(self.Y, self.n_projections)
        self._landscapesY = self.presto_.generate_landscapes(self._projectionsY)
        self._landscapeY = self.presto_.average_landscape(self._landscapesY)
        self.toy_landscape = {0: [1, 2, 3], 1: [0.5, 0.2, 0.1], 2: [0.1, 2, -2]}
        self.toy_landscape2 = {0: [0, 1, 2], 1: [0.5, 0.2, 0.1], 2: [0.1, 2, -2]}
        self.toy_landscape_norm = {i: np.sqrt(sum(x ** 2 for x in L)) for i, L in self.toy_landscape.items()}
        self.toy_landscape_norm2 = {i: np.sqrt(sum(x ** 2 for x in L)) for i, L in self.toy_landscape2.items()}
        self.toy_landscapes = [self.toy_landscape, self.toy_landscape2]

    def test_homology_dims(self):
        self.assertListEqual([0, 1, 2], self.presto_.homology_dims)

    def test_generate_projections(self):
        self.assertEqual(self.n_projections, len(self._projectionsX))
        self.assertEqual(self.X.shape[0], self._projectionsX[0].shape[0])
        self.assertEqual(self.n_components, self._projectionsX[0].shape[1])

    def test_generate_landscapes(self):
        self.assertEqual(self.max_homology_dim, len(self._landscapesX) - 1)
        self.assertEqual(self.n_projections, len(self._landscapesX[0]))

    def test_average_landscape(self):
        self.assertEqual(self.max_homology_dim, len(self._landscapeX) - 1)

    def test_compute_presto_scores(self):
        self.presto_.compute_presto_scores(self._landscapeX, self._landscapeY)

    def test_compute_landscape_norm(self):
        self.assertDictEqual(self.toy_landscape_norm, Presto._compute_landscape_norm(self.toy_landscape))

    def test_compute_landscape_norm_means(self):
        dict_mean = {i: (self.toy_landscape_norm[i] + self.toy_landscape_norm2[i]) / 2 for i in
                     self.toy_landscape_norm.keys()}
        self.assertDictEqual(dict_mean, Presto._compute_landscape_norm_means([self.toy_landscape, self.toy_landscape2]))
        dict_mean = {i: (2 * self.toy_landscape_norm[i] + 2 * self.toy_landscape_norm2[i]) / 4 for i in
                     self.toy_landscape_norm.keys()}
        self.assertDictEqual(dict_mean,
                             Presto._compute_landscape_norm_means(self.toy_landscapes * 2))

    def test_compute_presto_variance(self):
        landscape_norm_means, landscape_norms = Presto._compute_landscape_norm_means(self.toy_landscapes,
                                                                                     return_norms=True)
        expected = sum([sum([(L[dim] - landscape_norm_means[dim]) ** 2 for L in landscape_norms]) for dim in
                        self.presto_.homology_dims]) / len(landscape_norms)
        self.assertEqual(expected, self.presto_.compute_presto_variance(self.toy_landscapes))

    def test_compute_presto_coordinate_sensitivity(self):
        pass

    def test_compute_local_presto_sensitivity(self):
        pass

    def test_compute_global_presto_sensitivity(self):
        pass

    def test_normalize_space(self):
        pass

    def test_fit(self):
        pass

    def test_fit_transform(self):
        pass

    def tearDown(self) -> None:
        pass
