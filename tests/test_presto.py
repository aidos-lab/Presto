from unittest import TestCase
from presto import Presto
from sklearn.random_projection import GaussianRandomProjection as Gauss


class PrestoTest(TestCase):
    def setUp(self) -> None:
        self.projector = Gauss
        self.n_components = 3
        self.normalize = False
        self.max_homology_dim = 2
        self.resolution = 100
        self.normalization_approx_iterations = 1000
        self.seed = 42
        self.pesto = Presto(projector=self.projector, n_components=self.n_components, normalize=self.normalize,
                            max_homology_dim=self.max_homology_dim, resolution=self.resolution,
                            normalization_approx_iterations=self.normalization_approx_iterations, seed=self.seed)
        self.n_projections = 15
        self.X = self.pesto.rng.random(size=(100, 1000))
        self._projectionsX = self.pesto._generate_projections(self.X, self.n_projections)
        self._landscapesX = self.pesto._generate_landscapes(self._projectionsX)
        self._landscapeX = self.pesto._average_landscape(self._landscapesX)
        self.Y = self.pesto.rng.random(size=(200, 500))

    def test_homology_dims(self):
        self.assertListEqual([0, 1, 2], self.pesto.homology_dims)

    def test_generate_projections(self):
        self.assertEqual(self.n_projections, len(self._projectionsX))
        self.assertEqual(self.X.shape[0], self._projectionsX[0].shape[0])
        self.assertEqual(self.n_components, self._projectionsX[0].shape[1])

    def test_generate_landscapes(self):
        self.assertEqual(self.max_homology_dim, len(self._landscapesX) - 1)
        self.assertEqual(self.n_projections, len(self._landscapesX[0]))

    def test_average_landscape(self):
        self.assertEqual(self.max_homology_dim, len(self._landscapeX) - 1)

    def test_set_pestos(self):
        self.pesto._set_pestos(self.X, self.Y, self.n_projections)

    def test_normalize_space(self):
        pass

    def test_fit(self):
        pass

    def test_fit_transform(self):
        pass

    def tearDown(self) -> None:
        pass
