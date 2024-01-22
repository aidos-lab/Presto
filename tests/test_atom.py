from atom import Atom
from unittest import TestCase
import numpy as np


class AtomTest(TestCase):
    def setUp(self) -> None:
        self.n_components = 2
        self.normalize = False
        self.max_homology_dim = 1
        self.resolution = 100
        self.normalization_approx_iterations = 1000
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.X = self.rng.random(size=(100, 1000))
        self.Y = self.rng.random(size=(100, 1000))
        self.Z = self.rng.random(size=(100, 1000))
        self.data = [self.X, self.Y, self.Z]
        self.parallelize = False
        self.atom_ = Atom(self.data, n_components=self.n_components, normalize=self.normalize,
                          max_homology_dim=self.max_homology_dim, resolution=self.resolution,
                          normalization_approx_iterations=self.normalization_approx_iterations)
        self.MMS = np.array([[0., 28.27532526, 25.71660789],
                             [28.27532526, 0., 21.28421828],
                             [25.71660789, 21.28421828, 0.]])

    def test_compute_MMS_sequential(self):
        self.atom_.compute_MMS(n_projections=15, score_type="aggregate", parallelize=False)
        self.assertSequenceEqual(self.MMS.shape, self.atom_.MMS.shape)
        self.assertTrue(np.allclose(self.MMS, self.atom_.MMS))

    def test_compute_MMS_parallel(self):
        self.atom_.compute_MMS(n_projections=15, score_type="aggregate", parallelize=True)
        self.assertSequenceEqual(self.MMS.shape, self.atom_.MMS.shape)
        # we cannot use np.allclose on the entire array b/c the randomness will diverge
        # b/c we need independent presto objects
        self.assertTrue(np.allclose(self.MMS[0, 1], self.atom_.MMS[0, 1]))

    def test_cluster(self):
        pass

    def test_compute_set_cover(self):
        pass

    def tearDown(self) -> None:
        pass
