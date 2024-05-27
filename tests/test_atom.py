from presto.compare import Atom
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
        self.X = self.rng.random(size=(100, 10))
        self.Y = self.rng.random(size=(100, 10))
        self.Z = self.rng.random(size=(100, 10))
        self.data = [self.X, self.Y, self.Z]
        self.parallelize = False
        self.atom_ = Atom(self.data, params={})
        # self.MMS is the correct answer from the sequential implementation
        self.MMS = np.array(
            [
                [0.0, 0.20628384, 0.1547974],
                [0.20628384, 0.0, 0.09392947],
                [0.1547974, 0.09392947, 0.0],
            ]
        )

    def test_compute_MMS_sequential(self):
        self.atom_.compute_MMS(
            score_type="aggregate",
            n_components=self.n_components,
            normalize=self.normalize,
            max_homology_dim=self.max_homology_dim,
            resolution=self.resolution,
            normalization_approx_iterations=self.normalization_approx_iterations,
            parallelize=False,
        )
        self.assertSequenceEqual(self.MMS.shape, self.atom_.MMS.shape)
        self.assertTrue(np.allclose(self.MMS, self.atom_.MMS))
        # Ensure Symmetry
        self.assertTrue(np.allclose(self.atom_.MMS[0, :], self.atom_.MMS[:, 0]))

    def test_compute_MMS_parallel(self):
        self.atom_.compute_MMS(
            score_type="aggregate",
            n_components=self.n_components,
            normalize=self.normalize,
            max_homology_dim=self.max_homology_dim,
            resolution=self.resolution,
            normalization_approx_iterations=self.normalization_approx_iterations,
            parallelize=True,
        )
        self.assertSequenceEqual(self.MMS.shape, self.atom_.MMS.shape)
        print(self.MMS)
        print(self.atom_.MMS)
        self.assertTrue(np.allclose(self.MMS, self.atom_.MMS))
        # Ensure Symmetry
        self.assertTrue(np.allclose(self.atom_.MMS[0, :], self.atom_.MMS[:, 0]))

    def test_cluster(self):
        pass

    def test_compute_set_cover(self):
        pass

    def tearDown(self) -> None:
        pass
