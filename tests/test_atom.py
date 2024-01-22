from atom import Atom
from unittest import TestCase


class AtomTest(TestCase):
    def setUp(self) -> None:
        self.data = list()
        self.n_components = 2
        self.normalize = False
        self.max_homology_dim = 1
        self.resolution = 100
        self.normalization_approx_iterations = 1000
        self.seed = 42
        self.atom = Atom(self.data, n_components=self.n_components, normalize=self.normalize,
                         max_homology_dim=self.max_homology_dim, resolution=self.resolution,
                         normalization_approx_iterations=self.normalization_approx_iterations)

    def test_compute_MMS(self):
        pass

    def test_cluster(self):
        pass

    def test_compute_set_cover(self):
        pass

    def tearDown(self) -> None:
        pass