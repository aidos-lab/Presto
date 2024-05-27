import pytest
from presto.compare import Presto
from sklearn.random_projection import GaussianRandomProjection as Gauss
from sklearn.decomposition import PCA
import numpy as np


def test_homology_dims(presto):
    assert [0, 1, 2] == presto.homology_dims


def test_generate_projections(
    presto, random_projector, X, n_projections, n_components, seed
):
    projectionsX = presto._generate_projections(
        random_projector,
        X,
        dim=n_components,
        n_projections=n_projections,
        seed=seed,
    )
    assert n_projections == len(projectionsX)
    assert X.shape[0] == projectionsX[0].shape[0]
    assert n_components == projectionsX[0].shape[1]

    # Test Determinism
    for test_seed in [42, 68, 172]:
        projection1 = presto._generate_projections(Gauss, X, 10, 2, test_seed)
        projection2 = presto._generate_projections(Gauss, X, 10, 2, test_seed)
        for i, P1 in enumerate(projection1):
            P2 = projection2[i]
            assert np.allclose(P1, P2)


def test_generate_landscapes(
    presto, projectionsX, max_homology_dim, n_projections
):
    test_landscapesX = presto._generate_landscapes(
        projectionsX, homology_dims=range(0, max_homology_dim + 1)
    )
    assert n_projections == len(test_landscapesX)
    assert presto.max_homology_dim == len(test_landscapesX[0]) - 1


def test_average_landscape(presto, landscapesX, max_homology_dim):
    landscapeX = presto._average_landscape(landscapesX)
    assert max_homology_dim == len(landscapeX) - 1


def test_compute_presto_scores(
    presto, landscapeX, landscapeX2, landscapeY, X, n_projections
):
    scores_different = presto.compute_presto_scores(
        landscapeX, landscapeY, score_type="aggregate"
    )
    assert scores_different != 0
    scores_same = presto.compute_presto_scores(
        landscapeX, landscapeX2, score_type="aggregate"
    )
    assert scores_same == 0
    assert 0 == presto.fit_transform(
        X,
        X,
        score_type="aggregate",
        n_projections=n_projections,
    )


def test_compute_landscape_norm(toy_landscape, toy_landscape_norm):
    computed_norm = Presto._compute_landscape_norm(
        toy_landscape, score_type="separate"
    )
    assert set(toy_landscape_norm.keys()) == set(computed_norm.keys())
    for key in toy_landscape_norm.keys():
        assert (
            pytest.approx(toy_landscape_norm[key], 0.000000000001)
            == computed_norm[key]
        )


def test_compute_landscape_norm_means(
    toy_landscape, toy_landscape2, toy_landscape_norm, toy_landscape_norm2
):
    dict_mean = {
        i: (toy_landscape_norm[i] + toy_landscape_norm2[i]) / 2
        for i in toy_landscape_norm.keys()
    }
    computed_means = Presto._compute_landscape_norm_means(
        [toy_landscape, toy_landscape2],
    )
    assert set(dict_mean.keys()) == set(computed_means.keys())
    for key in dict_mean.keys():
        assert (
            pytest.approx(dict_mean[key], 0.000000000001) == computed_means[key]
        )


def test_compute_presto_variance(presto, toy_landscapes):
    landscape_norm_means, landscape_norms = (
        Presto._compute_landscape_norm_means(toy_landscapes, return_norms=True)
    )
    expected = sum(
        [
            sum(
                [
                    (L[dim] - landscape_norm_means[dim]) ** 2
                    for L in landscape_norms
                ]
            )
            for dim in presto.homology_dims
        ]
    ) / len(landscape_norms)
    assert expected == presto.compute_presto_variance(toy_landscapes)


def test_compute_presto_coordinate_sensitivity(presto, toy_landscapes):
    expected = np.sqrt(presto.compute_presto_variance(toy_landscapes))
    assert expected == presto.compute_presto_coordinate_sensitivity(
        toy_landscapes
    )


def test_compute_local_presto_sensitivity(
    presto, toy_landscapes, toy_landscapes2
):
    v1 = presto.compute_presto_variance(toy_landscapes)
    v2 = presto.compute_presto_variance(toy_landscapes2)
    expected = np.sqrt((v1 + v2) / 2)
    assert expected == presto.compute_local_presto_sensitivity(
        [toy_landscapes, toy_landscapes2]
    )


def test_compute_global_presto_sensitivity(
    presto, toy_landscapes, toy_landscapes2
):
    prestosensitivity_1 = presto.compute_local_presto_sensitivity(
        [toy_landscapes, toy_landscapes2]
    )
    prestosensitivity_2 = presto.compute_local_presto_sensitivity(
        [toy_landscapes2, toy_landscapes]
    )
    expected = (prestosensitivity_1 + prestosensitivity_2) / 2
    assert expected == presto.compute_global_presto_sensitivity(
        [[toy_landscapes, toy_landscapes2]]
    )
