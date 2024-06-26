import pytest
from presto import Presto, Atom
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection as Gauss
from gudhi.representations import Landscape
import tempfile
import numpy as np


@pytest.fixture
def deterministic_projector():
    return PCA


@pytest.fixture
def random_projector():
    return Gauss


@pytest.fixture
def n_components():
    return 3


@pytest.fixture
def max_homology_dim():
    return 2


@pytest.fixture
def LS(resolution):
    return Landscape(resolution=resolution, keep_endpoints=False)


@pytest.fixture
def resolution():
    return 100


@pytest.fixture
def normalization_approx_iterations():
    return 1000


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def presto(random_projector, max_homology_dim, resolution):
    return Presto(
        projector=random_projector,
        max_homology_dim=max_homology_dim,
        resolution=resolution,
    )


@pytest.fixture
def n_projections():
    return 3


@pytest.fixture
def rng(seed):
    return np.random.default_rng(seed)


@pytest.fixture
def X(rng):
    return rng.random(size=(100, 10))


@pytest.fixture
def projectionsX(
    presto, random_projector, X, n_projections, n_components, seed
):
    return presto._generate_projections(
        random_projector,
        X,
        n_projections=n_projections,
        dim=n_components,
        seed=seed,
    )


@pytest.fixture
def projectionsX2(
    presto, random_projector, X, n_projections, n_components, seed
):
    return presto._generate_projections(
        random_projector,
        X,
        n_projections=n_projections,
        dim=n_components,
        seed=seed,
    )


@pytest.fixture
def landscapesX(presto, projectionsX, LS, max_homology_dim):
    return presto._generate_landscapes(
        projectionsX,
        LS,
        range(0, max_homology_dim + 1),
    )


@pytest.fixture
def landscapesX2(presto, projectionsX2, LS, max_homology_dim):
    return presto._generate_landscapes(
        projectionsX2, LS, range(0, max_homology_dim + 1)
    )


@pytest.fixture
def landscapeX(presto, landscapesX):
    return presto._average_landscape(landscapesX)


@pytest.fixture
def landscapeX2(presto, landscapesX2):
    return presto._average_landscape(landscapesX2)


@pytest.fixture
def Y(rng):
    return rng.random(size=(200, 8))


@pytest.fixture
def projectionsY(
    presto, random_projector, Y, n_projections, n_components, seed
):
    return presto._generate_projections(
        random_projector,
        Y,
        n_projections=n_projections,
        dim=n_components,
        seed=seed,
    )


@pytest.fixture
def landscapesY(presto, projectionsY, LS, max_homology_dim):
    return presto._generate_landscapes(
        projections=projectionsY,
        LS=LS,
        homology_dims=range(0, max_homology_dim + 1),
    )


@pytest.fixture
def landscapeY(presto, landscapesY):
    return presto._average_landscape(landscapesY)


@pytest.fixture
def toy_landscape():
    return {
        0: [1, 2, 3],
        1: [0.5, 0.2, 0.1],
        2: [0.1, 2, -2],
    }


@pytest.fixture
def toy_landscape2():
    return {
        0: [0, 1, 2],
        1: [0.5, 0.2, 0.1],
        2: [0.1, 2, -2],
    }


@pytest.fixture
def toy_landscape_norm(toy_landscape):
    return {i: np.sqrt(sum(x**2 for x in L)) for i, L in toy_landscape.items()}


@pytest.fixture
def toy_landscape_norm2(toy_landscape2):
    return {i: np.sqrt(sum(x**2 for x in L)) for i, L in toy_landscape2.items()}


@pytest.fixture
def toy_landscapes(toy_landscape, toy_landscape2):
    return [toy_landscape, toy_landscape2]


@pytest.fixture
def toy_landscapes2():
    return [
        {0: [1, 2, 3], 1: [0.25, 0.2, 1.1], 2: [1.1, 2, -2]},
        {0: [1, 1, 1], 1: [0.15, 0.72, -0.91], 2: [2.1, 2, -2]},
        {0: [1, 0, 0], 1: [0.5, 2.2, 0.1], 2: [-0.1, 0, -2]},
    ]


@pytest.fixture
def data(rng):
    X = rng.random(size=(100, 10))
    Y = rng.random(size=(100, 10))
    Z = rng.random(size=(100, 10))
    return [X, Y, Z]


@pytest.fixture
def atom(data):
    return Atom(data)


@pytest.fixture
def MMS():
    return np.array(
        [
            [0.0, 0.297733, 0.25940901],
            [0.297733, 0.0, 0.2136469],
            [0.25940901, 0.2136469, 0.0],
        ]
    )


@pytest.fixture
def linkage():
    return "complete"


@pytest.fixture
def epsilon():
    return 0.5


@pytest.fixture
def MMS1():
    return np.array(
        [
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.8, 0.6],
            [0.2, 0.8, 0, 0.1],
            [0.3, 0.6, 0.1, 0],
        ]
    )


@pytest.fixture
def tmp_data_files(data):
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, X in enumerate(data):
            pkl_path = os.path.join(tmpdirname, f"X{i}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(X, f)
        yield tmpdirname


@pytest.fixture
def tmp_MMS_file(MMS):
    with tempfile.TemporaryDirectory() as tmpdirname:
        pkl_path = os.path.join(tmpdirname, f"mms.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(MMS, f)
        yield pkl_path
