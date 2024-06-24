import pytest
import tempfile
import os
from presto import Atom
import numpy as np
import networkx as nx


def test_atom_init(data, tmp_data_files):
    with pytest.raises(TypeError):
        Atom(data=True)

    atom = Atom(data=tmp_data_files)

    assert isinstance(atom.data, list)
    assert len(atom.data) == 3

    for i, X in enumerate(atom.data):
        assert isinstance(X, np.ndarray)
        assert np.array_equal(X, data[i])

    atom = Atom(data=data)
    assert isinstance(atom.data, list)
    assert len(atom.data) == 3

    assert atom.MMS is None
    assert atom.seed == 42


def test_mms_getter_setter(data, MMS, tmp_MMS_file):

    atom1 = Atom(data=data)
    # Set MMS to None
    with pytest.raises(TypeError):
        atom1.MMS = None
        atom1.MMS = 10

    # Invalid shape
    with pytest.raises(AssertionError):
        atom1.MMS = np.array([1, 2, 3])

    # Not symmetric
    with pytest.raises(AssertionError):
        atom1.MMS = np.array([[1, 2, 3], [3, 4, 8]])

    # Set MMS to np.array
    atom1.MMS = MMS
    assert np.allclose(atom1.MMS, MMS)

    # Set MMS based on file
    atom1.MMS = tmp_MMS_file
    assert np.allclose(atom1.MMS, MMS)

    computer = Atom(data=data)

    assert computer.MMS is None
    computer.compute_mms()
    assert computer.MMS is not None


def test_mms_save_load(atom, MMS):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_path = os.path.join(tempdir, "mms_test.pkl")
        atom.MMS = MMS
        atom.save_mms(temp_path)
        assert os.path.exists(temp_path), "Failed to save MMS."
        loaded_mms = atom._load_data(temp_path)
        assert np.array_equal(MMS, loaded_mms), "Loaded MMS does not match."
        assert np.array_equal(
            atom.MMS, loaded_mms
        ), "Loaded MMS does not match."


def test_validate_distance_matrix_valid(atom):
    M = np.zeros((4, 4))
    validated_matrix = atom.validate_distance_matrix(M, shape=(4, 4))
    assert np.array_equal(
        M, validated_matrix
    ), "Valid distance matrix validation failed."


def test_validate_distance_matrix_invalid_shape(atom):
    M = np.zeros((3, 3))
    with pytest.raises(AssertionError):
        atom.validate_distance_matrix(M, shape=4)


def test_validate_distance_matrix_invalid_type(atom):
    M = np.zeros((4, 4), dtype=int)
    with pytest.raises(AssertionError):
        atom.validate_distance_matrix(M, shape=4)


def test_validate_distance_matrix_non_symmetric(atom):
    M = np.ones((4, 4))
    with pytest.raises(AssertionError):
        atom.validate_distance_matrix(M, shape=4)


def test_compute_mms_sequential(
    atom,
    MMS,
    n_components,
    max_homology_dim,
    resolution,
    normalization_approx_iterations,
):
    atom.compute_mms(
        score_type="aggregate",
        n_components=n_components,
        normalize=False,
        max_homology_dim=max_homology_dim,
        resolution=resolution,
        normalization_approx_iterations=normalization_approx_iterations,
        parallelize=False,
    )
    # Check if the shapes are the same
    assert (
        MMS.shape == atom.MMS.shape
    ), f"Expected shape {MMS.shape}, but got {atom.MMS.shape}"

    # Check if the contents are close enough
    assert np.allclose(
        MMS, atom.MMS
    ), "Arrays MMS and atom.MMS are not sufficiently close"

    # Ensure symmetry
    assert np.allclose(
        atom.MMS[0, :], atom.MMS[:, 0]
    ), "Matrix atom.MMS is not symmetric"


def test_compute_mms_parallel(
    atom,
    MMS,
    n_components,
    max_homology_dim,
    resolution,
    normalization_approx_iterations,
):
    atom.compute_mms(
        score_type="aggregate",
        n_components=n_components,
        normalize=False,
        max_homology_dim=max_homology_dim,
        resolution=resolution,
        normalization_approx_iterations=normalization_approx_iterations,
        parallelize=True,
    )
    assert np.array_equal(MMS.shape, atom.MMS.shape)
    assert np.allclose(MMS, atom.MMS)
    # Ensure Symmetry
    assert np.allclose(atom.MMS[0, :], atom.MMS[:, 0])


def test_cluster(atom, epsilon, linkage):
    clustering = atom.cluster(epsilon, linkage)
    assert clustering.linkage == linkage
    assert atom.epsilon == epsilon
    assert clustering.labels_.shape[0] == atom.multiverse_size


def test_set_cover_graph(
    atom,
    MMS1,
):
    atom.multiverse_size = len(MMS1)
    for epsilon in [0.1, 0.5, 1.0]:
        atom.set_cover_epsilon = epsilon
        G = atom._set_cover_graph(MMS1, epsilon)
        # Bipartite graph
        assert len(G.nodes()) == atom.multiverse_size * 2
        assert np.array_equal(
            list(G.nodes()),
            [(i, 0) for i in range(atom.multiverse_size)]
            + [(i, 1) for i in range(atom.multiverse_size)],
        )

    # Only Trivial Edges
    G = atom._set_cover_graph(MMS1, 0)
    assert len(G.edges()) == 4
    assert np.array_equal(
        list(G.edges()),
        [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
            ((3, 0), (3, 1)),
        ],
    )

    # Some Non-Trivial Edges
    G = atom._set_cover_graph(MMS1, 0.1)
    assert len(G.edges()) == 8
    assert set(list(G.edges())) == set(
        [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
            ((3, 0), (3, 1)),
            ((0, 0), (1, 1)),
            ((1, 0), (0, 1)),
            ((2, 0), (3, 1)),
            ((3, 0), (2, 1)),
        ],
    )

    # All edges
    G = atom._set_cover_graph(MMS1, 1)
    assert len(G.edges()) == 16
    assert set(list(G.edges())) == set(
        [
            ((0, 0), (0, 1)),
            ((1, 0), (1, 1)),
            ((2, 0), (2, 1)),
            ((3, 0), (3, 1)),
            ((0, 0), (1, 1)),
            ((1, 0), (0, 1)),
            ((2, 0), (3, 1)),
            ((3, 0), (2, 1)),
            ((0, 0), (2, 1)),
            ((2, 0), (0, 1)),
            ((1, 0), (3, 1)),
            ((3, 0), (1, 1)),
            ((0, 0), (3, 1)),
            ((3, 0), (0, 1)),
            ((1, 0), (2, 1)),
            ((2, 0), (1, 1)),
        ],
    )


def test_compute_set_cover(MMS1):
    atom = Atom(data=[np.random.rand(10, 10) for _ in range(4)])
    G1 = nx.DiGraph()
    G1.add_edges_from([((0, 0), (1, 1)), ((0, 0), (2, 1)), ((1, 0), (3, 1))])
    expected_result1 = {0: [1, 2], 1: [3]}
    assert atom._compute_set_cover(G1) == expected_result1

    G2 = nx.DiGraph()
    expected_result2 = {}
    assert atom._compute_set_cover(G2) == expected_result2

    G3 = nx.DiGraph()
    G3.add_edges_from([((0, 0), (1, 1)), ((2, 0), (3, 1))])
    expected_result3 = {0: [1], 2: [3]}
    assert atom._compute_set_cover(G3) == expected_result3

    G4 = nx.DiGraph()
    G4.add_edges_from(
        [
            ((0, 0), (1, 1)),
            ((0, 0), (2, 1)),
            ((1, 0), (2, 1)),
            ((1, 0), (3, 1)),
            ((2, 0), (4, 1)),
            ((3, 0), (4, 1)),
        ]
    )
    expected_result4 = {0: [1, 2], 1: [2, 3], 3: [4]}
    assert atom._compute_set_cover(G4) == expected_result4

    G5 = nx.DiGraph()
    G5.add_edge((0, 0), (1, 1))
    expected_result5 = {0: [1]}
    assert atom._compute_set_cover(G5) == expected_result5

    atom.MMS = MMS1
    nontrivial_cover = atom.compute_set_cover(epsilon=0.1)
    assert nontrivial_cover == {0: [0, 1], 2: [2, 3]}
    trivial_cover = atom.compute_set_cover(epsilon=0)
    assert trivial_cover == {0: [0], 1: [1], 2: [2], 3: [3]}
    total_cover = atom.compute_set_cover(epsilon=0.5)
    assert total_cover == {0: [0, 1, 2, 3]}
