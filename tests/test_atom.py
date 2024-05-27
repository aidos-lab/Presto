import pytest
from presto import Atom
import numpy as np
import networkx as nx


def test_compute_MMS_sequential(
    atom,
    MMS,
    n_components,
    max_homology_dim,
    resolution,
    normalization_approx_iterations,
):
    atom.compute_MMS(
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


def test_compute_MMS_parallel(
    atom,
    MMS,
    n_components,
    max_homology_dim,
    resolution,
    normalization_approx_iterations,
):
    atom.compute_MMS(
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


def test_compute_set_cover(atom, MMS1):
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

    atom.set_mms(MMS1)
    nontrivial_cover = atom.compute_set_cover(epsilon=0.1)
    assert nontrivial_cover == {0: [0, 1], 2: [2, 3]}
    trivial_cover = atom.compute_set_cover(epsilon=0)
    assert trivial_cover == {0: [0], 1: [1], 2: [2], 3: [3]}
    total_cover = atom.compute_set_cover(epsilon=0.5)
    assert total_cover == {0: [0, 1, 2, 3]}
