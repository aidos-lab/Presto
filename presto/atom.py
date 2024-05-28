import pickle

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import itertools
import networkx as nx
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import coo_array
from presto.presto import Presto
from sklearn.decomposition import PCA


class Atom:
    """
    The `Atom` class: built for Approximate Topological Operations in the Multiverse.


    Atom allows you to compute presto distances between many embeddings that are generated during multiverse analyses. This allows users to generate multiverse metric spaces (MMS) and perform operations to compare sets of latent representations such as clustering and set cover.

    Parameters
    ----------
    data : list
        The list of embeddings representing the Atom.
    seed : int, optional
        The seed value for random number generation, by default 42.

    Attributes
    ----------
    data : list
        The list of embeddings representing the Atom.
    multiverse_size : int
        The number of embeddings in the Atom.
    MMS : ndarray or None
        The multiverse metric space (MMS) matrix.
    seed : int
        The seed value for random number generation.

    Methods
    -------
    compute_MMS(projector=PCA, n_projections=1, score_type='aggregate', n_components=2, normalize=False, max_homology_dim=1, resolution=100, normalization_approx_iterations=1000, parallelize=True)
        Compute a multiverse metric space (MMS) based on the `presto` score between embeddings.
    save_mms(path)
        Save the MMS matrix to a file.
    load_mms(path)
        Load the MMS matrix from a file.
    set_mms(MMS)
        Set the MMS matrix.
    cluster(epsilon, linkage='complete')
        Perform clustering on the Atom using the MMS matrix.
    compute_set_cover(epsilon)
        Compute a set of representatives for the embeddings in the Atom.


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> from presto import Atom
    >>> np.random.seed(0)
    >>> data = [np.random(10, 10) for _ in range(3)]
    >>> atom = Atom(data)
    >>> atom.compute_MMS()
    >>> atom.MMS
    array([[0.        , 0.45230579, 0.8491459 ],
       [0.45230579, 0.        , 0.62582007],
       [0.8491459 , 0.62582007, 0.        ]])

    """

    def __init__(
        self,
        data: list,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.multiverse_size = len(data)
        self.MMS = None
        self.seed = seed

    def compute_MMS(
        self,
        projector=PCA,
        n_projections: int = 1,
        score_type: str = "aggregate",
        n_components: int = 2,
        normalize: bool = False,
        max_homology_dim: int = 1,
        resolution: int = 100,
        normalization_approx_iterations: int = 1000,
        parallelize=True,
    ):
        """
        Compute a multiverse metric space (MMS).

        Parameters:
        ----------
        projector : class, optional
            The class of the projection method to use. Default is PCA.
        n_projections : int, optional
            The number of projections to use for computing the Presto score. Default is 1.
        score_type : str, optional
            The type of score to compute. Must be either "aggregate" or "average". Default is "aggregate".
        n_components : int, optional
            The number of components to keep in the projected space. Default is 2.
        normalize : bool, optional
            Whether to normalize the data before computing the Presto score. Default is False.
        max_homology_dim : int, optional
            The maximum homology dimension to consider when computing the Presto score. Default is 1.
        resolution : int, optional
            The resolution parameter for computing the Presto score. Default is 100.
        normalization_approx_iterations : int, optional
            The number of iterations to use for the normalization approximation. Default is 1000.
        parallelize : bool, optional
            Whether to parallelize the computation. Default is True.

        Returns:
        -------
        None

        Examples:
        ---------
        >>> # Compute MMS with default parameters
        >>> atom.compute_MMS()

        >>> # Compute MMS with custom parameters
        >>> atom.compute_MMS(projector=Gauss, n_projections=100, n_components=3, score_type="average", normalize=True, max_homology_dim=2, resolution=100, normalization_approx_iterations=1000, parallelize=True)
        """
        if score_type not in ["aggregate", "average"]:
            raise NotImplementedError(score_type)

        data_indices = list(range(self.multiverse_size))
        pairs = list(itertools.combinations(data_indices, 2))
        n_pairs = len(pairs)

        def compute_distance(pair):
            i, j = pair
            X, Y = self.data[i], self.data[j]
            if np.isnan(X).any() or np.isnan(Y).any():
                return np.nan, i, j
            else:
                return (
                    Presto(
                        projector=projector,
                        max_homology_dim=max_homology_dim,
                        resolution=resolution,
                    ).fit_transform(
                        X,
                        Y,
                        n_components=n_components,
                        normalize=normalize,
                        n_projections=n_projections,
                        score_type=score_type,
                        normalization_approx_iterations=normalization_approx_iterations,
                        seed=self.seed,
                    ),
                    i,
                    j,
                )

        if parallelize:
            with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
                # scores now have the shape (data, row, col)
                scores = tqdm(
                    list(executor.map(compute_distance, pairs)),
                    total=n_pairs,
                    unit="universes",
                    desc="Computing Presto Distances",
                )
        else:
            scores = list()
            for pair in tqdm(
                pairs,
                total=n_pairs,
                unit="universes",
                desc="Computing Presto Distances",
            ):
                scores.append(compute_distance(pair))

        values = list(map(lambda tup: tup[0], scores))
        rows = list(map(lambda tup: tup[1], scores))
        cols = list(map(lambda tup: tup[2], scores))

        self.MMS = coo_array(
            (values, (rows, cols)),
            shape=(self.multiverse_size, self.multiverse_size),
        ).todense()
        self.MMS += self.MMS.T

    def save_mms(self, path: str):
        """
        Save the MMS matrix to a file.

        Parameters
        ----------
        path : str
            The path to the file where the MMS matrix will be saved.

        Returns
        -------
        None

        Examples
        --------
        >>> atom.save_mms('/path/to/mms.pkl')
        """
        with open(path, "wb") as f:
            pickle.dump(self.MMS, f)

    def load_mms(self, path: str):
        """
        Load the MMS matrix from a file.

        Parameters
        ----------
        path : str
            The path to the file where the MMS matrix is saved.

        Returns
        -------
        None

        Examples
        --------
        >>> atom.load_mms('/path/to/mms.pkl')
        """
        with open(path, "rb") as f:
            MMS = pickle.load(f)
        self.set_mms(MMS)

    def set_mms(self, MMS):
        """
        Set the MMS matrix.

        Parameters
        ----------
        MMS : ndarray
            The MMS matrix.

        Returns
        -------
        None

        Examples
        --------
        >>> atom.set_mms(my_mms)
        """
        self.MMS = MMS

    def cluster(
        self,
        epsilon,
        linkage: str = "complete",
    ) -> AgglomerativeClustering:
        """
        Perform clustering on the Atom using the MMS matrix.

        Parameters
        ----------
        epsilon : float
            The distance threshold for clustering.
        linkage : str, optional
            The linkage criterion for clustering. Default is "complete".

        Returns
        -------
        AgglomerativeClustering
            The clustering object.

        Examples
        --------
        >>> atom.cluster(0.5)
        """
        if self.MMS is None:
            self.compute_MMS()

        # Log Quotient Parameters
        self.epsilon = epsilon
        self.linkage = linkage

        self.clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage=linkage,
            compute_distances=True,
            distance_threshold=epsilon,
            n_clusters=None,
        )
        self.clustering.fit(self.MMS)

        return self.clustering

    def compute_set_cover(self, epsilon):
        """
        Compute a set of representatives for the embeddings in the Atom.

        Parameters
        ----------
        epsilon : float
            The maximum distance between an embedding and its representative.

        Returns
        -------
        dict
            A dictionary where the keys are the indices of the embeddings and the values are lists of representatives.

        Examples
        --------
        >>> atom.compute_set_cover(0.5)
        """
        # Compute Set Cover
        if self.MMS is None:
            self.compute_MMS()

        G = self._set_cover_graph(self.MMS, epsilon)
        set_cover = self._compute_set_cover(G)

        return set_cover

    @staticmethod
    def _compute_set_cover(G_original):
        """
        Compute a set-cover approximation based on a greedy bipartite-graph heuristic.

        Parameters
        ----------
        G_original : networkx.DiGraph
            The original directed graph representing the set cover problem.

        Returns
        -------
        dict
            A dictionary where the keys are representatives and the values are lists of elements covered by each representative.

        Notes
        -----
        This function uses a greedy bipartite-graph heuristic to compute an approximation of the set cover problem. It iteratively selects representatives from the left side of the bipartite graph that cover the maximum number of elements from the right side. The algorithm continues until all elements from the right side are covered.

        The input graph `G_original` should be a directed graph where the nodes on the left side have a label of 0 and the nodes on the right side have a label of 1.

        Examples
        --------
        >>> G = nx.DiGraph()
        >>> G.add_nodes_from([(0, {'label': 0}), (1, {'label': 1}), (2, {'label': 1}), (3, {'label': 0})])
        >>> G.add_edges_from([(0, 1), (0, 2), (3, 1)])
        >>> _compute_set_cover(G)
        {0: [1, 2], 3: [1]}
        """
        set_cover_representatives = dict()
        G = G_original.copy(as_view=False)
        right = {i for i in G.nodes() if i[-1] == 1}
        while right:
            rep, rep_deg = max(
                {(i, G.out_degree(i)) for i in G.nodes() if i[-1] == 0},
                key=lambda tup: tup[-1],
            )
            set_cover_representatives[rep[0]] = sorted(
                [i[0] for i in G_original.successors(rep)]
            )
            current_successors = list(G.successors(rep))
            G.remove_nodes_from([rep, *current_successors])
            right -= set(current_successors)
        return set_cover_representatives

    @staticmethod
    def _set_cover_graph(MMS, epsilon):
        """
        Construct a bipartite graph for set-cover approximation.

        Parameters
        ----------
        MMS : numpy.ndarray
            The MMS (Minimum Matching Score) matrix.
        epsilon : float
            The maximum distance threshold for creating edges in the graph.

        Returns
        -------
        networkx.DiGraph
            The constructed bipartite graph.

        Notes
        -----
        The left node set has 0 as the second coordinate, and the right node set has 1 as the second coordinate.
        An edge exists from (i,0) to (j,1) if the distance between i and j is at most epsilon.
        """
        n_probes = MMS.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from([(i, 0) for i in range(n_probes)])
        G.add_nodes_from([(i, 1) for i in range(n_probes)])
        for i in range(n_probes):
            edges = [
                ((i, 0), (j, 1)) for j in np.argwhere(MMS[i] <= epsilon).ravel()
            ]
            if edges:
                G.add_edges_from(edges)
        return G
