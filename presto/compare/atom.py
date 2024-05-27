"ATOM: Approximate Topological Operations in the Multiverse "
import pickle

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import itertools
import networkx as nx
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import coo_array
from presto.compare.presto import Presto
from sklearn.decomposition import PCA


class Atom:
    def __init__(
        self,
        data: list,
        params: dict = {},
        seed: int = 42,
    ) -> None:
        self.data = data
        self.params = params
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
        Returns a pairwise distances matrix based on the
        `presto` score between embeddings.
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
        with open(path, "wb") as f:
            pickle.dump(self.MMS, f)

    def load_mms(self, path: str):
        with open(path, "rb") as f:
            MMS = pickle.load(f)
        self.set_mss(MMS)

    def set_mms(self, MMS):
        self.MMS = MMS

    def cluster(
        self,
        epsilon,
        linkage: str = "complete",
    ) -> AgglomerativeClustering:
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

    # TODO untested
    def compute_set_cover(self, epsilon):
        """
        Compute a set of representatives for a given set of embeddings
        such that each embedding has a representative at distance at most epsilon.
        Uses a greedy approximation to set cover that guarantees the cardinality of
        the set of representatives will be at most H(k) \in O(log k) times the size
        of the optimum.
        """
        # Log Set Cover Parameters
        self.set_cover_epsilon = epsilon

        # Compute Set Cover
        self.compute_MMS()
        self.set_cover = self._compute_set_cover()

        return self.set_cover

    # TODO untested + naive implementation (but scalability is probably not an issue here)
    # TODO do we really want to set attributes _and_ return their values?
    def _compute_set_cover(self):
        """
        Compute a set-cover approximation based on a greedy bipartite-graph heuristic.
        """
        self.set_cover_representatives = dict()
        G_original = self._set_cover_graph()
        G = G_original.copy(as_view=False)
        right = {i for i in G.nodes() if i[-1] == 1}
        while right:
            rep, rep_deg = max(
                {(i, G.out_degree(i)) for i in G.nodes() if i[-1] == 0},
                key=lambda tup: tup[-1],
            )
            self.set_cover_representatives[rep[0]] = sorted(
                [i[0] for i in G_original.successors(rep)]
            )
            current_successors = list(G.successors(rep))
            G.remove_nodes_from([rep, *current_successors])
            right -= set(current_successors)
        return self.set_cover_representatives

    # TODO untested
    def _set_cover_graph(self):
        """
        Construct a bipartite graph for set-cover approximation.
        Left node set has 0 as second coordinate, right node set has 1 as second coordinate.
        There is an edge from (i,0) to (j,1) if the distance between i and j is at most epsilon.
        TODO: At most or less than?
        """
        n_probes = self.MMS.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from([(i, 0) for i in range(n_probes)])
        G.add_nodes_from([(i, 1) for i in range(n_probes)])
        for i in range(n_probes):
            edges = [
                ((i, 0), (j, 1))
                for j in np.argwhere(
                    self.MMS[i] <= self.set_cover_epsilon
                ).ravel()
            ]
            if edges:
                G.add_edges_from(edges)
        return G
