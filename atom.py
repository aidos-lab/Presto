"ATOM: Approximate Topological Operations in the Multiverse "
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import itertools
import networkx as nx
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from presto import Presto


class Atom:
    def __init__(
            self,
            data: list,
            n_components: int = 2,
            normalize: bool = False,
            max_homology_dim: int = 1,
            resolution: int = 100,
            normalization_approx_iterations: int = 1000,
            seed: int = 42
    ) -> None:

        self.presto = Presto(n_components=n_components,
                             normalize=normalize,
                             max_homology_dim=max_homology_dim,
                             resolution=resolution,
                             normalization_approx_iterations=normalization_approx_iterations,
                             seed=seed)
        self.data = data
        self.multiverse_size = len(data)
        self.MMS = None

    def compute_MMS(self, n_projections: int = 15, score_type: str = "aggregate", ):
        """
        Compute a multiverse metric space (MMS).
        Returns a pairwise distances matrix based on the 
        `presto` score between embeddings.
        """
        # TODO I think we need to make the indices of the elements part of the pairs and then return them along
        # with the scores in a tuple to be able to allocate our scores correctly after the parallel execution
        # also, do you really want to copy the data this many times? We could generate combinations of indices,
        # initialize each worker with the entire data (if that works with this executor – which I believe it does),
        # and then have compute_distance use the indices to pick the correct elements from data?
        pairs = list(itertools.combinations(self.data, 2))
        # TODO we don't use that anywhere, and below it becomes a list
        scores = np.ndarray(shape=len(pairs))
        if score_type not in ["aggregate", "average"]:
            raise NotImplementedError(score_type)

        def compute_distance(pair):
            X, Y = pair
            if np.isnan(X).any() or np.isnan(Y).any():
                return np.nan
            else:
                return self.presto.fit_transform(X, Y, n_projections=n_projections, score_type=score_type)

        # TODO executor.map might evaluate pairs out of order, so we cannot simply set the scores as we do currently
        # See above for suggestion – also, I hope you didn't rely on in-order returns anywhere else (e.g., in other experiments?)
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            scores = list(
                tqdm(executor.map(compute_distance, pairs), total=len(pairs), desc="Computing Presto Distances",
                     unit="universes"))

        self.MMS = np.zeros((self.multiverse_size, self.multiverse_size))

        triu_indices = np.triu_indices(self.multiverse_size, k=1)
        self.MMS[triu_indices] = scores
        self.MMS.T[triu_indices] = scores
        # TODO you probably want to enable saving and loading of an MMS, such that we can more easily play around with
        # clustering, set cover, sensitivity analysis, etc.

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
    def compute_set_cover(
            self,
            epsilon
    ):
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
