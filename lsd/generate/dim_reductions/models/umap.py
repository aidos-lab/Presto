from dataclasses import dataclass

from umap import UMAP

from ..dr import DimReductionModule


# @dataclass
# class UMAPConfig(DimReductionConfig):
#     n_neighbors: int = 15
#     min_dist: float = 0.1
#     init: str = "spectral"
#     metric: str = "euclidean"
#     dim: int = 2


class UMAPProjector(DimReductionModule):
    def __init__(self, config: UMAPConfig):
        super().__init__(config)

    def project(self, data):
        operator = UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            n_components=self.config.dim,
            metric=self.config.metric,
            init=self.config.init,
            random_state=self.config.seed,
        )

        return operator.fit_transform(data)
