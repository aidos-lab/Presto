from umap import UMAP

from lsd.generate.dim_reductions.configs import Projector


class UMAPProjector:
    def __init__(self, config: Projector):
        super().__init__(config)

    def project(self, data):
        operator = UMAP(
            n_neighbors=self.config.nn,
            min_dist=self.config.min_dist,
            n_components=self.dim,
            metric=self.metric,
            init=self.config.init,
            random_state=self.config.seed,
        )

        return operator.fit_transform(data)


def initialize():
    return UMAPProjector
