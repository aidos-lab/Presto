from sklearn.manifold import LocallyLinearEmbedding as LLE

from lsd.generate.dim_reductions.models.projector import BaseProjector


class LLEProjector(BaseProjector):
    def __init__(self, config):
        super().__init__(config)

    def project(self, data):
        operator = LLE(
            n_neighbors=self.config.nn,
            reg=self.config.reg,
            n_components=self.config.dim,
        )
        return operator.fit_transform(data)


def initialize():
    return LLEProjector
