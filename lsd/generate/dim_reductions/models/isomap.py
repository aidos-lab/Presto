from sklearn.manifold import Isomap


from lsd.generate.dim_reductions.models.projector import BaseProjector


class IsomapProjector(BaseProjector):
    def __init__(self, config):
        super().__init__(config)

    def project(self, data):
        operator = Isomap(
            n_neighbors=self.config.nn,
            n_components=self.dim,
            metric=self.metric,
        )
        return operator.fit_transform(data)


def initialize():
    return IsomapProjector
