from phate import PHATE
from lsd.generate.dim_reductions.models.projector import BaseProjector


class PhateProjector(BaseProjector):
    def __init__(self, config):
        super(
            PhateProjector,
            self,
        ).__init__(config)

    def project(self, data):
        operator = PHATE(
            knn=self.config.k,
            gamma=self.config.gamma,
            knn_dist=self.metric,
            n_components=self.dim,
            verbose=0,
        )
        return operator.fit_transform(data)


def initialize():
    return PhateProjector
