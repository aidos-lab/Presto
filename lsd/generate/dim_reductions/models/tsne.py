from sklearn.manifold import TSNE

from lsd.generate.dim_reductions.models.projector import BaseProjector


class TSNEProjector(BaseProjector):
    def __init__(self, config):
        super(TSNEProjector, self).__init__(config)

    def project(self, data):
        operator = TSNE(
            perplexity=self.config.perplexity,
            early_exaggeration=self.config.ee,
            n_components=self.dim,
            metric=self.metric,
            random_state=self.config.seed,
        )

        return operator.fit_transform(data)


def initialize():
    return TSNEProjector
