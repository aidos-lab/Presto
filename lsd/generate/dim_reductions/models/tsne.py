import numpy as np
from sklearn.manifold import TSNE

from lsd.generate.dim_reductions.models.projector import BaseProjector
from lsd.generate.dim_reductions.configs import tSNE as tSNEConfig
import lsd.utils as ut


class TSNEProjector(BaseProjector):
    """
    Class for performing dimensionality reduction using t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Inherits from BaseProjector, which provides common functionality for projectors.

    Attributes
    ----------
    config : tSNEConfig
        Configuration object containing parameters for t-SNE projection.

    Methods
    -------
    project(data)
        Perform dimensionality reduction on input data using t-SNE.

    """

    def __init__(self, config: tSNEConfig):
        """
        Constructor method for TSNEProjector class.

        Parameters
        ----------
        config : tSNEConfig
            Configuration object containing parameters for t-SNE projection.
        """
        super().__init__(config)

    def project(self, data) -> np.ndarray:
        """
        Perform dimensionality reduction on input data using t-Distributed Stochastic Neighbor Embedding (t-SNE).

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        np.ndarray
            tSNE embedding of shape (n_samples,self.config.n_components).

        """
        params = ut.get_parameters(TSNE, self.config)
        operator = TSNE(**params)

        return operator.fit_transform(data)


def initialize() -> TSNEProjector:
    """
    Function to initialize an instance of TSNEProjector.

    Returns
    -------
    TSNEProjector
        Instance of TSNEProjector class.

    """
    return TSNEProjector
