import numpy as np
from sklearn.manifold import LocallyLinearEmbedding as LLE

from lsd.generate.dim_reductions.models.projector import BaseProjector
from lsd.generate.dim_reductions.configs import LLE as LLEConfig
import lsd.utils as ut


class LLEProjector(BaseProjector):
    """
    Class for performing dimensionality reduction using Locally Linear Embedding (LLE).

    Inherits from BaseProjector, which provides common functionality for projectors.

    Attributes
    ----------
    config : LLEConfig
        Configuration object containing parameters for LLE projection.

    Methods
    -------
    project(data)
        Perform dimensionality reduction on input data using LLE.

    """

    def __init__(self, config: LLEConfig):
        """
        Constructor method for LLEProjector class.

        Parameters
        ----------
        config : LLEConfig
            Configuration object containing parameters for LLE projection.
        """
        super().__init__(config)

    def project(self, data) -> np.ndarray:
        """
        Perform dimensionality reduction on input data using Locally Linear Embedding (LLE).

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        np.ndarray
            Locally Linear Embedding of shape (n_samples,self.config.n_components).
        """
        params = ut.get_parameters(LLE, self.config)
        operator = LLE(**params)
        return operator.fit_transform(data)


def initialize() -> LLEProjector:
    """
    Function to initialize an instance of LLEProjector.

    Returns
    -------
    LLEProjector
        Instance of LLEProjector class.

    """
    return LLEProjector
