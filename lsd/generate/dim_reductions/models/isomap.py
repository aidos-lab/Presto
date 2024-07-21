import numpy as np
from sklearn.manifold import Isomap

from lsd.generate.dim_reductions.models.projector import BaseProjector
from lsd.generate.dim_reductions.configs import Isomap as IsomapConfig
import lsd.utils as ut


class IsomapProjector(BaseProjector):
    """
    Class for performing dimensionality reduction using Isometric Mapping (Isomap).

    Inherits from BaseProjector, which provides common functionality for projectors.

    Attributes
    ----------
    config : IsomapConfig
        Configuration object containing parameters for Isomap projection.

    Methods
    -------
    project(data)
        Perform dimensionality reduction on input data using Isomap.

    """

    def __init__(self, config: IsomapConfig):
        """
        Constructor method for IsomapProjector class.

        Parameters
        ----------
        config : IsomapConfig
            Configuration object containing parameters for Isomap projection.
        """
        super().__init__(config)

    def project(self, data) -> np.ndarray:
        """
        Perform dimensionality reduction on input data using Isomap.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        np.ndarray
            ISOMAP embedding of shape (n_samples,self.config.n_components).

        """
        params = ut.get_parameters(Isomap, self.config)
        operator = Isomap(**params)
        return operator.fit_transform(data)


def initialize() -> IsomapProjector:
    """
    Function to initialize an instance of IsomapProjector.

    Returns
    -------
    IsomapProjector
        Instance of IsomapProjector class.

    """
    return IsomapProjector
