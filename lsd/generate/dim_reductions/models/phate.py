from phate import PHATE

from lsd.generate.dim_reductions.models.projector import BaseProjector
from lsd.generate.dim_reductions.configs import Phate as PHATEConfig
import lsd.utils as ut


class PhateProjector(BaseProjector):
    """
    Class for performing dimensionality reduction using Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE).

    Inherits from BaseProjector, which provides common functionality for projectors.

    Parameters
    ----------
    config : PHATEConfig
        Configuration object containing parameters for PHATE projection.

    Methods
    -------
    project(data)
        Perform dimensionality reduction on input data using PHATE.

    """

    def __init__(self, config: PHATEConfig):
        """
        Constructor method for PHATEProjector class.

        Parameters
        ----------
        config : PHATEConfig
            Configuration object containing parameters for PHATE projection.
        """
        super().__init__(config)

    def project(self, data):
        """
        Perform dimensionality reduction on input data using Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE).

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        np.ndarray
            PHATE embedding of shape (n_samples,self.config.n_components).
        """
        params = ut.get_parameters(PHATE, self.config)
        operator = PHATE(**params)
        return operator.fit_transform(data)


def initialize() -> PhateProjector:
    """
    Function to initialize an instance of PHATEProjector.

    Returns
    -------
    PhateProjector
    """
    return PhateProjector
