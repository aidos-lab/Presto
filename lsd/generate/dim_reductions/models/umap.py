from umap import UMAP

from lsd.generate.dim_reductions.models.projector import BaseProjector
from lsd.generate.dim_reductions.configs import UMAP as UMAPConfig
import lsd.utils as ut


class UMAPProjector(BaseProjector):
    """
    Class for performing dimensionality reduction using Uniform Manifold Approximation and Projection (UMAP).

    Inherits from BaseProjector, which provides common functionality for projectors.

    Attributes
    ----------
    config : UMAPPConfig
        Configuration object containing parameters for t-SNE projection.

    Methods
    -------
    project(data)
        Perform dimensionality reduction on input data using UMAP.

    """

    def __init__(self, config: UMAPConfig):
        """
        Constructor method for UMAPProjector class.

        Parameters
        ----------
        config : Projector
            Configuration object containing parameters for UMAP projection.
        """
        super().__init__(config)

    def project(self, data):
        """
        Perform dimensionality reduction on input data using Uniform Manifold Approximation and Projection (UMAP).

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        np.ndarray
            UMAP embedding of shape (n_samples,self.config.n_components).
        """
        params = ut.get_parameters(UMAP, self.config)
        operator = UMAP(**params)
        return operator.fit_transform(data)


def initialize() -> UMAPProjector:
    """
    Function to initialize an instance of UMAPProjector.

    Returns
    -------
    UMAPProjector
        Instance of UMAPProjector class.

    """
    return UMAPProjector
