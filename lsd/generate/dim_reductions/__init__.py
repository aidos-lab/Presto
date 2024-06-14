from .configs import UMAP, tSNE, Phate, Isomap, LLE
from .configs import iris, digits, linnerud, wine, breast_cancer, MNIST
from .configs import Thread
from .dr import DimReduction

__all__ = [
    "DimReduction",
    "UMAP",
    "tSNE",
    "Phate",
    "Isomap",
    "LLE",
    "iris",
    "digits",
    "linnerud",
    "wine",
    "breast_cancer",
    "MNIST",
    "Thread",
]
