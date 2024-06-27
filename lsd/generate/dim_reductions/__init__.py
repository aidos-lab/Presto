from .configs import UMAP, tSNE, Phate, Isomap, LLE
from .configs import (
    iris,
    digits,
    linnerud,
    wine,
    breast_cancer,
    MNIST,
    swiss_roll,
    barbell,
    moons,
    noisy_annulus,
)
from .configs import (
    Trainer,
    UMAPTrainer,
    tSNETrainer,
    PhateTrainer,
    IsomapTrainer,
    LLETrainer,
)
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
    "swiss_roll",
    "barbell",
    "moons",
    "noisy_annulus",
    "Trainer",
    "UMAPTrainer",
    "tSNETrainer",
    "PhateTrainer",
    "IsomapTrainer",
    "LLETrainer",
]
