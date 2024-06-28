import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from lsd.generate.autoencoders.datasets.base_dataset import DataModule
from lsd.utils import ConfigType


class MnistDataModule(DataModule):
    """
    Data module for the MNIST dataset.

    The `MnistDataModule` class is designed to handle data preparation, transformations,
    and loading for the MNIST dataset. It extends the abstract `DataModule` base class
    and implements the `setup` method to provide the entire dataset.

    Attributes
    ----------
    config : ConfigType
        Configuration object containing parameters for dataset processing and training.

    Methods
    -------
    setup()
        Constructs the entire MNIST dataset, combining both training and testing sets.
    """

    def __init__(self, config: ConfigType):
        """
        Initializes the MnistDataModule with the provided configuration.

        Parameters
        ----------
        config : ConfigType
            Configuration object that contains various parameters such as `seed`, `batch_size`,
            `num_workers`, `pin_memory`, and others required for dataset handling and training setup.
        """
        super().__init__(config)
        self.config = config

    def setup(self):
        """
        Constructs the entire MNIST dataset by concatenating the training and testing sets.

        The `setup` method downloads the MNIST dataset if it is not already present in the specified
        directory, applies a series of transformations including random vertical and horizontal flips,
        and combines the training and test sets into a single dataset.

        Returns
        -------
        torch.utils.data.ConcatDataset
            The combined dataset consisting of the MNIST training and testing datasets with applied transformations.
        """
        entire_dataset = torch.utils.data.ConcatDataset(
            [
                MNIST(
                    root=self.data_dir,
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
                MNIST(
                    root=self.data_dir,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
            ]
        )
        return entire_dataset


def initialize():
    """
    Initializes and returns an instance of `MnistDataModule`.

    Returns
    -------
    MnistDataModule
        An instance of the `MnistDataModule` class.
    """
    return MnistDataModule
