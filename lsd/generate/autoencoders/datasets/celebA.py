import torch
import torchvision.transforms as transforms
from torchvision.datasets import CelebA

from lsd.generate.autoencoders.datasets.base_dataset import DataModule
from lsd.utils import ConfigType


class CelebADataModule(DataModule):
    """
    Data module for the CelebA dataset.

    The `CelebADataModule` class is designed to handle data preparation, transformations, and loading for the CelebA dataset. It extends the abstract `DataModule` base class and implements the `setup` method to provide the entire dataset.

    Attributes
    ----------
    config : ConfigType
        Configuration object containing parameters for dataset processing and training.

    Methods
    -------
    setup()
        Constructs the entire CelebA dataset, combining both training and testing sets.
    """

    def __init__(self, config: ConfigType):
        """
        Initializes the CelebADataModule with the provided configuration.

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
        Constructs the entire CelebA dataset by concatenating the training and testing sets.

        The `setup` method downloads the CelebA dataset if it is not already present in the specified
        directory, applies a series of transformations including resizing and random vertical and horizontal flips,
        and combines the training and test sets into a single dataset.

        Returns
        -------
        torch.utils.data.ConcatDataset
            The combined dataset consisting of the CelebA training and testing datasets with applied transformations.
        """
        entire_dataset = torch.utils.data.ConcatDataset(
            [
                CelebA(
                    root=self.data_dir,
                    split="train",
                    transform=transforms.Compose(
                        [
                            transforms.Resize(size=(64, 64)),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                        ]
                    ),
                    download=True,
                ),
                CelebA(
                    root=self.data_dir,
                    split="test",
                    transform=transforms.Compose(
                        [
                            transforms.Resize(size=(64, 64)),
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
    Initializes and returns an instance of `CelebADataModule`.

    Returns
    -------
    CelebADataModule
        An instance of the `CelebADataModule` class.
    """
    return CelebADataModule
