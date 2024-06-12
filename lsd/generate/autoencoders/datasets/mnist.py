import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from lsd.generate.autoencoders.datasets.base_dataset import DataModule


class MnistDataModule(DataModule):
    """
    MNIST
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def setup(self):
        """
        This method constructs the entire dataset, note we concatenate the train/test datasets.
        This allows for k-fold cross validation later on.
        If you prefer to create the splits yourself, set the variables
        self.train_ds, self.test_ds and self.val_ds.
        """
        entire_dataset = torch.utils.data.ConcatDataset(
            [
                MNIST(
                    root=self.config.experiment,
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            # transforms.Lambda(lambda x: torch.flatten(x)),
                        ]
                    ),
                    download=True,
                ),
                MNIST(
                    root=self.config.experiment,
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.RandomVerticalFlip(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            # transforms.Lambda(lambda x: torch.flatten(x)),
                        ]
                    ),
                    download=True,
                ),
            ]
        )
        return entire_dataset


def initialize():
    return MnistDataModule
