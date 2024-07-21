from __future__ import annotations
import os
from abc import ABC, abstractmethod

import torch
from torch.utils.data import RandomSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from lsd.generate.autoencoders.configs import DataModule as DataModuleConfig


class DataModule(ABC):
    """
    Abstract base class for data modules in a PyTorch-based framework.

    The `DataModule` class is designed to provide a consistent interface for handling datasets
    in machine learning workflows. It encompasses methods for setting up the dataset,
    preparing data splits, and creating data loaders for training, validation, and testing.

    Attributes
    ----------
    entire_ds : Dataset
        The complete dataset containing all data, which will be split into training, validation, and test sets.
    train_ds : Dataset, optional
        The dataset used for training. Initialized as `None`.
    test_ds : Dataset, optional
        The dataset used for testing. Initialized as `None`.
    val_ds : Dataset, optional
        The dataset used for validation. Initialized as `None`.
    config : DataModuleConfig
        Configuration object containing parameters for dataset processing and training.
    data_dir : str
        Directory path where the dataset is stored or downloaded.
    random_sampler : RandomSampler, optional
        Sampler used for randomly sampling a subset of the training dataset based on `sample_size`.
    """

    entire_ds: Dataset
    train_ds: Dataset | None = None
    test_ds: Dataset | None = None
    val_ds: Dataset | None = None

    def __init__(self, config: DataModuleConfig) -> None:
        """
        Initializes the DataModule with the provided configuration.

        Parameters
        ----------
        config : DataModuleConfig
            Configuration object that contains various parameters such as `seed`, `batch_size`, `num_workers`,
            `pin_memory`, and others required for dataset handling and training setup.
        """
        super().__init__()
        self.config = config
        self.data_dir = self.get_data_directory(default_subdir="downloads")
        self.entire_ds = self.setup()
        self.prepare_data()

        if self.config.sample_size == 1:
            self.random_sampler = None
        else:
            num_samples = int(self.config.sample_size * len(self.train_ds))
            self.random_sampler = RandomSampler(
                self.train_ds,
                num_samples=num_samples,
            )

    @abstractmethod
    def setup(self) -> Dataset:
        """
        Abstract method to be implemented by subclasses to provide dataset setup.

        This method should return the complete dataset which includes all samples that
        will be later split into training, validation, and test sets.

        Returns
        -------
        Dataset
            The complete dataset to be used for training, validation, and testing.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclasses.
        """
        raise NotImplementedError()

    def prepare_data(self):
        """
        Splits the entire dataset into training, validation, and test sets.

        This method uses the `random_split` function to divide the dataset according to
        the ratios specified in `config.train_test_split` and seed specified by `config.train_test_seed`.
        """
        seed = self.config.get("train_test_seed", 42)
        generator = torch.Generator().manual_seed(seed)
        self.train_ds, self.test_ds, self.val_ds = (
            torch.utils.data.random_split(
                self.entire_ds,
                self.config.train_test_split,
                generator=generator,
            )
        )

    def train_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            A DataLoader object for the training dataset.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
            sampler=self.random_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            A DataLoader object for the validation dataset.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
        )

    def test_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the test dataset.

        Returns
        -------
        DataLoader
            A DataLoader object for the test dataset.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
        )

    def full_dataloader(self) -> DataLoader:
        """
        Creates a DataLoader for the entire dataset.

        Returns
        -------
        DataLoader
            A DataLoader object for the entire dataset.
        """
        return DataLoader(
            self.entire_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
            sampler=self.random_sampler,
        )

    def info(self):
        """
        Prints summary information about the dataset.

        The summary includes the length of the training, validation, and test datasets,
        as well as the class distribution in each split.
        """
        print("Length of training dataset:", len(self.train_ds))
        print("Length of validation dataset:", len(self.val_ds))
        print("Length of test dataset:", len(self.test_ds))
        print("Number of classes in the dataset:", self.config.num_classes)

        print("Class distribution in the training dataset:")
        train_counts = torch.zeros(self.config.num_classes)
        for x, y in self.train_dataloader():
            train_counts += torch.bincount(y, minlength=self.config.num_classes)
        self.display_histogram(train_counts)

        print("Class distribution in the validation dataset:")
        val_counts = torch.zeros(self.config.num_classes)
        for x, y in self.val_dataloader():
            val_counts += torch.bincount(y, minlength=self.config.num_classes)
        self.display_histogram(val_counts)

        print("Class distribution in the test dataset:")
        test_counts = torch.zeros(self.config.num_classes)
        for x, y in self.test_dataloader():
            test_counts += torch.bincount(y, minlength=self.config.num_classes)
        self.display_histogram(test_counts)

    @staticmethod
    def display_histogram(counts):
        """
        Displays a histogram for the given class distribution counts.

        Parameters
        ----------
        counts : torch.Tensor
            A tensor containing counts for each class.

        The histogram is printed to the console.
        """
        max_count = counts.max().item()
        for i, count in enumerate(counts):
            bar_length = int(count.item() * 50 / max_count)
            bar = "#" * bar_length
            print(f"{i}: {count.item():<5} |{bar:<50}")

    @staticmethod
    def get_data_directory(default_subdir):
        """
        Determines and creates the default data directory.

        Parameters
        ----------
        default_subdir : str
            Subdirectory name to be used for storing data.

        Returns
        -------
        str
            The path to the data directory.
        """
        default_data_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), default_subdir
        )
        os.makedirs(default_data_dir, exist_ok=True)
        return default_data_dir
