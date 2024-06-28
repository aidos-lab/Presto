from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import os
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader, ImbalancedSampler


class DataModule(ABC):
    entire_ds: Dataset
    train_ds: Dataset | None = None
    test_ds: Dataset | None = None
    val_ds: Dataset | None = None

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)
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
        This method should be implemented by the user.
        Please provide a concatenation of the entire dataset,
        train, test and validation. The prepare_data will create
        the appropriate splits.

        returns: Dataset
        """
        raise NotImplementedError()

    def prepare_data(self):
        self.train_ds, self.test_ds, self.val_ds = (
            torch.utils.data.random_split(
                self.entire_ds, self.config.train_test_split
            )
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            # sampler = ImbalancedSampler(self.train_ds),
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
            sampler=self.random_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
        )

    def full_dataloader(self) -> DataLoader:
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
        """Prints summary information about the dataset."""
        print("Length of training dataset:", len(self.train_ds))
        print("Length of validation dataset:", len(self.val_ds))
        print("Length of test dataset:", len(self.test_ds))
        print("Number of classes in the dataset:", self.config.num_classes)

        # Calculate class distribution in the datasets
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
        max_count = counts.max().item()
        for i, count in enumerate(counts):
            bar_length = int(count.item() * 50 / max_count)
            bar = "#" * bar_length
            print(f"{i}: {count.item():<5} |{bar:<50}")

    @staticmethod
    def get_data_directory(default_subdir):
        # Determine the default data directory
        default_data_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), default_subdir
        )
        os.makedirs(default_data_dir, exist_ok=True)
        return default_data_dir
