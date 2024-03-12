from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader, ImbalancedSampler

from config import DataModuleConfig


class DataModule(ABC):
    entire_ds: Dataset
    train_ds: Dataset | None = None
    test_ds: Dataset | None = None
    val_ds: Dataset | None = None

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self.config = config
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
        # if self.train_ds and self.test_ds and self.val_ds:
        #     return
        print("Random train/val/test split")
        self.train_ds, self.test_ds, self.val_ds = torch.utils.data.random_split(
            self.entire_ds, [0.6, 0.3, 0.1]
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
            self.train_ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=self.config.pin_memory,
            multiprocessing_context="fork",
            sampler=self.random_sampler,
        )

    def info(self):
        """Still requires a lot of work."""
        print("len train_ds", len(self.train_ds))
        print("len val_ds", len(self.val_ds))
        print("len test_ds", len(self.test_ds))
        print("data num_classes", self.entire_ds.num_classes)
        print(self.train_ds)
        print(self.val_ds)
        print(self.train_ds[0])
        counts = torch.zeros(self.entire_ds.num_classes)
        for data in self.train_dataloader():
            counts += torch.bincount(data.y, minlength=10)
        print("Bincount train", counts)
        counts = torch.zeros(self.entire_ds.num_classes)
        for data in self.val_dataloader():
            counts += torch.bincount(data.y, minlength=10)
        print("Bincount val", counts)
        counts = torch.zeros(self.entire_ds.num_classes)
        for data in self.test_dataloader():
            counts += torch.bincount(data.y, minlength=10)
        print("Bincount test", counts)
