import time
import importlib
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import mse_loss
from typing import Any, Dict, Callable

from lsd.generate.autoencoders.logger import Logger


class Gym:
    def __init__(self, config: Any, logger: Logger):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logger

        model = self._load_module(self.config.model)(self.config)
        self.dm = self._load_module(self.config.dataset)(self.config)
        self.optimizer = self._load_module(self.config.optimizer)(
            model.parameters(), self.config
        )

        self.model = torch.nn.DataParallel(model).to(self.device)
        self.loss = None

        self.logger.log("Gym initialized")

    def train(self) -> torch.nn.Module:
        start_time = time.time()
        if self.config.epochs == 0:
            self.logger.log(
                "No training epochs specified. Returning untrained model."
            )
            self.finalize_training(start_time)
            return self.model.module

        for epoch in range(self.config.epochs):
            self.logger.log(f"Starting epoch {epoch + 1}")
            stats = self.train_epoch()

            reported_loss = self.loss.item()
            self.logger.log(
                f"Epoch {epoch + 1} | Train loss: {reported_loss:.4f}"
            )

            if "Reconstruction_Loss" in stats:
                recon_loss = stats["Reconstruction_Loss"]
                self.logger.log(
                    f"Epoch {epoch + 1} | Train reconstruction loss: {recon_loss:.4f}"
                )

            if epoch % 10 == 0 and epoch != 0:
                self.compute_metrics(epoch)
                elapsed_time = time.time() - start_time
                self.logger.log(
                    f"Training 10 epochs took {elapsed_time:.4f} seconds"
                )
                start_time = time.time()

            if np.isnan(self.loss.item()):
                self.logger.log("NaN loss detected, stopping training.")
                self.successful_training = False
                break

        self.finalize_training(start_time)
        return self.model.module

    def train_epoch(self) -> Dict[str, torch.Tensor]:
        loader = self.dm.train_dataloader()
        stats = dict()

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.float().to(self.device), y.float().to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            results = self.model(x)
            stats = self.model.module.loss_function(
                *results, batch_idx=batch_idx, M_N=0.00025, optimizer_idx=0
            )
            self.loss = stats["loss"]
            self.loss.backward()

            clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.clip_max_norm
            )
            self.optimizer.step()

        return stats

    def finalize_training(self, start_time) -> None:
        test_loss = self.compute_recon_loss(
            self.model, self.dm.test_dataloader()
        )
        self.logger.log(
            f"Epoch {self.config.epochs} | Test reconstruction loss: {test_loss:.4f}"
        )
        elapsed_time = time.time() - start_time
        self.logger.log(
            f"Training {self.config.epochs} took: {elapsed_time:.4f}."
        )

    def compute_metrics(self, epoch: int) -> None:
        val_loss = self.compute_recon_loss(
            self.model, self.dm.val_dataloader(), self.device
        )
        self.logger.log(
            f"Epoch {epoch+1} | Validation loss: {val_loss.item():.4f}"
        )

    def latent_space(self) -> np.ndarray:
        embedding = torch.Tensor()
        loader = self.dm.full_dataloader()

        for x, _ in loader:
            train_data = x.float().to(self.device)
            batch_embedding = (
                self.model.module.latent(train_data).detach().cpu()
            )
            embedding = torch.cat((embedding, batch_embedding))

        return embedding.numpy()

    @staticmethod
    def compute_recon_loss(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> torch.Tensor:
        y_true, y_pred = torch.Tensor(), torch.Tensor()

        with torch.no_grad():
            model.to(device)
            for x, _ in loader:
                batch_gpu = x.to(device)
                y_true = torch.cat((y_true, x))
                recon = model.module.generate(batch_gpu).detach().cpu()
                y_pred = torch.cat((y_pred, recon))

            y_true, y_pred = y_true.to(device), y_pred.to(device)
            return mse_loss(y_pred, y_true)

    @staticmethod
    def _load_module(module_name: str) -> Callable:
        return importlib.import_module(module_name).initialize()
