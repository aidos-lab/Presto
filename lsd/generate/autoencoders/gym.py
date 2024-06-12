"Send your Autoencoder to the Gym for training!"

import time
import importlib

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import mse_loss

import lsd.utils as ut


class Gym:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = self._load_generator_module(self.config.model)(self.config)
        self.dm = self._load_generator_module(self.config.dataset)(self.config)
        self.optimizer = self._load_generator_module(self.config.optimizer)(
            model.parameters(), self.config
        )

        DP = torch.nn.DataParallel(model)
        # Send model to device
        DP.to(self.device)
        self.model = DP.module

    def run(self):
        """
        Runs an experiment given the loaded config files.
        """
        start = time.time()
        for epoch in range(self.config.epochs):
            stats = self.train_epoch()

            reported_loss = self.loss.item()

            print(f"epoch {epoch} | train loss {reported_loss:.4f}")

            if "Reconstruction_Loss" in stats.keys():
                recon_loss = stats["Reconstruction_Loss"]
                print(f"epoch {epoch} | train recon loss {recon_loss:.4f}")

            if epoch % 10 == 0:
                end = time.time()
                self.compute_metrics(epoch)
                print(
                    f"Training the model {epoch+1} epochs took: {end - start:.4f} seconds."
                )

                start = time.time()
            if np.isnan(self.loss.item()):
                self.successful_training = False
                break

        self.finalize_training()

    def train_epoch(self):
        self.model.to(self.device)

        loader = self.dm.train_dataloader()
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.float(), y.float()
            X, _ = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            results = self.model(X)

            stats = self.model.loss_function(
                *results,
                batch_idx=batch_idx,
                M_N=0.00025,
                optimizer_idx=0,
            )
            self.loss = stats["loss"]
            self.loss.backward()

            clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.clip_max_norm
            )

            self.optimizer.step()

        return stats

    def finalize_training(self):
        # Test Reconstruction Loss
        test_loss = Gym.compute_recon_loss(
            self.model, self.dm.test_dataloader()
        )
        epoch = self.config.epochs
        # Log statements
        print(f"epoch {epoch} | test recon loss {test_loss:.4f}")

    def compute_metrics(self, epoch):
        val_loss = Gym.compute_recon_loss(
            self.model, self.dm.val_dataloader(), self.device
        )

        # Log statements to console
        print(f"epoch {epoch} | val loss { val_loss.item():.4f}")

    @staticmethod
    def compute_recon_loss(model, loader, device="cpu"):
        with torch.no_grad():
            torch.cuda.empty_cache()
            y_true = torch.Tensor()
            y_pred = torch.Tensor()

            model.to(device)
            for X, _ in loader:
                batch_gpu = X.to(device)
                y_true = torch.cat((y_true, X))
                recon = model.generate(batch_gpu).detach().cpu()
                y_pred = torch.cat((y_pred, recon))

            y_true = y_true.to(device)
            y_pred = y_pred.to(device)
            recon_loss = mse_loss(y_pred, y_true)
            return recon_loss

    @staticmethod
    def save_model(model, id, outDir):
        pass

    @staticmethod
    def _load_generator_module(module):
        return importlib.import_module(module).initialize()

    # def save_model(model, id, folder):
    # path = os.path.join(folder, "models/")

    # if not os.path.isdir(path):
    #     os.makedirs(path)

    # file = os.path.join(path, f"model_{id}")
    # torch.save(model, f=file)
