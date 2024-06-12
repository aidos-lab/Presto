"Send your Autoencoder to the Gym for training!"

import time
import importlib

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

import lsd.utils as ut


class Gym:
    def __init__(self, config):
        """
        Creates the setup and does inits
        - Loads datamodules
        - Loads models
        - Initializes logger
        """
        self.config = config
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.model = self._load_generator_module(self.config.model)(self.config)
        self.dm = self._load_generator_module(self.config.dataset)(self.config)
        # self.optimizer = self._load_generator_module(
        #     self.config.optimizer, self.config
        # )

        DP = torch.nn.DataParallel(self.config.model)
        # Send model to device
        DP.to(self.device)
        self.model = DP.module

    def train(self):
        """
        Runs an experiment given the loaded config files.
        """
        start = time.time()
        for epoch in range(self.config.trainer_params.num_epochs):
            stats = self.train_epoch()

            reported_loss = self.loss.item()

            self.logger.log(
                msg=f"epoch {epoch} | train loss {reported_loss:.4f}",
                params={
                    "train loss": reported_loss,
                },
            )
            if "Reconstruction_Loss" in stats.keys():
                recon_loss = stats["Reconstruction_Loss"]
                self.logger.log(
                    msg=f"epoch {epoch} | train recon loss {recon_loss:.4f}"
                )

            if epoch % 10 == 0:
                end = time.time()
                self.compute_metrics(epoch)
                self.logger.log(
                    msg=f"Training the model 10 epochs took: {end - start:.4f} seconds."
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

            # clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        return stats

    def finalize_training(self):
        # Test Reconstruction Loss
        test_loss = compute_recon_loss(self.model, self.dm.test_dataloader())
        epoch = self.config.trainer_params.num_epochs
        # Log statements
        self.logger.log(
            f"epoch {epoch} | test recon loss {test_loss:.4f}",
            params={
                "test_loss": test_loss,
            },
        )

    def compute_metrics(self, epoch):
        val_loss = compute_recon_loss(self.model, self.dm.val_dataloader())

        # Log statements to console
        self.logger.log(
            msg=f"epoch {epoch} | val loss { val_loss.item():.4f}",
            params={"epoch": epoch, "val_loss": val_loss.item()},
        )

    def save_model(self):
        save_model(
            self.model, id=self.config.meta.id, folder=self.experiment_root
        )
        self.logger.log(msg=f"Model Saved!")

    @staticmethod
    def _load_generator_module(module):
        return importlib.import_module(module).initialize()

    # def save_model(model, id, folder):
    # path = os.path.join(folder, "models/")

    # if not os.path.isdir(path):
    #     os.makedirs(path)

    # file = os.path.join(path, f"model_{id}")
    # torch.save(model, f=file)
