import time
import importlib
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import mse_loss
from typing import Dict, Callable

from lsd.utils import ConfigType
from lsd.generate.autoencoders.logger import Logger


class Gym:
    """
    A Gym to train your generative autoencoder models!

    A class for managing the training and latent space generation of autoencoder models using PyTorch.

    Attributes
    ----------
    config : ConfigType
        Configuration settings for the model, dataset, and optimizer.
    logger : Logger
        Logger instance for recording training progress and metrics.
    device : torch.device
        Device to use for model training and inference.
    model : torch.nn.Module
        The autoencoder model wrapped in DataParallel for multi-GPU support.
    dm : DataManager
        Dataset manager instance to handle data loading.
    optimizer : torch.optim.Optimizer
        Optimizer used for training the model.
    loss : torch.Tensor
        Loss value computed during training.


    Methods
    -------
    train()
        Trains the autoencoder model for the specified number of epochs.
    latent_space(data_loader: str = "train_dataloader")
        Computes the latent space embeddings for the given data loader using self.model.

    Helper Methods
    --------------
    _train_epoch()
        Trains the model for a single epoch.
    _finalize_training(start_time: float)
        Finalizes the training process by computing and logging the test loss.
    _compute_metrics(epoch: int)
        Computes and logs the validation loss for the specified epoch.
    _init_model()
        Initializes and returns the model.
    _init_data_manager()
        Initializes and returns the data manager.
    _init_optimizer()
        Initializes and returns the optimizer.
    _clip_and_step()
        Clips gradients and performs an optimizer step.
    _extract_latent_space(loader: torch.utils.data.DataLoader)
        Extracts and returns latent space embeddings for a given data loader.
    _compute_recon_loss(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu")
        Static method for computing reconstruction loss.
    _load_module(module_name: str)
        Static method to dynamically load a module.
    _log(message: str)
        Logs a message using the provided logger.
    _log_epoch_stats(epoch: int, stats: Dict[str, torch.Tensor])
        Logs training loss statistics for a given epoch.
    _log_time(start_time: float, message: str)
        Logs the elapsed time for an operation with a given start time.
    """

    LOG_INTERVAL = 10  # Log every 10 epochs

    def __init__(self, config: ConfigType, logger: Logger):
        """
        Initializes the Gym class with the given configuration and logger.

        Parameters
        ----------
        config : ConfigType
            Configuration settings for the model, dataset, and optimizer.
        logger : Logger
            Logger instance for recording training progress and metrics.
        """
        self.config = config
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._set_seed()

        self.model = self._init_model()
        self.dm = self._init_data_manager()
        self.optimizer = self._init_optimizer()

        self.loss = None

        self.logger.log("Gym initialized")

    def train(self) -> torch.nn.Module:
        """
        Trains the autoencoder model for the specified number of epochs.

        Returns
        -------
        torch.nn.Module
            The trained autoencoder model.
        """
        start_time = time.time()

        if self.config.epochs == 0:
            self._log(
                "No training epochs specified. Returning untrained model."
            )
            self._finalize_training(start_time)
            return self.model.module

        for epoch in range(self.config.epochs):
            self._log(f"Starting epoch {epoch + 1}")
            stats = self._train_epoch()

            self._log_epoch_stats(epoch, stats)

            if epoch % self.LOG_INTERVAL == 0 and epoch != 0:
                self._compute_metrics(epoch)
                self._log_time(start_time, "Training 10 epochs took")
                start_time = time.time()

            if np.isnan(self.loss.item()):
                self._log("NaN loss detected, stopping training.")
                break

        self._finalize_training(start_time)
        return self.model.module

    def latent_space(self, data_loader: str = "train_dataloader") -> np.ndarray:
        """
        Computes the latent space embeddings for the given data loader.

        Parameters
        ----------
        data_loader : str, optional
            The data loader to use for generating embeddings. Defaults to "train_dataloader".

        Returns
        -------
        np.ndarray
            Numpy array containing the latent space embeddings.
        """
        loader = getattr(self.dm, data_loader)()
        embedding = self._extract_latent_space(loader)
        return embedding.numpy()

    def _train_epoch(self) -> Dict[str, torch.Tensor]:
        """
        Trains the model for a single epoch.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing loss statistics for the epoch.
        """
        loader = self.dm.train_dataloader()
        stats = dict()

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.float().to(self.device), y.float().to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            results = self.model(x)
            stats = self.model.module.loss_function(
                *results,
                batch_idx=batch_idx,
                M_N=self.config.kld,
                optimizer_idx=self.config.optimizer_idx,
            )
            self.loss = stats["loss"]
            self.loss.backward()
            self._clip_and_step()

        return stats

    def _finalize_training(self, start_time: float) -> None:
        """
        Finalizes the training process by computing and logging the test loss.

        Parameters
        ----------
        start_time : float
            The starting time of the training process to compute the elapsed time.
        """
        test_loss = self._compute_recon_loss(
            self.model, self.dm.test_dataloader()
        )
        self._log(
            f"Epoch {self.config.epochs} | Test reconstruction loss: {test_loss:.4f}"
        )
        self._log_time(start_time, f"Training {self.config.epochs} took")

    def _compute_metrics(self, epoch: int) -> None:
        """
        Computes and logs the validation loss for the specified epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number for which to compute the validation loss.
        """
        val_loss = self._compute_recon_loss(
            self.model, self.dm.val_dataloader(), self.device
        )
        self._log(f"Epoch {epoch+1} | Validation loss: {val_loss.item():.4f}")

    def _init_model(self) -> torch.nn.Module:
        """Initializes and returns the model."""
        model_class = self._load_module(self.config.model)
        model = model_class(self.config)
        return torch.nn.DataParallel(model).to(self.device)

    def _init_data_manager(self) -> ConfigType:
        """Initializes and returns the data manager."""
        data_manager_class = self._load_module(self.config.dataset)
        return data_manager_class(self.config)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initializes and returns the optimizer."""
        optimizer_class = self._load_module(self.config.optimizer)
        return optimizer_class(self.model.parameters(), self.config)

    def _clip_and_step(self) -> None:
        """Clips gradients and performs an optimizer step."""
        clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.clip_max_norm
        )
        self.optimizer.step()

    def _extract_latent_space(
        self, loader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Extracts and returns latent space embeddings for a given data loader."""
        embedding = torch.Tensor()
        for x, _ in loader:
            train_data = x.float().to(self.device)
            batch_embedding = (
                self.model.module.latent(train_data).detach().cpu()
            )
            embedding = torch.cat((embedding, batch_embedding))
        return embedding

    def _set_seed(self) -> None:
        """
        Set the manual seed for reproducibility.
        """
        seed = self.config.get("seed", 42)
        print(f"Using seed {seed} for reproducibility.")
        np.random.seed(seed)  # NumPy random seed
        torch.manual_seed(seed)  # PyTorch CPU random seed

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # PyTorch CUDA random seed
            torch.cuda.manual_seed_all(seed)  # PyTorch all GPUs random seed

        # Ensure that cuDNN is deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _compute_recon_loss(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Computes the reconstruction loss for the model on the given data loader.

        Parameters
        ----------
        model : torch.nn.Module
            The autoencoder model to evaluate.
        loader : torch.utils.data.DataLoader
            Data loader providing the dataset for loss computation.
        device : str, optional
            Device to use for computation ('cpu' or 'cuda'). Defaults to 'cpu'.

        Returns
        -------
        torch.Tensor
            The computed reconstruction loss as a tensor.
        """
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
        """
        Dynamically loads a module by its name and returns its initialize function.

        Parameters
        ----------
        module_name : str
            Name of the module to load.

        Returns
        -------
        Callable
            The initialize function of the loaded module.
        """
        return importlib.import_module(module_name).initialize()

    def _log(self, message: str) -> None:
        """Logs a message using the provided logger."""
        self.logger.log(message)

    def _log_epoch_stats(
        self, epoch: int, stats: Dict[str, torch.Tensor]
    ) -> None:
        """Logs training loss statistics for a given epoch."""
        reported_loss = self.loss.item()
        self._log(f"Epoch {epoch + 1} | Train loss: {reported_loss:.4f}")

        if "Reconstruction_Loss" in stats:
            recon_loss = stats["Reconstruction_Loss"]
            self._log(
                f"Epoch {epoch + 1} | Train reconstruction loss: {recon_loss:.4f}"
            )

    def _log_time(self, start_time: float, message: str) -> None:
        """Logs the elapsed time for an operation with a given start time."""
        elapsed_time = time.time() - start_time
        self._log(f"{message} {elapsed_time:.4f} seconds")
