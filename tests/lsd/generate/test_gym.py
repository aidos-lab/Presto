from unittest.mock import patch, MagicMock
import torch
import numpy as np


def test_device_initialization_cpu(gym_fixture):
    with patch("torch.cuda.is_available", return_value=False):
        assert gym_fixture.device.type == "cpu"


def test_train_epoch(gym_fixture):
    with patch("torch.utils.data.DataLoader") as mock_dataloader, patch(
        "torch.optim.Optimizer"
    ) as mock_optimizer:

        mock_dataloader.return_value = [
            (torch.randn(5, 10), torch.randn(5, 10))
        ]
        gym_fixture.dm.train_dataloader.return_value = mock_dataloader

        mock_optimizer_instance = MagicMock()
        mock_optimizer.return_value = mock_optimizer_instance

        stats = gym_fixture.train_epoch()
        assert isinstance(stats, dict)


def test_compute_recon_loss(gym_fixture):
    with patch("torch.utils.data.DataLoader") as mock_dataloader:
        mock_dataloader.return_value = [
            (torch.randn(5, 10), torch.randn(5, 10))
        ]
        gym_fixture.dm.test_dataloader.return_value = mock_dataloader

        loss = gym_fixture.compute_recon_loss(
            gym_fixture.model, mock_dataloader
        )
        assert isinstance(loss, torch.Tensor)


def test_latent_space(gym_fixture):
    mock_dataloader = MagicMock()
    mock_dataloader.return_value = [(torch.randn(5, 10), torch.randn(5, 10))]
    gym_fixture.dm.full_dataloader.return_value = mock_dataloader

    latent_space = gym_fixture.latent_space()
    assert isinstance(latent_space, np.ndarray)
