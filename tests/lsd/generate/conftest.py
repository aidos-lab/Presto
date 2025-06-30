import pytest
from unittest.mock import patch, MagicMock
import torch

from lsd.generate.autoencoders.gym import Gym
from lsd.generate.autoencoders.logger import Logger


# Fixture for Logger initialization
@pytest.fixture
def logger_fixture(tmp_path):
    return Logger(
        exp=tmp_path,
        name="test_run",
        wandb_logging=False,
        out_file=True,
    )


@pytest.fixture
def gym_fixture():
    config = MagicMock()
    config.model = "mock_model_module"
    config.dataset = "mock_dataset_module"
    config.optimizer = "mock_optimizer_module"
    config.epochs = 10
    config.clip_max_norm = 5.0

    mock_logger = MagicMock()

    with patch("importlib.import_module") as mock_import_module:
        mock_module = MagicMock()
        mock_import_module.return_value.initialize.return_value = mock_module
        return Gym(config=config, logger=mock_logger)
