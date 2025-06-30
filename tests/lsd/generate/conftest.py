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


# Transformer-specific fixtures
@pytest.fixture
def mock_huggingface_model():
    """Mock HuggingFace model for testing."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    
    # Mock output structure
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(1, 10, 768)  # batch=1, seq_len=10, hidden=768
    mock_model.return_value = mock_output
    
    return mock_model


@pytest.fixture
def mock_huggingface_tokenizer():
    """Mock HuggingFace tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[101, 2023, 2003, 2019, 2742, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    return mock_tokenizer


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model for testing."""
    mock_st = MagicMock()
    mock_st.encode.return_value = torch.randn(384)  # Standard embedding size
    mock_st.tokenize.return_value = [101, 2023, 2003, 102]
    return mock_st


@pytest.fixture
def sample_text_dataset():
    """Sample text dataset for testing."""
    return [
        {"article": "This is the first article about machine learning."},
        {"article": "This is the second article about deep learning."},
        {"text": "This is a text sample for testing purposes."},
        {"sentence": "This is a sentence for testing."}
    ]
