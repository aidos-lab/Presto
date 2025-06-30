import pytest
import numpy as np
import tempfile
import torch
from unittest.mock import Mock, patch

from lsd.generate.transformers.tf import Transformer
from lsd.generate.transformers.models.huggingface import HuggingFaceModel
from lsd.generate.transformers.models.sbert import SentenceTransformerModel
from lsd.generate.transformers.models.pretrained import BasePretrainedModel


class TestBasePretrainedModel:
    """Test cases for the BasePretrainedModel abstract class."""

    def test_base_model_instantiation(self):
        """Test that BasePretrainedModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePretrainedModel({})


class TestHuggingFaceModel:
    """Test cases for the HuggingFaceModel class."""

    @pytest.fixture
    def sample_config(self):
        return {"name": "distilbert-base-uncased"}

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        model = Mock()
        # Mock the output structure
        output = Mock()
        output.last_hidden_state = torch.randn(
            1, 4, 768
        )  # batch_size=1, seq_len=4, hidden_size=768
        model.return_value = output
        model.eval = Mock()
        return model

    @patch("lsd.generate.transformers.models.huggingface.AutoModel")
    @patch("lsd.generate.transformers.models.huggingface.AutoTokenizer")
    def test_huggingface_model_initialization(
        self, mock_tokenizer_class, mock_model_class, sample_config
    ):
        """Test HuggingFace model initialization."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()

        model = HuggingFaceModel(sample_config)

        assert model.config == sample_config
        assert model.model is not None
        assert model.tokenizer is not None
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased"
        )
        mock_model_class.from_pretrained.assert_called_once_with(
            "distilbert-base-uncased"
        )

    @patch("lsd.generate.transformers.models.huggingface.AutoModel")
    @patch("lsd.generate.transformers.models.huggingface.AutoTokenizer")
    def test_huggingface_model_embed(
        self, mock_tokenizer_class, mock_model_class, sample_config
    ):
        """Test text embedding functionality."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        output = Mock()
        output.last_hidden_state = torch.randn(1, 4, 768)
        mock_model.return_value = output
        mock_model.eval = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Test embedding
        model = HuggingFaceModel(sample_config)
        embeddings = model.embed("This is a test sentence.")

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == 1  # batch size
        assert embeddings.shape[1] == 768  # embedding dimension

    def test_process_text_single_string(self, sample_config):
        """Test text processing with a single string input."""
        with patch(
            "lsd.generate.transformers.models.huggingface.AutoModel"
        ), patch(
            "lsd.generate.transformers.models.huggingface.AutoTokenizer"
        ) as mock_tokenizer_class:

            mock_tokenizer = Mock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            model = HuggingFaceModel(sample_config)
            model.process_text("Hello world")

            mock_tokenizer.assert_called_once_with(
                ["Hello world"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

    def test_process_text_list_input(self, sample_config):
        """Test text processing with a list of strings."""
        with patch(
            "lsd.generate.transformers.models.huggingface.AutoModel"
        ), patch(
            "lsd.generate.transformers.models.huggingface.AutoTokenizer"
        ) as mock_tokenizer_class:

            mock_tokenizer = Mock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            model = HuggingFaceModel(sample_config)
            test_texts = ["Hello world", "This is a test"]
            model.process_text(test_texts)

            mock_tokenizer.assert_called_once_with(
                test_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )


class TestSentenceTransformerModel:
    """Test cases for the SentenceTransformerModel class."""

    @pytest.fixture
    def sample_config(self):
        return {"name": "all-MiniLM-L6-v2"}

    @patch("lsd.generate.transformers.models.sbert.ST")
    def test_sbert_model_initialization(self, mock_st_class, sample_config):
        """Test SentenceTransformer model initialization."""
        mock_st_instance = Mock()
        mock_st_class.return_value = mock_st_instance

        model = SentenceTransformerModel(sample_config)

        assert model.config == sample_config
        assert model.model == mock_st_instance
        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("lsd.generate.transformers.models.sbert.ST")
    def test_sbert_model_embed(self, mock_st_class, sample_config):
        """Test SentenceTransformer embedding functionality."""
        mock_st_instance = Mock()
        mock_st_instance.encode.return_value = torch.randn(384)
        mock_st_class.return_value = mock_st_instance

        model = SentenceTransformerModel(sample_config)
        embeddings = model.embed("This is a test sentence.")

        mock_st_instance.encode.assert_called_once_with(
            "This is a test sentence.", convert_to_tensor=True
        )
        assert embeddings is not None

    @patch("lsd.generate.transformers.models.sbert.ST")
    def test_sbert_model_process_text(self, mock_st_class, sample_config):
        """Test SentenceTransformer text processing."""
        mock_st_instance = Mock()
        mock_st_instance.tokenize.return_value = [1, 2, 3, 4, 5]
        mock_st_class.return_value = mock_st_instance

        model = SentenceTransformerModel(sample_config)
        tokens = model.process_text("Hello world")

        mock_st_instance.tokenize.assert_called_once_with("Hello world")
        assert tokens == [1, 2, 3, 4, 5]


class TestTransformer:
    """Test cases for the Transformer class."""

    @pytest.fixture
    def sample_params(self):
        return {
            "experiment": "test_experiment",
            "file": "test_universe_0.yml",
            "data_choices": {
                "name": "cnn_dailymail",
                "version": "3.0.0",
                "split": "train",
                "host": "huggingface",
                "num_samples": 100,
            },
            "model_choices": {
                "name": "DistilBERT",
                "module": "lsd.generate.transformers.models.huggingface",
            },
            "implementation_choices": {
                "name": "Tokenizer",
                "module": "standard_tokenizer",
            },
        }

    @patch("lsd.generate.transformers.tf.load_dataset")
    @patch("lsd.generate.transformers.tf.importlib")
    def test_transformer_initialization(
        self, mock_importlib, mock_load_dataset, sample_params
    ):
        """Test Transformer class initialization."""
        # Mock dataset loading
        mock_dataset = Mock()
        mock_dataset.select.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset

        # Mock model loading
        mock_module = Mock()
        mock_model_class = Mock()
        mock_module.initialize.return_value = mock_model_class
        mock_importlib.import_module.return_value = mock_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_params["outDir"] = tmp_dir
            transformer = Transformer(sample_params)

            assert transformer.dataset == mock_dataset
            assert transformer.model == mock_model_class

    @patch("lsd.generate.transformers.tf.load_dataset")
    @patch("lsd.generate.transformers.tf.importlib")
    def test_transformer_generate(
        self, mock_importlib, mock_load_dataset, sample_params
    ):
        """Test Transformer generate functionality."""
        # Mock dataset
        mock_dataset = [
            {"article": "First test article"},
            {"article": "Second test article"},
        ]
        mock_load_dataset.return_value = mock_dataset

        # Mock model
        mock_model_instance = Mock()
        mock_embeddings = torch.randn(2, 384)  # 2 samples, 384 dimensions
        mock_model_instance.embed.return_value = mock_embeddings

        mock_model_class = Mock()
        mock_model_class.return_value = mock_model_instance

        mock_module = Mock()
        mock_module.initialize.return_value = mock_model_class
        mock_importlib.import_module.return_value = mock_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_params["outDir"] = tmp_dir
            transformer = Transformer(sample_params)
            transformer.dataset = mock_dataset  # Override with our mock

            embeddings = transformer.generate()

            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (2, 384)

    def test_extract_texts_from_dataset_huggingface_format(self):
        """Test text extraction from HuggingFace dataset format."""
        transformer = Transformer.__new__(
            Transformer
        )  # Create without calling __init__

        # Mock dataset with articles
        transformer.dataset = [
            {"article": "First article text"},
            {"article": "Second article text"},
            {"article": "Third article text"},
        ]

        texts = transformer._extract_texts_from_dataset()
        expected = [
            "First article text",
            "Second article text",
            "Third article text",
        ]
        assert texts == expected

    def test_extract_texts_from_dataset_text_format(self):
        """Test text extraction from text field format."""
        transformer = Transformer.__new__(
            Transformer
        )  # Create without calling __init__

        # Mock dataset with text field
        transformer.dataset = [
            {"text": "First text sample"},
            {"text": "Second text sample"},
        ]

        texts = transformer._extract_texts_from_dataset()
        expected = ["First text sample", "Second text sample"]
        assert texts == expected

    def test_extract_texts_from_dataset_list_format(self):
        """Test text extraction from simple list format."""
        transformer = Transformer.__new__(
            Transformer
        )  # Create without calling __init__

        # Mock dataset as simple list
        transformer.dataset = ["First text", "Second text", "Third text"]

        texts = transformer._extract_texts_from_dataset()
        expected = ["First text", "Second text", "Third text"]
        assert texts == expected

    def test_extract_texts_from_dataset_empty_dataset(self):
        """Test error handling for empty datasets."""
        transformer = Transformer.__new__(
            Transformer
        )  # Create without calling __init__
        transformer.dataset = []

        with pytest.raises(
            ValueError, match="No text content found in dataset"
        ):
            transformer._extract_texts_from_dataset()

    def test_extract_texts_from_dataset_no_text_fields(self):
        """Test error handling when no text fields are found."""
        transformer = Transformer.__new__(
            Transformer
        )  # Create without calling __init__

        # Mock dataset with no recognizable text fields
        transformer.dataset = [
            {"id": 1, "label": "positive"},
            {"id": 2, "label": "negative"},
        ]

        with pytest.raises(
            ValueError, match="No text content found in dataset"
        ):
            transformer._extract_texts_from_dataset()


class TestTransformerIntegration:
    """Integration tests for transformer functionality."""

    def test_initialize_functions(self):
        """Test that initialize functions return the correct classes."""
        from lsd.generate.transformers.models.huggingface import (
            initialize as hf_init,
        )
        from lsd.generate.transformers.models.sbert import (
            initialize as sbert_init,
        )

        assert hf_init() == HuggingFaceModel
        assert sbert_init() == SentenceTransformerModel

    def test_transformer_configs_exist(self):
        """Test that required transformer configurations exist."""
        from lsd.generate.transformers.configs import (
            Mistral,
            Ada,
            MiniLM,
            arXiv,
            CNN,
            Tokenizer,
        )

        # Test that config classes can be instantiated
        mistral = Mistral()
        ada = Ada()
        mini_lm = MiniLM()
        arxiv = arXiv()
        cnn = CNN()
        tokenizer = Tokenizer()

        # Test expected default values
        assert mistral.module == "lsd.generate.transformers.models.huggingface"
        assert mistral.name == "Mistral"
        assert mistral.version == "v1"

        assert cnn.name == "cnn_daily_mail"
        assert cnn.version == "3.0.0"

        assert tokenizer.name == "Tokenizer"
        assert tokenizer.version == "v1"
        assert tokenizer.aggregation == "mean"
