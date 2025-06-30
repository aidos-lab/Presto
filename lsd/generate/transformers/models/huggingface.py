from transformers import AutoModel, AutoTokenizer
import torch
from typing import Union, List

from lsd.generate.transformers.models.pretrained import BasePretrainedModel


class HuggingFaceModel(BasePretrainedModel):
    """
    Implementation of a Pretrained Model using HuggingFace Transformers.

    This class handles loading transformer models from HuggingFace Hub,
    processing text, and generating embeddings using the transformer's
    hidden states.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model-related parameters.
        Expected keys:
        - 'name': Model name/identifier on HuggingFace Hub
        - 'version': Version of the model (optional)

    Attributes
    ----------
    config : dict
        Stores the configuration parameters provided during instantiation.
    model : AutoModel
        The loaded HuggingFace transformer model.
    tokenizer : AutoTokenizer
        The corresponding tokenizer for the model.

    Methods
    -------
    load_model()
        Loads the HuggingFace transformer model and tokenizer.
    process_text(text: Union[str, List[str]])
        Tokenizes and processes the input text.
    embed(text: Union[str, List[str]])
        Generates embeddings for the input text using the transformer.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """
        Loads the HuggingFace transformer model and tokenizer.

        The model is loaded from the `name` key in the config dictionary.
        Defaults to 'distilbert-base-uncased' if no name is provided.
        """
        model_name = self.config.get("name", "distilbert-base-uncased")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    def process_text(self, text: Union[str, List[str]]):
        """
        Processes the text by tokenizing it using the HuggingFace tokenizer.

        Parameters
        ----------
        text : Union[str, List[str]]
            The input text to be processed. Can be a single string or list of strings.

        Returns
        -------
        dict
            The tokenized version of the input text with attention masks.
        """
        if isinstance(text, str):
            text = [text]
        
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

    def embed(self, text: Union[str, List[str]]):
        """
        Generates embeddings for the input text using the loaded HuggingFace model.

        Parameters
        ----------
        text : Union[str, List[str]]
            The input text to be embedded. Can be a single string or list of strings.

        Returns
        -------
        torch.Tensor
            The generated embeddings for the input text. Uses mean pooling
            of the last hidden states.
        """
        # Process text
        inputs = self.process_text(text)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get last hidden states
            last_hidden_states = outputs.last_hidden_state
            
            # Apply mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            
            # Expand attention mask to match hidden states dimensions
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            
            # Apply mask and compute mean
            masked_embeddings = last_hidden_states * attention_mask_expanded
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            
            # Mean pooling
            embeddings = summed_embeddings / summed_mask
            
        return embeddings


def initialize():
    """
    Initializes and returns an instance of the HuggingFaceModel.

    Returns
    -------
    HuggingFaceModel
        An instance of the HuggingFaceModel class.
    """
    return HuggingFaceModel
