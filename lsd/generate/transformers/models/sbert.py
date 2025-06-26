from lsd.generate.transformers.models.pretrained import BasePretrainedModel
from sentence_transformers import SentenceTransformer as ST


class SentenceTransformerModel(BasePretrainedModel):
    """
    Implementation of a Pretrained Model using SBERT's SentenceTransformer.

    This class handles loading the model, processing text, and embedding
    sentences using the SentenceTransformer from the SBERT library.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model-related parameters.

    Attributes
    ----------
    config : dict
        Stores the configuration parameters provided during instantiation.
    model : SentenceTransformer
        The loaded SentenceTransformer model.

    Methods
    -------
    load_model()
        Loads the SentenceTransformer model specified in the config.
    process_text(text: str)
        Tokenizes and processes the input text.
    embed(text: str)
        Generates embeddings for the input text.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Loads the SentenceTransformer model specified in the config.

        The model is loaded from the `model_name` key in the config dictionary.
        """
        model_name = self.config.get("name", "all-MiniLM-L6-v2")
        self.model = ST(model_name)

    def process_text(self, text: str):
        """
        Processes the text by tokenizing it using the SentenceTransformer's tokenizer.

        Parameters
        ----------
        text : str
            The input text to be processed.

        Returns
        -------
        list
            The tokenized version of the input text.
        """
        return self.model.tokenize(text)

    def embed(self, text: str):
        """
        Generates embeddings for the input text using the loaded SentenceTransformer model.

        Parameters
        ----------
        text : str
            The input text to be embedded.

        Returns
        -------
        numpy.ndarray
            The generated embeddings for the input text.
        """
        return self.model.encode(text, convert_to_tensor=True)


def initialize():
    """
    Initializes and returns an instance of the SentenceTransformerModel.

    Returns
    -------
    SentenceTransformerModel
        An instance of the SentenceTransformerModel class.
    """
    return SentenceTransformerModel
