from abc import ABC, abstractmethod


class BasePretrainedModel(ABC):
    """
    Abstract Base Class for Pretrained Transformer Models.

    Child classes must implement the following methods:
    - load_model
    - load_tokenizer
    - embed


    Parameters
    ----------
    config : ConfigType
        Configuration parameters for setting up the model. This can be a
        dictionary or an `ConfigType` object.

    Attributes
    ----------
    config : ConfigType
        Stores the configuration parameters provided during instantiation.

    Methods
    -------
    load_model()
        Abstract method for loading the pretrained model.
    load_tokenizer()
        Abstract method for loading the tokenizer.
    embed(text: str)
        Abstract method for embedding the input text.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def process_text(self):
        # Tokenize etc
        pass

    @abstractmethod
    def embed(self, text):
        pass
