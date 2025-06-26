from transformers import AutoModel, AutoTokenizer

from lsd.generate.transformers.models.pretrained import BasePretrainedModel


class HuggingFaceModel(BasePretrainedModel):
    """

    Args:
        BasePretrainedModel (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)


def initialize():
    return HuggingFaceModel
