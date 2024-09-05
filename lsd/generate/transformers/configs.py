from typing import Protocol
from dataclasses import dataclass


#  ╭──────────────────────────────────────────────────────────╮
#  │ Transformer Configurations                               │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class PretrainedLanguageModel(Protocol):
    module: str = "lsd.generate.transformers.models.pretrained"


@dataclass
class Ada(PretrainedLanguageModel):
    pass


@dataclass
class Mistral(PretrainedLanguageModel):
    module: str = "lsd.generate.transformers.models.huggingface"
    name: str = "Mistral"
    version: str = "v1"


@dataclass
class DistilRoberta(PretrainedLanguageModel):
    pass


@dataclass
class MiniLM(PretrainedLanguageModel):
    pass


@dataclass
class MPNET(PretrainedLanguageModel):
    pass


@dataclass
class QA_DistilBert(PretrainedLanguageModel):
    pass


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class HuggingFaceData(Protocol):
    name: str
    version: str
    split: str = "train"
    host: str = "huggingface"


@dataclass
class arXiv(HuggingFaceData):
    pass


@dataclass
class BBC(HuggingFaceData):
    pass


@dataclass
class CNN(HuggingFaceData):
    name: str = "cnn_daily_mail"
    version: str = "3.0.0"


@dataclass
class Patents(HuggingFaceData):
    pass


@dataclass
class LocalData(Protocol):
    name: str
    path: str
    host: str = "local"


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Implementation(Protocol):
    version: str


@dataclass
class Sampler(Implementation):
    pass


@dataclass
class Tokenizer(Implementation):
    name: str = "Tokenizer"
    version: str = "v1"
    aggregation: str = "mean"
