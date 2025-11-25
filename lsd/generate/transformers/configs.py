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
    module: str = "lsd.generate.transformers.models.huggingface"
    name: str = "Ada"
    version: str = "v1"


@dataclass
class Mistral(PretrainedLanguageModel):
    module: str = "lsd.generate.transformers.models.huggingface"
    name: str = "Mistral"
    version: str = "v1"


@dataclass
class DistilRoberta(PretrainedLanguageModel):
    module: str = "lsd.generate.transformers.models.huggingface"
    name: str = "distilroberta-base"
    version: str = "v1"


@dataclass
class MiniLM(PretrainedLanguageModel):
    module: str = "lsd.generate.transformers.models.sbert"
    name: str = "sentence-transformers/all-MiniLM-L6-v2"
    version: str = "v1"


@dataclass
class MPNET(PretrainedLanguageModel):
    module: str = "lsd.generate.transformers.models.sbert"
    name: str = "sentence-transformers/all-mpnet-base-v2"
    version: str = "v1"


@dataclass
class QA_DistilBert(PretrainedLanguageModel):
    module: str = "lsd.generate.transformers.models.huggingface"
    name: str = "distilbert-base-cased-distilled-squad"
    version: str = "v1"


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
    name: str = "arxiv"
    version: str = "1.0.0"


@dataclass
class BBC(HuggingFaceData):
    name: str = "bbc"
    version: str = "1.0.0"


@dataclass
class CNN(HuggingFaceData):
    name: str = "cnn_daily_mail"
    version: str = "3.0.0"


@dataclass
class Patents(HuggingFaceData):
    name: str = "patents"
    version: str = "1.0.0"


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
