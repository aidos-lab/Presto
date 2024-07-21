from typing import Any, Protocol
from dataclasses import dataclass


#  ╭──────────────────────────────────────────────────────────╮
#  │ Transformer Configurations                               │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Transformer(Protocol):
    pass


@dataclass
class PretrainedLanguageModel(Transformer):
    pass


@dataclass
class ADA(PretrainedLanguageModel):
    pass


@dataclass
class MISTRAL(PretrainedLanguageModel):
    pass


@dataclass
class DISTILROBERTA(PretrainedLanguageModel):
    pass


@dataclass
class MINILM(PretrainedLanguageModel):
    pass


@dataclass
class MPNET(PretrainedLanguageModel):
    pass


@dataclass
class QA_DISTILBERT(PretrainedLanguageModel):
    pass


#  ╭──────────────────────────────────────────────────────────╮
#  │ Data Configurations                                      │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Embedding(Protocol):
    pass


@dataclass
class arXiv(Embedding):
    pass


@dataclass
class BBC(Embedding):
    pass


@dataclass
class CNN(Embedding):
    pass


@dataclass
class Patents(Embedding):
    pass


#  ╭──────────────────────────────────────────────────────────╮
#  │ Implementation Configurations                            │
#  ╰──────────────────────────────────────────────────────────╯


@dataclass
class Implementation(Protocol):
    pass
