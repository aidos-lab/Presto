from abc import ABC, abstractmethod
import pickle
import numpy as np

#  ╭──────────────────────────────────────────────────────────╮
#  │ Base Multiverse Generator Class                          │
#  ╰──────────────────────────────────────────────────────────╯


class Base(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def generate(self):
        pass
