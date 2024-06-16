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

    @staticmethod
    def write_pkl(data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def read(path):
        if path.endswith(".npz"):
            return np.load(path)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise NotImplementedError(f"File type not supported: {path}")
