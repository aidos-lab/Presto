from abc import ABC, abstractmethod

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


# base experiment class with empty methods for training/generating =>
# These then get specified by the particular experiment class e.g. autoencoders that has a specific training method.
# Generalized setup script that creates cartesian product, and writes config files for each experiment.
# main should then load a config, from this determine which experiment class to instantiate, and execute the train method.
