import os
from abc import ABC, abstractmethod

from generators.config import Config
from omegaconf import OmegaConf

#  ╭──────────────────────────────────────────────────────────╮
#  │ Experiments                                              │
#  ╰──────────────────────────────────────────────────────────╯


class GeneratorModule(ABC):
    def __init__(self, config_file_path: str):
        self.params = self.read_params(config_file_path)

    @staticmethod
    def read_params(config_file_path: str) -> dict:
        assert os.path.isfile(
            config_file_path
        ), f"Config file not found at {config_file_path}"
        assert config_file_path.endswith([".yaml", ".yml"]), "Config a `yaml` file."

        return OmegaConf.load(config_file_path)

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
