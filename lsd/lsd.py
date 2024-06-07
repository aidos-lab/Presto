"Latent Space Designer: Let LSD take you into the multiverse"
import os
import importlib
import omegaconf
from itertools import product

import lsd.config as config


class LSD:
    def __init__(
        self,
        multiverseConfig,
        labels: list = [
            "data_choices",
            "model_choices",
            "implementation_choices",
        ],
    ):
        try:
            self.cfg = getattr(config, multiverseConfig)
        except AttributeError:
            raise ValueError(
                f"{multiverseConfig} Configuration Class not found! Please check `config.py` for available Configs."
            )
        for label in labels:
            assert hasattr(
                self.cfg, label
            ), f"Configuration must have {label} attribute."
            self.__setattr__(label, None)
        self.multiverse_labels = labels
        self._multiverse = None

    def __repr__(self):
        return f"Latent Space Designer: {self.cfg.__name__}"

    def __str__(self):
        return f"Latent Space Designer: Let LSD take you into the multiverse"

    @property
    def multiverse(self):
        if self._multiverse is None:
            self._load_multiverse()

        return self._multiverse

    @multiverse.setter
    def multiverse(self, multiverse):
        assert isinstance(multiverse, dict) or isinstance(
            multiverse, omegaconf.dictconfig.DictConfig
        ), "Multiverse must be a dictionary."
        self._multiverse = {}
        for label in self.multiverse_labels:
            assert (
                label in multiverse.keys()
            ), f"Multiverse must have {label} key."
            self.__setattr__(label, multiverse[label])
            self._multiverse[label] = multiverse[label][self.cfg.base]

    def design(self):
        """Read in Multiverse configuration, create a direct product, and write individual config files for each model in the multiverse."""

        if self._multiverse is None:
            self._load_multiverse()

        print(self._multiverse)
        generators = product(*list(self._multiverse.values()))

        module = importlib.import_module(self.cfg.module)
        for D, M, I in generators:
            Dconfig = getattr(module, D)
            Mconfig = getattr(module, M)
            Iconfig = getattr(module, I)

            print(Dconfig.name, Mconfig.name, Iconfig.name)

        base = getattr(module, self.cfg.base)
        print(base)

        # Given 3 config objects, assign each a vector in the

    def generate(self):

        self.logger.log(msg="Starting LSD")
        self.logger.log(msg="Reading parameters")

        # NOTE: Old idea
        # Start a logger
        # read parameter driver, log parameters
        # find the correct experiment class
        # instantiate the experiment class
        # execute the setup method to write necessary config files
        # execute the train method (might be trivial) (iterating over config files)
        # execute the generate method to generate and save latent spaces

        # TODO:
        # for config in generated_experiment_folder:
        # read in

        pass

    def _load_multiverse(self):
        self._multiverse = {}
        for label in self.multiverse_labels:
            params = self.filter_params(
                getattr(self.cfg, label),
                label,
                self.cfg.base,
            )
            self.__setattr__(label, params)
            self._multiverse[label] = params

    @staticmethod
    def load_generator(module):
        return importlib.import_module(module).initialize()

    @staticmethod
    def read_params(config_file_path: str) -> dict:
        assert os.path.isfile(
            config_file_path
        ), f"Config file not found at {config_file_path}"
        assert config_file_path.endswith(
            (".yaml", ".yml")
        ), "Config must be a `yaml` file."

        return omegaconf.OmegaConf.load(config_file_path)

    @staticmethod
    def load_params(path: str):
        assert path.endswith((".yaml", ".yml")), "File must be a `yaml` file."
        assert os.path.isfile(path), f"File not found at {path}"
        with open(path, "r") as f:
            content = omegaconf.OmegaConf.load(f)
        return content

    @staticmethod
    def filter_params(
        path: str,
        choices: str,
        base: str,
    ):
        content = LSD.load_params(path)
        return content.get(choices).get(base)
