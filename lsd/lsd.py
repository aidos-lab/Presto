"Latent Space Designer: Let LSD take you into the multiverse"
import os
import importlib
from itertools import product
from omegaconf import OmegaConf

import lsd.config as config


class LSD:
    def __init__(self, multiverse):
        try:
            self.cfg = getattr(config, multiverse)
        except AttributeError:
            raise ValueError(
                f"{multiverse} Configuration Class not found! Please check `config.py` for available configurations."
            )
        self._multiverse = None
        self._D = None
        self._M = None
        self._I = None

    def __repr__(self):
        return f"Latent Space Designer: {self.cfg.__name__}"

    def __str__(self):
        return f"Latent Space Designer: Let LSD take you into the multiverse"

    @property
    def multiverse(self):
        if self._multiverse is None:
            self.load_multiverse()
        return self._multiverse

    @property
    def D(self):
        if self._D is None:
            self.load_multiverse()
        return self._D

    @property
    def M(self):
        if self._M is None:
            self.load_multiverse()
        return self._M

    @property
    def I(self):
        if self._I is None:
            self.load_multiverse()
        return self._I

    def load_multiverse(self):
        # Maybe can get this elegantly while checking these exist in the config
        reference = ["data_choices", "model_choices", "implementation_choices"]

        self._D = self.filter_params(
            self.cfg.data_choices, reference[0], self.cfg.id_
        )
        self._M = self.filter_params(
            self.cfg.model_choices, reference[1], self.cfg.id_
        )
        self._I = self.filter_params(
            self.cfg.implementation_choices,
            reference[2],
            self.cfg.id_,
        )

        assert (
            self._D and self._M and self._I
        ), "Empty Multiverse. Please check your configuration files."

        self._multiverse = {}
        for choice, tag in zip(
            [self.D, self.M, self.I],
            reference,
        ):
            choices = {}
            for generator, parameters in choice.items():
                keys = list(parameters.keys())
                vectors = list(product(*list(parameters.values())))
                choices[generator] = (keys, vectors)
            self._multiverse[tag] = choices

    def design(self):
        """Read in Multiverse configuration, create a direct product, and write individual config files for each model in the multiverse."""
        pass

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

        return OmegaConf.load(config_file_path)

    @staticmethod
    def load_params(path: str):
        assert path.endswith((".yaml", ".yml")), "File must be a `yaml` file."
        assert os.path.isfile(path), f"File not found at {path}"
        with open(path, "r") as f:
            content = OmegaConf.load(f)
        return content

    @staticmethod
    def filter_params(
        path: str,
        choices: str,
        id_: str,
    ):
        content = LSD.load_params(path)
        return content.get(choices).get(id_)
