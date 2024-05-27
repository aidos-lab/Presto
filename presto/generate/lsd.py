"Latent Space Designer: Let LSD take you into the multiverse"
import importlib
import os

from omegaconf import OmegaConf

import presto.generate.config as config
from presto.generate.log import Logger


class LSD:
    def __init__(self, config_file_path: str):
        self.params = self.read_params(config_file_path)
        self.registry = {}
        self.multiverse = None

        # self.logger = Logger()

    def load_generator(self):
        cfg = self.load_config(self.params["generator"])
        try:
            module = importlib.import_module(cfg.module)
            module.initialize(self.registry)
            return self.registry["generator"]
        except KeyError:
            raise ValueError(f"Module {config.name} not found!")

    def design(self):
        """Find correct experiment class, instantiate it and execute setup method to generate proper cartesian product"""

        generator = self.load_generator()

        # Writes files for each item in the multiverse
        generator(self.params)

        # self.logger = Logger(
        #     exp="dev",
        #     name="multiverse",
        #     dev=False,
        #     out_file=True,
        # )
        pass

    def generate(self):

        self.logger.log(msg="Starting LSD")
        self.logger.log(msg="Reading parameters")

        # Start a logger
        # read parameter driver, log parameters
        # find the correct experiment class
        # instantiate the experiment class
        # execute the setup method to write necessary config files
        # execute the train method (might be trivial) (iterating over config files)
        # execute the generate method to generate and save latent spaces

        pass

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
    def load_config(generator_name: str):
        config_name = config.generator_mapping[generator_name]
        cfg = getattr(config, config_name)
        return cfg
