"Latent Space Designer: Let LSD take you into the multiverse"
import os
import importlib
import omegaconf
from itertools import product

import lsd.config as config
import lsd.utils as ut


class LSD:
    def __init__(
        self,
        multiverseConfig,
        outDir: str,
        experimentName: str = None,
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

        assert os.path.isdir(outDir), "Output directory must be specified."
        self.outDir = outDir

        if experimentName:
            self.experimentName = experimentName
        else:
            self.experimentName = self.cfg.base + ut.temporal_id()

    def __repr__(self):
        return f"Latent Space Designer: {self.cfg.__name__}"

    def __str__(self):
        return f"Latent Space Designer: Let LSD take you into the multiverse"

    @property
    def multiverse(self):
        if self._multiverse is None:
            self._load_multiverse()

        return self._multiverse

    @property
    def generators(self):
        return {
            label: sorted(list(self.multiverse[label].keys()))
            for label in self.multiverse_labels
        }

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
            if multiverse[label]:
                self._multiverse[label] = multiverse[label][self.cfg.base]
            else:
                self._multiverse[label] = {}

    def design(self):
        """Read in Multiverse configuration, create a direct product, and write individual config files for each model in the multiverse."""

        if self._multiverse is None:
            self._load_multiverse()

        module = importlib.import_module(self.cfg.module)
        universes = []
        for label, generators in self.generators.items():
            configs = []
            for G in generators:
                gen_cfg = getattr(module, G)
                params, vectors = self._design_parameter_space(
                    gen_cfg, label, G
                )
                for theta in vectors:
                    data = dict(zip(params, theta))
                    configs.append(ut.LoadClass.instantiate(gen_cfg, data))
            universes.append(configs)

        for i, U in enumerate(product(*universes)):
            universe = config.Universe(*U)
            outPath = os.path.join(
                self.outDir, f"{self.experimentName}/configs/universe_{i}.yml"
            )
            self.write_cfg(outPath, universe)

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

    def _design_parameter_space(self, cfg, label, generator):
        user_params = self.multiverse[label][generator].keys()
        params = self._intersection(cfg.__annotations__, user_params)
        vectors = self._cartesian_product(
            label,
            generator,
            params,
        )
        return params, vectors

    def _cartesian_product(self, label: str, generator: str, parameters: list):

        return product(
            *[
                self.multiverse[label][generator][param]
                for param in parameters
                if generator in self.multiverse[label]
                and param in self.multiverse[label][generator].keys()
            ]
        )

    @staticmethod
    def _intersection(x, y):
        return sorted(list(set(x).intersection(y)))

    @staticmethod
    def read_params(path: str):
        assert path.endswith((".yaml", ".yml")), "File must be a `yaml` file."
        assert os.path.isfile(path), f"File not found at {path}"
        with open(path, "r") as f:
            content = omegaconf.OmegaConf.load(f)
        return content

    @staticmethod
    def write_cfg(config_file_path: str, cfg):
        assert config_file_path.endswith(
            (".yaml", ".yml")
        ), "Config must be a `yaml` file."
        if not os.path.exists(os.path.dirname(config_file_path)):
            os.makedirs(os.path.dirname(config_file_path))
        c = omegaconf.OmegaConf.create(cfg)
        with open(config_file_path, "w") as f:
            omegaconf.OmegaConf.save(c, f)

    @staticmethod
    def filter_params(
        path: str,
        choices: str,
        base: str,
    ):
        content = LSD.read_params(path)
        return content.get(choices).get(base)
