"Latent Space Designer: Let LSD take you into the multiverse"
import os
import importlib
import omegaconf
from itertools import product
from dataclasses import fields
import shutil

import lsd.config as config
import lsd.utils as ut


class LSD:
    """
    Latent Space Designer: Let LSD take you into the multiverse.

    The LSD class is the main interface for designing and generating latent spaces for various generative models. In particular, LSD lets the user generate a multiverse of embeddings based on different data, modeling, and implementation choices. The class is designed to be extensible and can be used with any configuration class that defines the multiverse components.

    Parameters
    ----------
    multiverseConfig : str
        The name of the configuration class for the multiverse. This class should be defined in the `config.py` file.
    outDir : str
        Output directory to store the generated files.
    experimentName : str, optional
        Name for the experiment, by default None. If not provided, LSD will generate a unique name.
    labels : list, optional
        Labels for different choices in the multiverse, by default ["data_choices", "model_choices", "implementation_choices"]. These labels should match the labels in the yaml configurations.

    Attributes
    ----------
    cfg : omegaconf.dictconfig.DictConfig
        Configuration object for the multiverse.
    multiverse_labels : list
        List of labels for multiverse components.
    _multiverse : dict
        Internal storage for the multiverse components.
    design_path : str
        Path where the designed configurations are stored.
    outDir : str
        Full output directory path including experiment name.
    experimentName : str
        Name of the experiment.

    Properties
    ----------
    multiverse : dict
        The multiverse components loaded from the configuration.
    generators : dict
        A dictionary of generators for each label in the multiverse.
    experiment : str
        The output directory for the experiment.
    configs : str
        The directory path for the generated configuration files.
    latent_spaces : str
        The directory path for the generated latent spaces.
    models : str
        The directory path for the generated models.
    logs : str
        The directory path for the generated logs.

    Methods
    -------
    design()
        Read in Multiverse configuration, create a direct product, and write individual config files for each model in the multiverse.
    generate()
        Generate representations for each configuration in the design path.

    """

    def __init__(
        self,
        multiverseConfig,
        outDir: str,
        experimentName: str = None,
        labels: list = None,
    ):
        if labels is None:
            labels = ["data_choices", "model_choices", "implementation_choices"]

        self.cfg = self._get_config(multiverseConfig)
        self.multiverse_labels = labels
        self._validate_labels()
        self._validate_outDir(outDir)

        self.experimentName = (
            experimentName
            if experimentName
            else self._generate_experiment_name()
        )
        self.outDir = os.path.join(outDir, self.experimentName)
        self._multiverse = None
        self.design_path = None

    def __repr__(self):
        return f"Latent Space Designer: {self.cfg.__name__}"

    def __str__(self):
        return "Latent Space Designer: Let LSD take you into the multiverse"

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Getters & Setters                                        │
    #  ╰──────────────────────────────────────────────────────────╯
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
        self._validate_multiverse(multiverse)
        self._multiverse = {}
        for label in self.multiverse_labels:
            label_config = multiverse.get(label, {})
            if label_config is None:
                label_config = {}
            self.__setattr__(label, label_config)
            self._multiverse[label] = label_config.get(self.cfg.base, {})

    @property
    def experiment(self):
        return self.outDir

    @property
    def configs(self):
        return self._validate_directory(os.path.join(self.outDir, "configs/"))

    @property
    def latent_spaces(self):
        return self._validate_directory(
            os.path.join(self.outDir, "latent_spaces/")
        )

    @property
    def models(self):
        return self._validate_directory(os.path.join(self.outDir, "models/"))

    @property
    def logs(self):
        return self._validate_directory(os.path.join(self.outDir, "logs/"))

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Key Members: Design and Generate                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def design(self):
        """
        Read in Multiverse configuration, create a direct product, and write individual config files for each model in the multiverse.
        """
        if self._multiverse is None:
            self._load_multiverse()

        self._prepare_design_path()

        self.module = importlib.import_module(self.cfg.module)
        universes = self._generate_universe_configs()

        self._save_universe_configs(universes)

    def generate(self):
        """
        Generate representations for each configuration in the design path.
        """
        if self.design_path is None:
            self.design()

        base = getattr(self.module, self.cfg.base)

        print(
            f"Generating representations for {self.multiverse_size} universes..."
        )
        for cfg in sorted(os.listdir(self.design_path), key=ut.file_id_sorter):
            if cfg.startswith("universe_") and cfg.endswith(".yml"):
                cfg_path = os.path.join(self.design_path, cfg)
                self._generate(self.outDir, cfg_path, base)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helpers                                                  │
    #  ╰──────────────────────────────────────────────────────────╯

    @staticmethod
    def _generate(outDir, cfg_path, base):
        """
        Generate data/models based on the configuration.

        Parameters
        ----------
        outDir : str
            Output directory for generated files.
        cfg_path : str
            Path to the configuration file.
        base : Callable
            Base function or class to generate data/models.
        """
        theta = omegaconf.OmegaConf.load(cfg_path)
        theta.experiment = outDir
        theta.file = cfg_path
        M = base(theta)
        M.train()
        M.generate()
        del M

    def _prepare_design_path(self):
        """Prepare the directory for storing design configurations."""
        configs_outDir = os.path.join(self.outDir, "configs/")
        if os.path.isdir(configs_outDir):
            self.clean()
        self.design_path = configs_outDir

    def _generate_universe_configs(self):
        """Generate configuration combinations for the multiverse."""
        universes = []
        for label, generators in self.generators.items():
            configs = self._generate_configurations(generators, label)
            universes.append(configs)
        return universes

    def _generate_configurations(self, generators, label):
        """Generate configurations for each generator in a label."""
        configs = []
        for G in generators:
            gen_cfg = getattr(self.module, G)
            params, vectors = self._design_parameter_space(gen_cfg, label, G)
            configs.extend(
                self._generate_single_config(params, vectors, gen_cfg)
            )
        return configs

    def _generate_single_config(self, params, vectors, gen_cfg):
        """Generate a single configuration based on parameters and vectors."""
        return [
            ut.LoadClass.instantiate(gen_cfg, dict(zip(params, theta)))
            for theta in vectors
        ]

    def _save_universe_configs(self, universes):
        """Save the generated universe configurations."""
        self.multiverse_size = 0
        for i, U in enumerate(product(*universes)):
            universe = config.Universe(*U)
            outPath = os.path.join(self.design_path, f"universe_{i}.yml")
            self.write_cfg(outPath, universe)
            self.multiverse_size += 1

    def _load_multiverse(self):
        """Load and parse the multiverse configuration."""
        self._multiverse = {}
        for label in self.multiverse_labels:
            params = self.filter_params(
                getattr(self.cfg, label), label, self.cfg.base
            )
            self.__setattr__(label, params)
            self._multiverse[label] = params

    def _design_parameter_space(self, cfg, label, generator):
        """
        Design the parameter space for a given configuration.

        Parameters
        ----------
        cfg : dataclass
            Configuration dataclass for a generator.
        label : str
            Label for the generator.
        generator : str
            Generator name.

        Returns
        -------
        tuple
            Parameters and vectors for the configuration.
        """
        user_params = self.multiverse[label][generator].keys()
        cfg_params = [f.name for f in fields(cfg)]
        params = self._intersection(cfg_params, user_params)
        vectors = self._cartesian_product(label, generator, params)
        return params, vectors

    def _cartesian_product(self, label, generator, parameters):
        """
        Generate the cartesian product of parameters for a given generator.

        Parameters
        ----------
        label : str
            Label for the generator.
        generator : str
            Generator name.
        parameters : list
            List of parameter names.

        Returns
        -------
        itertools.product
            Cartesian product of parameter values.
        """
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
        """
        Find the intersection of two lists and return a sorted list.

        Parameters
        ----------
        x : list
            First list of elements.
        y : list
            Second list of elements.

        Returns
        -------
        list
            Sorted list containing the intersection of the two input lists.
        """
        return sorted(list(set(x).intersection(y)))

    @staticmethod
    def read_params(path: str):
        """
        Read parameters from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        omegaconf.dictconfig.DictConfig
            Loaded parameters as a DictConfig object.

        Raises
        ------
        AssertionError
            If the file does not end with `.yaml` or `.yml`.
            If the file is not found at the specified path.
        """
        assert path.endswith((".yaml", ".yml")), "File must be a `yaml` file."
        assert os.path.isfile(path), f"File not found at {path}"
        with open(path, "r") as f:
            content = omegaconf.OmegaConf.load(f)
        return content

    @staticmethod
    def write_cfg(config_file_path: str, cfg):
        """
        Write configuration to a YAML file.

        Parameters
        ----------
        config_file_path : str
            Path to the output YAML file.
        cfg : dict or omegaconf.dictconfig.DictConfig
            Configuration data to be written to the file.

        Raises
        ------
        AssertionError
            If the output file does not end with `.yaml` or `.yml`.
        """
        assert config_file_path.endswith(
            (".yaml", ".yml")
        ), "Config must be a `yaml` file."
        if not os.path.exists(os.path.dirname(config_file_path)):
            os.makedirs(os.path.dirname(config_file_path))
        c = omegaconf.OmegaConf.create(cfg)
        with open(config_file_path, "w") as f:
            omegaconf.OmegaConf.save(c, f)

    @staticmethod
    def filter_params(path: str, choices: str, base: str):
        """
        Filter parameters from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.
        choices : str
            Choices key in the YAML file.
        base : str
            Base key in the YAML file.

        Returns
        -------
        dict
            Filtered parameters based on the choices and base keys.
        """
        content = LSD.read_params(path)
        return content.get(choices, {}).get(base, {})

    def _get_config(self, config_name):
        """
        Retrieve the configuration object from the config module.

        Parameters
        ----------
        config_name : str
            Name of the configuration class.

        Returns
        -------
        omegaconf.dictconfig.DictConfig
            Configuration object.

        Raises
        ------
        ValueError
            If the configuration class is not found.
        """
        try:
            return getattr(config, config_name)
        except AttributeError:
            raise ValueError(
                f"{config_name} Configuration Class not found! Please check `config.py` for available Configs."
            )

    def _validate_labels(self):
        """
        Ensure the configuration contains all required labels.

        Raises
        ------
        ValueError
            If the configuration does not have the required label attributes.
        """
        for label in self.multiverse_labels:
            if not hasattr(self.cfg, label):
                raise ValueError(f"Configuration must have {label} attribute.")

    def _validate_outDir(self, outDir):
        """
        Ensure the output directory exists.

        Parameters
        ----------
        outDir : str
            Path to the output directory.

        Raises
        ------
        ValueError
            If the output directory does not exist.
        """
        if not os.path.isdir(outDir):
            raise ValueError(f"Must specify an existing output directory.")

    def _generate_experiment_name(self):
        """
        Generate a unique experiment name.

        Returns
        -------
        str
            Unique experiment name.
        """
        return self.cfg.base + ut.temporal_id()

    def _validate_multiverse(self, multiverse):
        """
        Validate the multiverse structure.

        Parameters
        ----------
        multiverse : dict or omegaconf.dictconfig.DictConfig
            Multiverse configuration.

        Raises
        ------
        TypeError
            If the multiverse is not a dictionary or DictConfig.
        ValueError
            If the multiverse does not contain required labels.
        """
        if not isinstance(multiverse, (dict, omegaconf.dictconfig.DictConfig)):
            raise TypeError("Multiverse must be a dictionary.")
        for label in self.multiverse_labels:
            if label not in multiverse:
                raise ValueError(f"Multiverse must have {label} key.")

    def _validate_directory(self, path):
        """
        Ensure the directory exists.

        Parameters
        ----------
        path : str
            Path to the directory.

        Returns
        -------
        str
            Validated directory path.

        Raises
        ------
        ValueError
            If the directory does not exist.
        """
        if not os.path.isdir(path):
            raise ValueError(f"Directory not found at {path}")
        return path

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ File Handlers                                            │
    #  ╰──────────────────────────────────────────────────────────╯

    def clean(self):
        """
        Clean the output directory.
        """
        LSD.clean_dir(self.outDir)

    def clean_configs(self):
        """
        Clean the configs directory.
        """
        LSD.clean_dir(self.configs)

    def clean_latent_spaces(self):
        """
        Clean the latent spaces directory.
        """
        LSD.clean_dir(self.latent_spaces)

    def clean_models(self):
        """
        Clean the models directory.
        """
        LSD.clean_dir(self.models)

    def clean_logs(self):
        """
        Clean the logs directory.
        """
        LSD.clean_dir(self.logs)

    @staticmethod
    def clean_dir(path: str):
        """
        Delete a directory.

        Parameters
        ----------
        path : str
            Path to the directory.

        Raises
        ------
        AssertionError
            If the directory does not exist.
        """
        assert os.path.isdir(path), f"Directory not found at {path}"
        shutil.rmtree(path)
