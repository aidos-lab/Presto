"Latent Space Designer: Let LSD take you into the multiverse"
import os
import importlib
import omegaconf
import itertools
from dataclasses import fields, dataclass
import shutil
from typing import Union, Tuple, Callable

import lsd.config as config
import lsd.utils as ut
from lsd.utils import ConfigType


class LSD:
    """
    Latent Space Designer: Let LSD take you into the multiverse.

    The LSD class is the main interface for designing and generating latent spaces for various generative models. In particular, LSD lets the user generate a multiverse of embeddings based on different data, modeling, and implementation choices. The class is designed to be extensible and can be used with any configuration class that defines the multiverse components.

    Parameters
    ----------
    multiverseConfig : str
        The multiverse configuration type. This must correspond to a dataclass defined in the `lsd.config`.
    outDir : str
        Output directory to store the generated files.
    experimentName : str, optional
        Name for the experiment, by default None. If not provided, LSD will generate a unique name.
    labels : list, optional
        Labels for different choices in the multiverse, by default ["data_choices", "model_choices", "implementation_choices"]. These labels should match the labels in the YAML configurations.

    Attributes
    ----------
    cfg : ConfigType
        Configuration object for the multiverse.
    multiverse_labels : list
        List of labels for multiverse components.
    design_path : str
        Path where the designed configurations are stored.
    outDir : str
        Full output directory path including experiment name.
    experimentName : str
        Name of the experiment.
    _multiverse : ConfigType
        Internal storage for the multiverse components.

    Properties
    ----------
    multiverse : ConfigType
        The multiverse components loaded from the configuration.
    generators : ConfigType
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
        Read in the Multiverse configuration, create a direct product of possible configurations, and write individual config files for each model in the multiverse. This method ensures that the multiverse configuration is loaded and prepares the design path for storing the generated configurations.

    generate()
        Generate representations for each configuration in the design path.
        This method iterates over each configuration file in the design path and generates the corresponding representations. These representations are saved as pkl files (see `lsd.base.Base` and its children for more details on the file handling).

    clean()
        Clean the output directory by deleting all its contents.

    clean_configs()
        Clean the configs directory by deleting all its contents.

    clean_latent_spaces()
        Clean the latent spaces directory by deleting all its contents.

    clean_models()
        Clean the models directory by deleting all its contents.

    clean_logs()
        Clean the logs directory by deleting all its contents.

    Notes
    -----
    - The `clean_*` methods rely on `LSD.clean_dir` to delete the directories.
    - The `design` method prepares the configurations and writes them to the design path, which is crucial for the `generate` method to function correctly.

    Examples
    --------
    To use the `LSD` class, you would typically do the following:

        >>> lsd = LSD(multiverseConfig='path/to/config.yaml', outDir='/output/path')
        >>> lsd.design()
        >>> lsd.generate()

    After generating the representations, you might want to clean up:

        >>> lsd.clean()
        >>> lsd.clean_configs()
        >>> lsd.clean_latent_spaces()
        >>> lsd.clean_models()
        >>> lsd.clean_logs()
    """

    def __init__(
        self,
        multiverseConfig,
        outDir: str,
        experimentName: str = None,
        labels: list = None,
    ) -> None:
        """
        Initialize the LSD object.

        This constructor sets up the LSD object with the given multiverse configuration, output directory, experiment name, and optional labels. It ensures that the labels and output directory are validated and prepares the configuration for further use.

        Parameters
        ----------
        multiverseConfig : str
            The multiverse configuration type. This must correspond to a dataclass defined in the `lsd.config`.
        outDir : str
            The base output directory where the experiment data will be stored.
        experimentName : str, optional
            The name of the experiment. If not provided, a default name will be generated.
        labels : list, optional
            A list of labels for the components of the multiverse. If not provided,
            defaults to `["data_choices", "model_choices", "implementation_choices"]`.

        Raises
        ------
        ValueError
            If the provided labels are invalid.
        OSError
            If there is an issue with the output directory path.
        """
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
    def multiverse(self) -> ConfigType:
        """
        Getter property for accessing the multiverse configuration.

        Returns:
        -------
        dict
            The multiverse configuration.
        """
        if self._multiverse is None:
            self._load_multiverse()
        return self._multiverse

    @property
    def generators(self) -> ConfigType:
        """
        Getter property for generators. Gets a sorted list of generators for label in the multiverse (defaults are data_choices, model_choices, implementation_choices).

        Returns:
        -------
        dict
            Dictionary where keys are labels and values are sorted lists of generators.
        """
        return {
            label: sorted(list(self.multiverse[label].keys()))
            for label in self.multiverse_labels
        }

    @multiverse.setter
    def multiverse(self, multiverse: ConfigType) -> None:
        """
        Setter method for multiverse property. Use this method to set a new multiverse configuration to an `LSD` object.

        Parameters:
        -------
        multiverse : ConfigType
            The new multiverse configuration to set.

        Raises:
        -------
        ValueError
            If the provided multiverse configuration is invalid.

        Notes:
        -------
        The input multiverse configuration dictionary must have keys that match the labels associated with your LSD class.

        Examples:
        -------
        >>> lsd = LSD("MultiverseConfig", "output/",labels=["A", "B", "C"])
        >>> lsd.multiverse = dict(A={}, B={}, C={})
        >>> lsd.multiverse
        {'A': {}, 'B': {}, 'C': {}}

        """
        self._validate_multiverse(multiverse)
        self._multiverse = {}
        for label in self.multiverse_labels:
            label_config = multiverse.get(label, {})
            if label_config is None:
                label_config = {}
            self.__setattr__(label, label_config)
            self._multiverse[label] = label_config.get(self.cfg.base, {})

    @property
    def experiment(self) -> str:
        """
        Getter property for accessing the experiment output directory.

        Returns:
        -------
        str
            The path to the experiment output directory.
        """
        return self.outDir

    @property
    def configs(self) -> str:
        """
        Getter property for accessing the directory containing configuration files.

        Returns:
        -------
        str
            The path to the configuration files directory.
        """
        return self._validate_directory(os.path.join(self.outDir, "configs/"))

    @property
    def latent_spaces(self) -> str:
        """
        Getter property for accessing the directory containing latent spaces.

        Returns:
        -------
        str
            The path to the latent spaces directory.
        """
        return self._validate_directory(
            os.path.join(self.outDir, "latent_spaces/")
        )

    @property
    def models(self) -> str:
        """
        Getter property for accessing the directory containing model files.

        Returns:
        -------
        str
            The path to the models directory.
        """
        return self._validate_directory(os.path.join(self.outDir, "models/"))

    @property
    def logs(self) -> str:
        """
        Getter property for accessing the directory containing log files.

        Returns:
        -------
        str
            The path to the logs directory.
        """
        return self._validate_directory(os.path.join(self.outDir, "logs/"))

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Key Members: Design and Generate                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def design(self) -> None:
        """
        Read in Multiverse configuration, create a direct product, and write individual config files for each model in the multiverse.

        Raises:
        -------
        ValueError
            If the Multiverse configuration is invalid or incomplete.
        """
        if self._multiverse is None:
            self._load_multiverse()

        self._prepare_design_path()

        self.module = importlib.import_module(self.cfg.module)
        universes = self._generate_universe_configs()

        self._save_universe_configs(universes)

    def generate(self) -> None:
        """
        Generate representations for each configuration in the design path.

        If `design_path` is None, it invokes the `design` method to ensure configurations are prepared.

        Raises:
        -------
        FileNotFoundError
            If the design path does not exist or cannot be accessed.
        """
        if self.design_path is None:
            self.design()

        base = getattr(self.module, self.cfg.base)

        for cfg in sorted(os.listdir(self.design_path), key=ut.file_id_sorter):
            if cfg.startswith("universe_") and cfg.endswith(".yml"):
                cfg_path = os.path.join(self.design_path, cfg)
                self._generate(self.outDir, cfg_path, base)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helpers                                                  │
    #  ╰──────────────────────────────────────────────────────────╯

    @staticmethod
    def _generate(outDir, cfg_path, base) -> None:
        """
        Generate latent spaces based on a particular config file.

        This function relies on the `train` and `generate` methods implemented in the generator class to produce and save the latent spaces.

        Parameters
        ----------
        outDir : str
            Output directory for generated files.
        cfg_path : str
            Path to the configuration file.
        base : Callable
            A generator class (child of `lsd.base.Base`).

        Raises
        ------
        NotImplementedError
            If the generator class does not have `train` and `generate` methods implemented.

        Notes:
        -------
        This method is designed to work with children of `lsd.base.Base`, which requires  `train` and `generate` methods to be implemented. The method instantiates the generator class with the configuration file and calls the `train` and `generate` methods to generate the latent spaces.
        """
        theta = omegaconf.OmegaConf.load(cfg_path)
        theta.experiment = outDir
        theta.file = cfg_path
        M = base(theta)
        M.train()
        M.generate()
        del M

    def _prepare_design_path(self) -> None:
        """
        Prepare the directory for storing design configurations.

        This method sets up the output directory for storing configuration files
        generated for the multiverse. If the directory already exists, it will be cleaned to remove any existing configurations.

        Raises
        ------
        OSError
            If there is an issue with directory creation or cleaning.
        """
        configs_outDir = os.path.join(self.outDir, "configs/")
        if os.path.isdir(configs_outDir):
            self.clean()
        self.design_path = configs_outDir

    def _generate_universe_configs(self) -> list:
        """Generate configuration combinations for the multiverse.

        This method creates a list of configurations for each label in the
        multiverse by combining all possible configurations generated by
        the specified generators.

        Returns
        -------
        universes : list
            A list of configuration combinations for the multiverse.
        """
        universes = []
        for label, generators in self.generators.items():
            configs = self._generate_configurations(generators, label)
            universes.append(configs)
        return universes

    def _generate_configurations(self, generators, label) -> list:
        """
        Generate configurations for each generator in a label.

        This method generates a list of configurations for the specified label.
        Each generator's parameters from the `LSD`'s multiverse are matched with the parameters in the generator's configuration `dataclass` to create unique configurations that encode the generators parameter space or "grid search".

        Parameters
        ----------
        generators : list
            List of generator names for the specified label.
        label : str
            The label for which configurations are to be generated.

        Returns
        -------
        configs : list
            A list of configurations for the specified label.
        """
        configs = []
        for G in generators:
            gen_cfg = getattr(self.module, G)
            params, vectors = self._design_parameter_space(gen_cfg, label, G)
            configs.extend(
                self._generate_individual_configs(params, vectors, gen_cfg)
            )
        return configs

    @staticmethod
    def _generate_individual_configs(params, vectors, gen_cfg) -> list:
        """
        Generate a individual configurations for each vector in a parameter grid search.

        This method creates a list of configurations by instantiating a generator configuration based on the provided parameter values in each vector.

        Parameters
        ----------
        params : list
            List of parameter names to be used in the configuration.
        vectors : list
            List of vectors representing different coordinates in the parameter space.
        gen_cfg : object
            The generator configuration object used to instantiate configurations.

        Returns
        -------
        list
            A list of single configurations based on the provided parameters
            and vectors.
        """
        return [
            ut.LoadClass.instantiate(gen_cfg, dict(zip(params, theta)))
            for theta in vectors
        ]

    def _save_universe_configs(self, universes) -> None:
        """
        Save the generated universe configurations.

        This method saves each generated universe configuration to a file in the
        design path directory, incrementing the `multiverse_size` attribute for
        each saved configuration.

        Parameters
        ----------
        universes : list
            A list of universe configuration combinations to be saved.

        """
        self.multiverse_size = 0
        for i, U in enumerate(itertools.product(*universes)):
            universe = config.Universe(*U)
            outPath = os.path.join(self.design_path, f"universe_{i}.yml")
            self.write_cfg(outPath, universe)
            self.multiverse_size += 1

    def _load_multiverse(self) -> None:
        """
        Load and parse the LSD cfg object to produce a dictionary of parameters encoding the different choices associated with a multiverse.

        This method loads the parameters for each label in the multiverse configuration and organizes it into a common structure.

        Raises
        ------
        AttributeError
            If the specified label or base configuration does not exist in the
            multiverse configuration object.
        """
        self._multiverse = {}
        for label in self.multiverse_labels:
            params = self.filter_params(
                getattr(self.cfg, label), label, self.cfg.base
            )
            self.__setattr__(label, params)
            self._multiverse[label] = params

    def _design_parameter_space(
        self,
        cfg,
        label,
        generator,
    ) -> Tuple[list, itertools.product]:
        """
        Design the parameter space for a given configuration. This method combines the user-defined parameters with the configuration dataclass to generate a grid search of unique configurations.

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
        params : list
            Parameters and vectors for the configuration.
        vectors : itertools.product
        """
        user_params = self.multiverse[label][generator].keys()
        cfg_params = [f.name for f in fields(cfg)]
        params = self._intersection(cfg_params, user_params)
        vectors = self._cartesian_product(label, generator, params)
        return params, vectors

    def _cartesian_product(
        self,
        label,
        generator,
        parameters,
    ) -> itertools.product:
        """
        Generate the cartesian product of parameters for a given generator based on user inputs and available parameters in the generator's dataclass configuration.

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
        return itertools.product(
            *[
                self.multiverse[label][generator][param]
                for param in parameters
                if generator in self.multiverse[label]
                and param in self.multiverse[label][generator].keys()
            ]
        )

    @staticmethod
    def _intersection(x, y) -> list:
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
    def read_params(path: str) -> ConfigType:
        """
        Read parameters from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        ConfigType
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
    def write_cfg(config_file_path: str, cfg) -> None:
        """
        Write configuration to a YAML file.

        Parameters
        ----------
        config_file_path : str
            Path to the output YAML file.
        cfg : ConfigType
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
    def filter_params(
        path: str,
        choices: str,
        base: str,
    ) -> ConfigType:
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

    @staticmethod
    def _get_config(config_name) -> dataclass:
        """
        Retrieve the configuration object from the config module.

        Parameters
        ----------
        config_name : str
            Name of the configuration class.

        Returns
        dataclass
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

    def _validate_labels(self) -> None:
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

    def _validate_outDir(self, outDir) -> None:
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

    def _generate_experiment_name(self) -> str:
        """
        Generate a unique experiment name.

        Returns
        -------
        str
            Unique experiment name.
        """
        return self.cfg.base + ut.temporal_id()

    def _validate_multiverse(self, multiverse) -> None:
        """
        Validate the multiverse structure.

        Parameters
        ----------
        multiverse : ConfigType
            Multiverse configuration.

        Raises
        ------
        TypeError
            If the multiverse is not a dictionary or DictConfig.
        ValueError
            If the multiverse does not contain required labels.
        """
        if not isinstance(multiverse, (dict, omegaconf.DictConfig)):
            raise TypeError(
                "Multiverse must be a dictionary or omegaconf.DictConfig."
            )
        for label in self.multiverse_labels:
            if label not in multiverse:
                raise ValueError(f"Multiverse must have {label} key.")

    def _validate_directory(self, path: str) -> str:
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

    def clean(self) -> None:
        """
        Clean the output directory.

        This method deletes all contents of the output directory specified by `self.outDir`.

        Raises
        ------
        AssertionError
            If the output directory does not exist.
        IOError
            If there is an error deleting the contents of the output directory.
        """
        LSD.clean_dir(self.outDir)

    def clean_configs(self) -> None:
        """
        Clean the configs directory.

        This method deletes all contents of the configuration files directory.

        Raises
        ------
        AssertionError
            If the configs directory does not exist.
        IOError
            If there is an error deleting the contents of the configs directory.
        """
        LSD.clean_dir(self.configs)

    def clean_latent_spaces(self) -> None:
        """
        Clean the latent spaces directory.

        This method deletes all contents of the latent spaces directory.

        Raises
        ------
        AssertionError
            If the latent spaces directory does not exist.
        IOError
            If there is an error deleting the contents of the latent spaces directory.
        """
        LSD.clean_dir(self.latent_spaces)

    def clean_models(self) -> None:
        """
        Clean the models directory.

        This method deletes all contents of the models directory.

        Raises
        ------
        AssertionError
            If the models directory does not exist.
        IOError
            If there is an error deleting the contents of the models directory.
        """
        LSD.clean_dir(self.models)

    def clean_logs(self) -> None:
        """
        Clean the logs directory.

        This method deletes all contents of the logs directory.

        Raises
        ------
        AssertionError
            If the logs directory does not exist.
        IOError
            If there is an error deleting the contents of the logs directory.
        """
        LSD.clean_dir(self.logs)

    @staticmethod
    def clean_dir(path: str) -> None:
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
