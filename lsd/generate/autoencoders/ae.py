import omegaconf
import os
from lsd import Base
from lsd.generate.autoencoders.gym import Gym
from lsd.generate.autoencoders.logger import Logger
from lsd.utils import (
    extract_yaml_id,
    get_wandb_env,
    test_wandb_connection,
    write_pkl,
    ConfigType,
)


class Autoencoder(Base):
    """
    Autoencoder Latent Space Generator.

    This class provides functionality to set up, train, and generate latent
    spaces using variational autoencoders (VAEs). It extends the `Base` class
    and implements specific methods for managing autoencoder workflows.

    Parameters
    ----------
    params : ConfigType
        Configuration parameters for setting up the autoencoder. This includes
        details about data loading, model configuration, and training settings.

    Attributes
    ----------
    params : ConfigType
        Configuration parameters provided during instantiation.
    trainer_cfg : ConfigType
        Configuration for the trainer initialized during setup.
    gym : Gym
        Gym instance used for training the autoencoder.
    wandb : bool
        Flag indicating whether Weights & Biases (wandb) integration is enabled.
    modelsDir : str
        Directory path for storing trained models.
    latentsDir : str
        Directory path for storing generated latent spaces.
    modelFile : str
        Output file path for the saved model.
    outFile : str
        Output file path for the saved latent space.

    Methods
    -------
    setup()
        Set up the autoencoder configuration and create necessary directories.
    initialize_trainer_config()
        Initialize and return the base trainer configuration.
    configure_trainer(trainer_cfg)
        Configure the trainer settings by updating with the parameter values.
    initialize_logger()
        Initialize and return a logger for the training process.
    train()
        Train the autoencoder models using the configured settings.
    generate()
        Generate the latent space using the trained autoencoder.

    Helper Functions
    ----------------
    _update_trainer_config(trainer_cfg, sub_dict)
        Update the trainer configuration with key-value pairs from the given
        dictionary.
    _save_model(model)
        Save the trained model to a file.
    _save_latent_space(latent_space)
        Save the generated latent space to a file.
    _create_directory(base_path, sub_path)
        Create a directory given a base path and subdirectory name.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> params = OmegaConf.create({
    ...     'experiment': 'my_experiment',
    ...     'file': 'config.yml',
    ...     'data_choices': {'module': 'data_module', 'name': 'my_data'},
    ...     'model_choices': {'module': 'model_module', 'name': 'my_model'},
    ...     'implementation_choices': {'module': 'optim_module', 'name': 'my_optimizer'}
    ... })
    >>> autoencoder = Autoencoder(params)
    >>> autoencoder.train()
    >>> autoencoder.generate()
    """

    def __init__(self, params: ConfigType) -> None:
        """
        Constructor for the Autoencoder instance.

        Parameters
        ----------
        params : ConfigType
            Configuration parameters for setting up the autoencoder.
        """
        super().__init__(params)
        self.trainer_cfg = self.setup()

    def setup(self) -> ConfigType:
        """
        Set up the autoencoder generator based on the provided parameter configuration.

        This method initializes the trainer configuration, configures the
        trainer settings, creates output directories for latent spaces and
        models, and checks Weights & Biases (wandb) connection.

        Returns
        -------
        ConfigType
            The initialized trainer configuration.
        """
        trainer_cfg = self.initialize_trainer_config()
        self.configure_trainer(trainer_cfg)
        self.create_output_directories(trainer_cfg)
        self.wandb = test_wandb_connection(get_wandb_env())
        return trainer_cfg

    def initialize_trainer_config(self) -> ConfigType:
        """
        Initialize and return the base trainer configuration.

        This method creates an empty configuration using `omegaconf` and fills
        it with basic information extracted from the provided parameters.

        Returns
        -------
        ConfigType
            The initialized trainer configuration.
        """
        trainer_cfg = omegaconf.OmegaConf.create({})
        trainer_cfg.experiment = self.params.get("experiment", "")
        trainer_cfg.id = extract_yaml_id(self.params.get("file", ""))
        trainer_cfg.model = self.params.get("model_choices", {}).get(
            "module", ""
        )
        trainer_cfg.dataset = self.params.get("data_choices", {}).get(
            "module", ""
        )
        trainer_cfg.optimizer = self.params.get(
            "implementation_choices", {}
        ).get("module", "")
        trainer_cfg.generators = [
            self.params.get("data_choices", {}).get("name", ""),
            self.params.get("model_choices", {}).get("name", ""),
            self.params.get("implementation_choices", {}).get("name", ""),
        ]

        return trainer_cfg

    def configure_trainer(self, trainer_cfg) -> None:
        """
        Configure the trainer config objectg with relevant parameters .

        This method iterates over the parameter values and updates the trainer
        configuration. See `_update_trainer_config` for more details.

        Parameters
        ----------
        trainer_cfg : ConfigType
            The trainer configuration to be updated.
        """
        for sub_dict in self.params.values():
            if isinstance(sub_dict, (dict, omegaconf.DictConfig)):
                self._update_trainer_config(trainer_cfg, sub_dict)

    def initialize_logger(self) -> Logger:
        """
        Initialize and return a logger for the training process.

        This method creates a logger instance to log training progress and
        results, and optionally integrates with Weights & Biases (wandb).

        Returns
        -------
        Logger
            The initialized logger for the training process.
        """
        return Logger(
            exp=self.trainer_cfg.experiment,
            name=f"universe_{self.trainer_cfg.get('id')}",
            wandb_logging=self.wandb,
            out_file=True,
        )

    def create_output_directories(self, trainer_cfg) -> None:
        """
        Create directories for storing latent spaces and models if they do not already exist.

        Parameters
        ----------
        trainer_cfg : ConfigType
            The trainer configuration containing the experiment path.
        """
        self.latentsDir = self._create_directory(
            trainer_cfg.experiment, "latent_spaces"
        )

        self.modelsDir = self._create_directory(
            trainer_cfg.experiment, "models"
        )

        self.outFile = os.path.join(
            self.latentsDir, f"universe_{trainer_cfg.id}.pkl"
        )
        self.modelFile = os.path.join(
            self.modelsDir, f"model_{trainer_cfg.id}.pkl"
        )

    def train(self) -> None:
        """
        Train the autoencoder models using the configured settings.

        This method initializes a logger, sets up a Gym instance for training,
        and saves the trained model to a specified file.
        """
        logger = self.initialize_logger()
        self.gym = Gym(self.trainer_cfg, logger)
        model = self.gym.train()
        self._save_model(model)
        del model

    def generate(self) -> None:
        """
        Generate the latent space using the trained autoencoder.

        This method uses the trained model to generate a latent space
        representation of the data and saves it to a specified output file.
        """
        latent_space = self.gym.latent_space()
        self._save_latent_space(latent_space)
        del self.gym, latent_space

    def _update_trainer_config(self, trainer_cfg, sub_dict) -> None:
        """
        Update the trainer configuration with key-value pairs from the given dictionary.

        Parameters
        ----------
        trainer_cfg : ConfigType
            The trainer configuration to be updated.
        sub_dict : dict
            Dictionary of configuration parameters to update the trainer with.

        Notes
        -----
        This method ignores the `module` and `name` keys in the sub-dictionary to avoid conflicts between data, model, and optimizer configurations.
        """
        for key, value in sub_dict.items():
            if key not in ["module", "name"]:
                trainer_cfg[key] = value

    def _save_model(self, model) -> None:
        """
        Save the trained model to a file.

        This method serializes the trained model and saves it to the specified
        output file path.

        Parameters
        ----------
        model : object
            The trained model to be saved.
        """
        write_pkl(model, self.modelFile)

    def _save_latent_space(self, latent_space) -> None:
        """
        Save the generated latent space to a file.

        This method serializes the generated latent space and saves it to the
        specified output file path.

        Parameters
        ----------
        latent_space : object
            The generated latent space to be saved.
        """
        write_pkl(latent_space, self.outFile)

    @staticmethod
    def _create_directory(base_path, sub_path) -> str:
        """
        Create a directory given a base path and subdirectory name.

        This method checks if the directory exists and creates it if it does not.

        Parameters
        ----------
        base_path : str
            The base path where the directory should be created.
        sub_path : str
            The name of the subdirectory to create.

        Returns
        -------
        str
            The full path of the created directory.
        """
        dir_path = os.path.join(base_path, sub_path)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
