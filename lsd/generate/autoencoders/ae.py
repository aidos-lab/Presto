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
)


class Autoencoder(Base):
    def __init__(self, params: omegaconf.DictConfig):
        super().__init__(params)
        self.trainer_cfg = self.setup()

    def setup(self):
        """
        Set up the autoencoder configuration and create necessary directories.
        """
        trainer_cfg = self.initialize_trainer_config()
        self.configure_trainer(trainer_cfg)
        self.create_output_directories(trainer_cfg)
        self.wandb = test_wandb_connection(get_wandb_env())
        return trainer_cfg

    def generate(self):
        """
        Generate the latent space using the trained autoencoder.
        """
        print(f"Generate latent space for {self.trainer_cfg.generators}.")
        latent_space = self.gym.latent_space()
        self.save_latent_space(latent_space)
        del self.gym, latent_space

    def train(self):
        """
        Train the autoencoder models using the configured settings.
        """
        logger = self.initialize_logger()
        self.gym = Gym(self.trainer_cfg, logger)
        model = self.gym.train()
        self.save_model(model)
        del model

    def initialize_trainer_config(self):
        """
        Initialize and return the base trainer configuration.
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

    def configure_trainer(self, trainer_cfg):
        """
        Configure the trainer settings by updating with the parameter values.
        """
        for sub_dict in self.params.values():
            if isinstance(sub_dict, omegaconf.dictconfig.DictConfig):
                self.update_trainer_config(trainer_cfg, sub_dict)

    def update_trainer_config(self, trainer_cfg, sub_dict):
        """
        Update the trainer configuration with key-value pairs from the given dictionary.
        """
        for key, value in sub_dict.items():
            if key not in ["module", "name"]:
                trainer_cfg[key] = value

    def create_output_directories(self, trainer_cfg):
        """
        Create directories for storing latent spaces and models if they do not already exist.
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

    def _create_directory(self, base_path, sub_path):
        """
        Create a directory given a base path and subdirectory name.
        """
        dir_path = os.path.join(base_path, sub_path)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def initialize_logger(self):
        """
        Initialize and return a logger for the training process.
        """
        return Logger(
            exp=self.trainer_cfg.experiment,
            name=f"universe_{self.trainer_cfg.get('id')}",
            wandb_logging=self.wandb,
            out_file=True,
        )

    def save_model(self, model):
        """
        Save the trained model to a file.
        """
        write_pkl(model, self.modelFile)

    def save_latent_space(self, latent_space):
        """
        Save the generated latent space to a file.
        """
        write_pkl(latent_space, self.outFile)
