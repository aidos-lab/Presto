import os

from .gym import Gym

from lsd import Base


class Autoencoder(Base):
    def __init__(self, params: dict):
        super().__init__(params)
        self.cfg_folder = self.setup(params)

    def setup(self):
        "process and unpack the parameters into their proper configs."
        print("Setting up Autoencoder")
        path = "path/to/ae/configs/"
        return path

    def train(self):
        """Train a list of autoencoder configurations."""
        for cfg in os.listdir(self.cfg_folder):  # Python memory issues???
            cfg_file = os.path.join(self.cfg_folder, cfg)
            print(f"Training {cfg_file}")
            exp = Gym(experiment=cfg_file)
            exp.run()
            self.model = exp.save_model()
        # for each config file
        pass

    def generate(self, model):

        "Use a pretrained model to generate a latent space."
        pass

    @staticmethod
    def read_model_params(params: dict):
        """Read in params file and set up model configurations"""
        model_params = params["model_params"]

        return

    @staticmethod
    def read_data_params(params: dict):
        pass

    @staticmethod
    def read_trainer_params(params: dict):
        pass


def initialize():
    return Autoencoder
