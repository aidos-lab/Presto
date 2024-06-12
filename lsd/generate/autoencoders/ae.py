import importlib
from lsd.generate.autoencoders.gym import Gym
import omegaconf

from lsd import Base


class Autoencoder(Base):
    def __init__(self, params: omegaconf.DictConfig):
        super().__init__(params)
        self.trainer_cfg = self.setup()

    def setup(self):
        trainer_cfg = omegaconf.OmegaConf.create({})

        # Extract and format module and name fields
        trainer_cfg.model = self.params.model_choices.get("module", "")
        trainer_cfg.dataset = self.params.data_choices.get("module", "")
        trainer_cfg.optimizer = self.params.implementation_choices.get(
            "module", ""
        )

        trainer_cfg.generators = [
            self.params.data_choices.get("name", ""),
            self.params.model_choices.get("name", ""),
            self.params.implementation_choices.get("name", ""),
        ]

        # Unpack all keys from nested dictionaries
        for _, sub_dict in self.params.items():
            for key, value in sub_dict.items():
                if key not in [
                    "module",
                    "name",
                ]:  # Exclude already processed keys
                    trainer_cfg[key] = value

        return trainer_cfg

    def train(self):
        """Train a list of autoencoder configurations."""
        print(self.trainer_cfg)
        exp = Gym(self.trainer_cfg)
        # exp.run()
        # self.model = exp.save_model()
        # for each config file

    def generate(self, model):

        "Use a pretrained model to generate a latent space."
        pass
