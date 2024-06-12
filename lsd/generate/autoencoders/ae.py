import omegaconf

from lsd import Base
from lsd.generate.autoencoders.gym import Gym
from lsd.utils import extract_yaml_id


class Autoencoder(Base):
    def __init__(self, params: omegaconf.DictConfig):
        super().__init__(params)
        self.trainer_cfg = self.setup()

    def setup(self):
        trainer_cfg = omegaconf.OmegaConf.create({})
        trainer_cfg.experiment = self.params.experiment
        trainer_cfg.id = extract_yaml_id(self.params.file)

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
            if isinstance(sub_dict, omegaconf.dictconfig.DictConfig):
                for key, value in sub_dict.items():
                    if key not in [
                        "module",
                        "name",
                    ]:  # Exclude already processed keys
                        trainer_cfg[key] = value

        return trainer_cfg

    def train(self):
        """Train a list of autoencoder configurations."""
        exp = Gym(self.trainer_cfg)
        exp.run()
        # for each config file

    def generate(self):

        print(f"Generate latent space for {self.trainer_cfg.generators}.")

        "Use a pretrained model to generate a latent space."
        pass
