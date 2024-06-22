import os
import importlib
from sklearn import datasets as sc
import omegaconf

from lsd import Base
from lsd.utils import extract_yaml_id, write_pkl


class DimReduction(Base):
    def __init__(self, params: dict):
        super().__init__(params)
        self.projector_cfg = self.setup()

    def setup(self):
        projector_cfg = self._initialize_projector_config()
        self.load_data()
        self.configure_projector(projector_cfg)
        self.initialize_model(projector_cfg)
        self.create_latent_directory(projector_cfg)
        return projector_cfg

    def load_data(self):
        """
        Load data according to the specified data choices.
        """
        data_choices = self.params.get("data_choices", {})
        generator_name = data_choices.get("generator")

        if generator_name:
            if "module" in data_choices:
                self._load_custom_data(data_choices)
            else:
                self._load_sklearn_data(generator_name)

    def train(self):
        pass

    def generate(self):
        "Use a pretrained model to generate a latent space."
        model = self.model(self.projector_cfg)
        L = model.project(self.data)
        self.save_latent_space(L)
        del model, L

    def _initialize_projector_config(self):
        """
        Initialize and return the base projector configuration.
        """
        projector_cfg = omegaconf.OmegaConf.create({})
        projector_cfg.experiment = self.params.get("experiment", "")
        projector_cfg.id = extract_yaml_id(self.params.get("file", ""))
        projector_cfg.model = self.params.get("model_choices", {}).get(
            "module", ""
        )
        return projector_cfg

    def configure_projector(self, projector_cfg):
        """
        Configure the projector settings by updating with the parameter values.
        """
        for sub_dict in self.params.values():
            if isinstance(sub_dict, omegaconf.dictconfig.DictConfig):
                self._update_projector_config(projector_cfg, sub_dict)

    def _update_projector_config(self, projector_cfg, sub_dict):
        """
        Update the projector configuration with key-value pairs from the given dictionary.
        """
        for key, value in sub_dict.items():
            if key not in ["module", "name"]:
                projector_cfg[key] = value

    def _load_custom_data(self, data_choices):
        """
        Load custom data using a user-specified module and generator.
        """
        try:
            data_module = importlib.import_module(data_choices["module"])
            data_loader = getattr(data_module, data_choices["generator"])
            self.data, self.labels = data_loader(**data_choices)
        except Exception as e:
            raise ImportError(f"Failed to load custom data: {e}")

    def _load_sklearn_data(self, generator_name):
        """
        Load data using sklearn's dataset generator.
        """
        try:
            loader = getattr(sc, generator_name)
            self.data, self.labels = loader(return_X_y=True)
        except AttributeError:
            raise ValueError(
                f"Sklearn does not have a data generator named '{generator_name}'"
            )

    def initialize_model(self, projector_cfg):
        """
        Initialize the model specified in the projector configuration.
        """
        model_module = projector_cfg.get("model")
        try:
            self.model = importlib.import_module(model_module).initialize()
        except ImportError as e:
            raise ImportError(
                f"Failed to initialize model from module '{model_module}': {e}"
            )

    def create_latent_directory(self, projector_cfg):
        """
        Create a directory to store latent spaces if it does not already exist.
        """
        self.latentsDir = os.path.join(
            projector_cfg.experiment, "latent_spaces/"
        )
        os.makedirs(self.latentsDir, exist_ok=True)
        self.outFile = os.path.join(
            self.latentsDir, f"universe_{projector_cfg.id}.pkl"
        )

    def save_latent_space(self, latent_space):
        """
        Save the generated latent space to a file.
        """
        write_pkl(latent_space, self.outFile)
