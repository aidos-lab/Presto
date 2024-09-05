from lsd import Base
from lsd.utils import extract_yaml_id, ConfigType
import importlib
from datasets import load_dataset
import omegaconf


class Transformer(Base):
    def __init__(self, params: ConfigType) -> None:
        super().__init__(params)

        self.tf_cfg = self.setup()

    def setup(self) -> ConfigType:
        tf_cfg = self._initialize_tf_config()

        self.initialize_model(tf_cfg)
        self.load_data(tf_cfg)

        self._create_latent_directory(tf_cfg)

        return tf_cfg

    def train(self) -> None:
        pass

    def generate(self):
        pass

    def initialize_model(self, tf_cfg: ConfigType) -> None:
        module = tf_cfg.get("model")
        try:
            self.model = importlib.import_module(module).initialize()
        except ImportError as e:
            raise ImportError(
                f"Failed to initialize model from module '{module}': {e}"
            )

    def load_data(self, tf_cfg: ConfigType):

        if tf_cfg.data_host.lower() == "local":
            self._load_local_data(tf_cfg)
        else:
            self._load_remote(tf_cfg)

    @staticmethod
    def _load_remote(tf_cfg: ConfigType):

        dataset_name = tf_cfg.get("dataset", None)
        version = tf_cfg.get("data_version", "3.0.0")
        split = tf_cfg.get("split", "train")

        print(f"Loading {dataset_name} version {version} with split {split}")

        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name, version, split=split)

    @staticmethod
    def _load_local_data(tf_cfg: ConfigType):
        print("Local")
        pass

    def _initialize_tf_config(self) -> ConfigType:
        """
        Initialize and return the base trainer configuration.

        This method creates an empty configuration using `omegaconf` and fills
        it with basic information extracted from the provided parameters.

        Returns
        -------
        ConfigType
            The initialized trainer configuration.
        """
        tf_cfg = omegaconf.OmegaConf.create({})
        tf_cfg.experiment = self.params.get("experiment", "")
        tf_cfg.id = extract_yaml_id(self.params.get("file", ""))
        tf_cfg.model = self.params.get("model_choices", {}).get("module", "")
        tf_cfg.dataset = self.params.get("data_choices", {}).get("name", "")
        tf_cfg.data_version = self.params.get("data_choices", {}).get(
            "version", ""
        )
        tf_cfg.data_split = self.params.get("data_choices", {}).get("split", "")
        tf_cfg.data_host = self.params.get("data_choices", {}).get("host", "")

        # TODO: What implementation parameters are key for generation?
        tf_cfg.implementation = self.params.get(
            "implementation_choices", {}
        ).get("module", "")

        tf_cfg.generators = [
            self.params.get("data_choices", {}).get("name", ""),
            self.params.get("model_choices", {}).get("name", ""),
            self.params.get("implementation_choices", {}).get("name", ""),
        ]

        return tf_cfg

    def _create_latent_directory(self, tf_cfg: ConfigType):
        pass
