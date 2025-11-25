from lsd import Base
from lsd.utils import extract_yaml_id, write_pkl, ConfigType
import importlib
from datasets import load_dataset
import omegaconf
import os


class Transformer(Base):
    def __init__(self, params: ConfigType) -> None:
        super().__init__(params)

        self.tf_cfg = self.setup()

    def setup(self) -> ConfigType:
        tf_cfg = self._initialize_tf_config()

        self.configure_transformer(tf_cfg)
        self.load_data(tf_cfg)
        self.initialize_model(tf_cfg)

        self._create_latent_directory(tf_cfg)

        return tf_cfg

    def train(self) -> None:
        pass

    def generate(self):
        """
        Generate latent representations using the loaded transformer model.

        This method processes the loaded dataset through the transformer model
        to generate latent embeddings that are saved for further analysis.
        """
        if not hasattr(self, "dataset") or self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        if not hasattr(self, "model") or self.model is None:
            raise ValueError(
                "Model not initialized. Call initialize_model() first."
            )

        # Extract text from dataset based on dataset type
        texts = self._extract_texts_from_dataset()

        print(f"Generating embeddings for {len(texts)} text samples...")

        # Generate embeddings using the model
        model_instance = self.model(self.tf_cfg)
        embeddings = model_instance.embed(texts)

        # Convert to numpy array if it's a tensor
        if hasattr(embeddings, "numpy"):
            embeddings = embeddings.numpy()
        elif hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().numpy()

        self._save_latent_space(embeddings)
        return embeddings

    def _extract_texts_from_dataset(self):
        """
        Extract text content from the loaded dataset.

        Returns
        -------
        List[str]
            List of text strings extracted from the dataset.
        """
        texts = []
        try:
            for item in self.dataset:
                if isinstance(item, dict):
                    if "article" in item and isinstance(item["article"], str):
                        texts.append(item["article"])
                    elif "text" in item and isinstance(item["text"], str):
                        texts.append(item["text"])
                    elif "highlights" in item and isinstance(
                        item["highlights"], str
                    ):
                        texts.append(item["highlights"])
                    elif "sentence" in item and isinstance(
                        item["sentence"], str
                    ):
                        texts.append(item["sentence"])
                    else:
                        for value in item.values():
                            if isinstance(value, str) and len(value) > 10:
                                texts.append(value)
                                break
                elif isinstance(item, str):
                    texts.append(item)
        except TypeError:
            # If not iterable, perhaps it's something else
            pass

        if not texts:
            raise ValueError("No text content found in dataset")

        return texts

    def initialize_model(self, tf_cfg: ConfigType) -> None:
        module = tf_cfg.get("model")
        if not module:
            raise ValueError("Model module not specified in configuration")
        try:
            self.model = importlib.import_module(module).initialize()
        except ImportError as e:
            raise ImportError(
                f"Failed to initialize model from module '{module}': {e}"
            )

    def load_data(self, tf_cfg: ConfigType):
        """
        Load dataset based on configuration.
        """
        if tf_cfg.data_host.lower() == "local":
            self.dataset = self._load_local_data(tf_cfg)
        else:
            self.dataset = self._load_remote(tf_cfg)

    def configure_transformer(self, tf_cfg) -> None:
        """
        Configure the transformer settings by updating with the parameter values.

        This method iterates over the parameter values and updates the transformer
        configuration accordingly.

        Parameters
        ----------
        tf_cfg : ConfigType
            The transformer configuration to be updated.
        """
        for sub_dict in self.params.values():
            if isinstance(sub_dict, (dict, omegaconf.DictConfig)):
                self._update_tf_config(tf_cfg, sub_dict)

    def _load_remote(self, tf_cfg: ConfigType):
        """
        Load dataset from HuggingFace Hub.
        """
        dataset_name = tf_cfg.get("dataset", None)
        version = tf_cfg.get("data_version", "3.0.0")
        split = tf_cfg.get("data_split", "train")
        num_samples = tf_cfg.get("num_samples", None)

        print(f"Loading {dataset_name} version {version} with split {split}")

        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name, revision=version, split=split)

        # Limit samples if specified
        if num_samples is not None:
            if hasattr(dataset, "select"):
                try:
                    dataset = dataset.select(
                        range(min(num_samples, len(dataset)))
                    )
                except TypeError:
                    # If len fails (e.g., Mock), just select num_samples
                    dataset = dataset.select(range(num_samples))
            # If no select, assume already limited or handle elsewhere

        return dataset

    def _load_local_data(self, tf_cfg: ConfigType):
        """
        Load dataset from local files.
        """
        raise NotImplementedError(
            "Local data loading for Transformers is not yet implemented."
        )

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
        tf_cfg.num_samples = self.params.get("data_choices", {}).get(
            "num_samples", None
        )

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

    def _update_tf_config(self, tf_cfg, sub_dict) -> None:
        """
        Update the transformer configuration with key-value pairs from the given dictionary.

        Parameters
        ----------
        tf_cfg : ConfigType
            The transformer configuration to be updated.
        sub_dict : dict
            Dictionary of configuration parameters to update the transformer with.

        Notes
        -----
        This method ignores the `module` and `name` keys in the sub-dictionary to avoid conflicts between data, model, and optimizer configurations.
        """
        for key, value in sub_dict.items():
            if key not in ["module", "name"]:
                tf_cfg[key] = value

    def _create_latent_directory(self, tf_cfg: ConfigType) -> None:
        """
        Create a directory to store latent spaces if it does not already exist.

        Parameters
        ----------
        tf_cfg : ConfigType
            The transformer configuration containing the experiment path.
        """
        self.latentsDir = self._create_directory(
            tf_cfg.experiment, "latent_spaces"
        )
        self.outFile = os.path.join(
            self.latentsDir, f"universe_{tf_cfg.id}.pkl"
        )

    @staticmethod
    def _create_directory(base_path, sub_path) -> str:
        dir_path = os.path.join(base_path, sub_path)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

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
