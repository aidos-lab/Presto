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
        """
        Generate latent representations using the loaded transformer model.
        
        This method processes the loaded dataset through the transformer model
        to generate latent embeddings that are saved for further analysis.
        """
        if not hasattr(self, 'dataset') or self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        # Extract text from dataset based on dataset type
        texts = self._extract_texts_from_dataset()
        
        print(f"Generating embeddings for {len(texts)} text samples...")
        
        # Generate embeddings using the model
        model_instance = self.model(self.tf_cfg)
        embeddings = model_instance.embed(texts)
        
        # Convert to numpy array if it's a tensor
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        elif hasattr(embeddings, 'detach'):
            embeddings = embeddings.detach().numpy()
        
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
        
        # Handle different dataset structures
        if hasattr(self.dataset, 'to_iterable_dataset'):
            # HuggingFace dataset
            for item in self.dataset:
                if 'article' in item:
                    texts.append(item['article'])
                elif 'text' in item:
                    texts.append(item['text'])
                elif 'sentence' in item:
                    texts.append(item['sentence'])
                else:
                    # Try to find any string field
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 10:  # Reasonable text length
                            texts.append(value)
                            break
        elif isinstance(self.dataset, list):
            # List of text strings
            texts = self.dataset
        elif hasattr(self.dataset, '__iter__'):
            # Any iterable
            texts = list(self.dataset)
        else:
            raise ValueError(f"Unsupported dataset type: {type(self.dataset)}")
            
        if not texts:
            raise ValueError("No text content found in dataset")
            
        return texts

    def initialize_model(self, tf_cfg: ConfigType) -> None:
        module = tf_cfg.get("model")
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
        dataset = load_dataset(dataset_name, version, split=split)
        
        # Limit samples if specified
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        return dataset

    def _load_local_data(self, tf_cfg: ConfigType):
        """
        Load dataset from local files.
        """
        print("Loading local data...")
        # TODO: Implement local data loading
        # For now, return a sample dataset
        return ["Sample text 1", "Sample text 2", "Sample text 3"]

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
