import importlib
import os
from typing import Optional, Union

import numpy as np
import omegaconf
from sklearn import datasets as sc
from sklearn.decomposition import PCA

from lsd import Base
from lsd.utils import ConfigType, extract_yaml_id, write_pkl


class DimReduction(Base):
    """
    Dimensionality Reduction for Latent Space Generation.

    This class provides functionality for generating low-dimensional embeddings
    using various dimensionality reduction algorithms. It extends the `Base`
    class and implements specific methods for managing dimensionality reduction workflows.

    Attributes
    ----------
    params : ConfigType
        Configuration parameters provided during instantiation.
    projector_cfg : ConfigType
        Configuration for the projector initialized during setup.
    data : array-like
        The dataset used for dimensionality reduction.
    labels : array-like
        The labels associated with the dataset.
    model : object
        The initialized dimensionality reduction model.
    modelsDir : str
        Directory path for storing preprocessed data sets.
    latentsDir : str
        Directory path for storing generated latent spaces.
    modelFile : str
        Output file path for the saved preprocessed data set.
    outFile : str
        Output file path for the saved latent space.

    Methods
    -------
    setup()
        Set up the dimensionality reduction configuration and create necessary
        directories.
    load_data()
        Load data according to the specified data choices.
    configure_projector(projector_cfg)
        Configure the projector settings by updating with the parameter values.
    initialize_model(projector_cfg)
        Initialize the model specified in the projector configuration.
    train()
        If needed, preprocess input data using PCA down to a configurable maximum ambient dimension.
    generate()
        Use the configured projector to generate a (low-dimensional) latent space.

    Helper Functions
    ----------------
    _initialize_projector_config()
        Initialize and return the base projector configuration.
    _update_projector_config(projector_cfg, sub_dict)
        Update the projector configuration with key-value pairs from the given
        dictionary.
    _load_custom_data(data_choices)
        Load custom data using a user-specified module and generator.
    _load_sklearn_data(generator_name)
        Load data using sklearn's dataset generator.
    _create_latent_directory(projector_cfg)
        Create a directory to store latent spaces if it does not already exist.
    _create_model_directory(projector_cfg)
        Create a directory to store preprocessed "model" datasets if it does not already exist.
    _save_preprocessed_data(preprocessed_data)
        Save the "trained" PCA projection as the model from which we derive the lower dimensional projections.
    _save_latent_space(latent_space)
        Save the generated latent space to a file.
    _should_apply_pca(max_dim)
        Determine if PCA should be applied based on the maximum ambient dimension.
    _apply_pca(data, max_dim)
        Apply PCA to the data to reduce its dimensionality.
    _parse_max_ambient_dimension()
        Helper function to parse and return `max_ambient_dim` as an integer or `NoneType`.

    Notes
    -----
    In the case that no PCA preprocessing is required, the saved model is set to `None`.

    Example
    -------
    >>> from omegaconf import OmegaConf
    >>> params = {
    ...     'experiment': 'dim_reduction_experiment',
    ...     'file': 'config.yml',
    ...     'data_choices': {'module': 'data_module', 'generator': 'load_custom_data'},
    ...     'model_choices': {'module': 'model_module', 'name': 'UMAP'}
    ... }
    >>> dim_reduction = DimReduction(params)
    >>> dim_reduction.load_data()
    >>> dim_reduction.generate()
    """

    def __init__(self, params: ConfigType):
        """
        Constructor for the DimReduction class.

        This method initializes the class with the provided parameters and
        sets up the dimensionality reduction configuration.

        Parameters
        ----------
        params : ConfigType
            Configuration parameters for setting up the dimensionality reduction.
        """
        super().__init__(params)
        self.projector_cfg = self.setup()

    def setup(self) -> ConfigType:
        """
        Set up the dimensionality reduction configuration and create necessary directories.

        This method initializes the projector configuration, configures the
        projector settings, creates output directories for latent spaces,models datasets, and prepares the data for processing.

        Returns
        -------
        projector_cfg : ConfigType
            The initialized projector configuration.
        """
        projector_cfg = self._initialize_projector_config()
        self.load_data()
        self.configure_projector(projector_cfg)
        self.initialize_model(projector_cfg)
        # Create directories for saving latent spaces and models
        self._create_latent_directory(projector_cfg)
        self._create_model_directory(projector_cfg)
        return projector_cfg

    def load_data(self) -> None:
        """
        Load data according to the specified data choices.

        This method determines the data loading strategy based on the `data_choices`
        parameter. It either loads data using a custom module or via sklearn's
        dataset generators.

        Raises
        ------
        ImportError
            If there is an error loading custom data.
        ValueError
            If the specified sklearn data generator does not exist.
        """
        data_choices = self.params.get("data_choices", {})
        generator_name = data_choices.get("generator")

        if generator_name:
            if "module" in data_choices:
                self._load_custom_data(data_choices)
            else:
                self._load_sklearn_data(generator_name)

    def configure_projector(self, projector_cfg) -> None:
        """
        Configure the projector settings by updating with the parameter values.

        This method iterates over the parameter values and updates the projector
        configuration accordingly.

        Parameters
        ----------
        projector_cfg : ConfigType
            The projector configuration to be updated.
        """
        for sub_dict in self.params.values():
            if isinstance(sub_dict, (dict, omegaconf.DictConfig)):
                self._update_projector_config(projector_cfg, sub_dict)

    def initialize_model(self, projector_cfg) -> None:
        """
        Initialize the model specified in the projector configuration.

        This method initializes the dimensionality reduction model based on the
        specified module in the `projector_cfg`.

        Parameters
        ----------
        projector_cfg : ConfigType
            The projector configuration containing the model module.

        Raises
        ------
        ImportError
            If there is an error initializing the model.
        """
        model_module = projector_cfg.get("model")
        try:
            self.model = importlib.import_module(model_module).initialize()
        except ImportError as e:
            raise ImportError(
                f"Failed to initialize model from module '{model_module}': {e}"
            )

    def train(self) -> None:
        """
        Preprocess input data using PCA down to a configurable maximum ambient dimension.

        This method applies Principal Component Analysis (PCA) to raw data that lies in very high dimensions,
        in order to limit latency when using more complex algorithms for dimensionality reduction.
        The number of dimensions to retain after PCA is specified by the `max_ambient_dim` parameter in the `projector_cfg`.
        If the number of features in the data exceeds this value, PCA is applied to reduce the dimensionality.

        Although the `DimReduction` class does not involve traditional training steps, this preprocessing ensures that
        high-dimensional data is effectively reduced, making it feasible to use complex algorithms for generating
        low-dimensional embeddings.

        Notes
        -----
        - The attribute `projector_cfg.used_pca` is set to `True` if PCA is applied.
        - PCA is only performed if the number of features in `data` exceeds `max_ambient_dim`.
        - This method is essential for preprocessing data to make it suitable for projection using more advanced manifold learning techniques,
          which may be computationally intensive without prior dimensionality reduction.
        - Projected data is saved as a "model" to be used for generating latent spaces. If PCA is not applied, the model is set to `None`.

        Returns
        -------
        None
        """
        max_dim = self._parse_max_ambient_dimension(
            self.projector_cfg.max_ambient_dim
        )

        if self._should_apply_pca(max_dim):
            self.data = self._apply_pca(self.data, max_dim)
            self.projector_cfg.used_pca = True

        self._save_preprocessed_data(self.data if max_dim else None)

    def generate(self) -> None:
        """
        Use a pre-trained model to generate a latent space.

        This method uses the initialized model to project the data into a
        lower-dimensional space and saves the resulting latent space to a file.
        """
        model = self.model(self.projector_cfg)
        L = model.project(self.data)
        self._save_latent_space(L)
        del model, L

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _initialize_projector_config(self) -> ConfigType:
        """
        Initialize and return the base projector configuration.

        This method creates an empty configuration using `omegaconf` and fills
        it with basic information extracted from the provided parameters.

        Returns
        -------
        ConfigType
            The initialized projector configuration.
        """
        projector_cfg = omegaconf.OmegaConf.create({})
        projector_cfg.experiment = self.params.get("experiment", "")
        projector_cfg.id = extract_yaml_id(self.params.get("file", ""))
        projector_cfg.model = self.params.get("model_choices", {}).get(
            "module", ""
        )
        projector_cfg.used_pca = False
        return projector_cfg

    def _update_projector_config(self, projector_cfg, sub_dict) -> None:
        """
        Update the projector configuration with key-value pairs from the given dictionary.

        Parameters
        ----------
        projector_cfg : ConfigType
            The projector configuration to be updated.
        sub_dict : dict
            Dictionary of configuration parameters to update the projector with.
        """
        for key, value in sub_dict.items():
            if key not in ["module", "name"]:
                projector_cfg[key] = value

    def _load_custom_data(self, data_choices) -> None:
        """
        Load custom data using a user-specified module and generator.

        This method loads data using a custom module and generator function
        specified in the `data_choices` parameter.

        Parameters
        ----------
        data_choices : dict
            Dictionary containing module and generator function details for
            loading custom data.

        Raises
        ------
        ImportError
            If there is an error loading the custom data.
        """
        try:
            data_module = importlib.import_module(data_choices["module"])
            data_loader = getattr(data_module, data_choices["generator"])
            self.data, self.labels = data_loader(**data_choices)
        except Exception as e:
            raise ImportError(f"Failed to load custom data: {e}")

    def _load_sklearn_data(self, generator_name) -> None:
        """
        Load data using sklearn's dataset generator.

        This method loads a dataset from sklearn's dataset generators based on
        the specified `generator_name`.

        Parameters
        ----------
        generator_name : str
            The name of the sklearn dataset generator to use.

        Raises
        ------
        ValueError
            If the specified sklearn data generator does not exist.
        """
        try:
            loader = getattr(sc, generator_name)
            self.data, self.labels = loader(return_X_y=True)
        except AttributeError:
            raise ValueError(
                f"Sklearn does not have a data generator named '{generator_name}'"
            )

    def _create_latent_directory(self, projector_cfg) -> None:
        """
        Create a directory to store latent spaces if it does not already exist.

        This method creates a directory for storing generated latent spaces
        based on the experiment path in the `projector_cfg`.

        Parameters
        ----------
        projector_cfg : ConfigType
            The projector configuration containing the experiment path.
        """
        self.latentsDir = os.path.join(
            projector_cfg.experiment, "latent_spaces/"
        )
        os.makedirs(self.latentsDir, exist_ok=True)
        self.outFile = os.path.join(
            self.latentsDir, f"universe_{projector_cfg.id}.pkl"
        )
        # Create directories for saving latent spaces and models

    def _create_model_directory(self, projector_cfg) -> None:
        """
        Create a directory to store models (PCA projections) if it does not already exist.

        This method creates a directory for storing generated latent spaces
        based on the experiment path in the `projector_cfg`.

        Parameters
        ----------
        projector_cfg : ConfigType
            The projector configuration containing the experiment path.
        """
        self.modelsDir = os.path.join(projector_cfg.experiment, "models/")
        os.makedirs(self.modelsDir, exist_ok=True)
        self.modelFile = os.path.join(
            self.modelsDir, f"universe_{projector_cfg.id}.pkl"
        )

    def _save_preprocessed_data(self, preprocessed_data) -> None:
        """
        Save the "trained" PCA projection as the model from which we derive the lower dimensional projections.

        This method serializes the trained model and saves it to the specified
        output file path.

        Parameters
        ----------
        preprocessed_data : object
            The trained dataset to be saved as a "model".
        """
        write_pkl(preprocessed_data, self.modelFile)

    def _save_latent_space(self, latent_space) -> None:
        """
        Save the generated latent space to a file.

        This method saves the low-dimensional latent space to a specified file
        in a pickle format. The file path is constructed based on the current
        experiment's configuration.

        Parameters
        ----------
        latent_space : array-like
            The latent space generated by the dimensionality reduction model.

        Raises
        ------
        IOError
            If there is an issue saving the latent space to the file.
        """
        write_pkl(latent_space, self.outFile)

    def _should_apply_pca(self, max_dim: Optional[int]) -> bool:
        """
        Determine if PCA should be applied based on the maximum ambient dimension.

        Parameters
        ----------
        max_dim : Optional[int]
            The maximum ambient dimension.

        Returns
        -------
        bool
            True if PCA should be applied, False otherwise.
        """
        return max_dim is not None and self.data.shape[1] > max_dim

    def _apply_pca(self, data, max_dim: int) -> np.ndarray:
        """
        Apply PCA to the data to reduce its dimensionality.

        Parameters
        ----------
        data : np.ndarray
            The input data to be reduced.
        max_dim : int
            The maximum number of dimensions to retain.

        Returns
        -------
        np.ndarray
            The data with reduced dimensionality.
        """
        pca = PCA(n_components=max_dim, random_state=self.projector_cfg.seed)
        return pca.fit_transform(data)

    @staticmethod
    def _parse_max_ambient_dimension(max_ambient_dim) -> Union[int, None]:
        """
        Parse and convert the maximum ambient dimension.

        This helper function retrieves the `max_ambient_dim` value from the
        `projector_cfg`. If `max_ambient_dim` is a string and equals "None"
        (case-insensitive), it converts it to `None`. Otherwise, it returns
        the value as is, typically an integer.

        Returns
        -------
        max_Dim : int or None
            The maximum number of dimensions to retain after PCA. If the
            `max_ambient_dim` is "None" as a string, it returns `None`.
            Otherwise, it returns the integer value.

        Raises
        ------
        ValueError
            If the `max_ambient_dim` is neither a valid integer nor "None"
            as a string.

        Examples
        --------
        >>> dr = DimReduction(params)
        >>> dr.projector_cfg.max_ambient_dim = "None"
        >>> dr._parse_max_ambient_dimension()
        None

        >>> dr.projector_cfg.max_ambient_dim = 10
        >>> dr._parse_max_ambient_dimension()
        10
        """
        if isinstance(max_ambient_dim, str):
            if max_ambient_dim.lower() == "none":
                return None
            else:
                raise ValueError(
                    f"Invalid max_ambient_dim value: {max_ambient_dim}. Must be an integer or 'None'."
                )

        return max_ambient_dim
