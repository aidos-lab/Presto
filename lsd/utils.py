import hashlib
import os
import pickle
import re
from inspect import signature
from dataclasses import fields
from datetime import datetime
from typing import Union, Dict, Any

import numpy as np
from dotenv import load_dotenv
import wandb
from omegaconf import DictConfig

#  ╭──────────────────────────────────────────────────────────╮
#  │  Custom Type for Configuration objects                   │
#  ╰──────────────────────────────────────────────────────────╯
DictType = Dict[str, Any]
ConfigType = Union[DictType, DictConfig]


#  ╭──────────────────────────────────────────────────────────╮
#  │  Utility Functions & Classes                             │
#  ╰──────────────────────────────────────────────────────────╯


class LoadClass:
    """Unpack an input dictionary to load a class.

    This utility class facilitates the instantiation of classes using an input
    dictionary. It ensures that only valid fields of the class are used for
    instantiation, and caches field information for efficient repeated use.

    Attributes
    ----------
    classFieldCache : dict
        A cache that stores the field names for each class to instantiate.

    Methods
    -------
    instantiate(classToInstantiate, argDict)
        Instantiate a class using an input dictionary.
    """

    classFieldCache = {}

    @classmethod
    def instantiate(cls, classToInstantiate, argDict):
        """
        Instantiate a class using an input dictionary.

        This method filters the input dictionary to only include valid fields
        for the class to be instantiated and creates an instance of that class.

        Parameters
        ----------
        classToInstantiate : type
            The class to be instantiated.
        argDict : dict
            A dictionary of arguments for the class constructor.

        Returns
        -------
        object
            An instance of the specified class.

        Examples
        --------
        >>> class Example:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        >>> instance = LoadClass.instantiate(Example, {'a': 1, 'b': 2, 'c': 3})
        >>> print(instance.a, instance.b)
        1 2
        """
        if classToInstantiate not in cls.classFieldCache:
            cls.classFieldCache[classToInstantiate] = {
                f.name for f in fields(classToInstantiate) if f.init
            }

        fieldSet = cls.classFieldCache[classToInstantiate]
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return classToInstantiate(**filteredArgDict)


def get_parameters(Operator, config) -> dict:
    """
    Extracts the relevant parameters from the configuration for the given operator.

    This utility function  uses introspection to retrieve the parameters of the operator's
    `__init__` method and matches them with the corresponding values in the configuration.

    This is used for various classes in the LSD library to extract parameters from the configuration.

    Parameters
    ----------
    Operator : class
        The projection operator class whose parameters are to be retrieved.
    config : Projector
        The configuration object containing parameters for the projection.

    Returns
    -------
    dict
        A dictionary of parameters for the operator, populated with values from the configuration.
    """
    params = signature(Operator.__init__).parameters
    args = {
        name: config.get(name, param.default)
        for name, param in params.items()
        if param.default != param.empty
    }
    return args


def temporal_id():
    """
    Generate a short, unique temporal identifier.

    This function creates a unique identifier based on the current time,
    suitable for timestamping and differentiating between different events.

    Returns
    -------
    str
        A 7-character unique identifier based on the current timestamp.

    Examples
    --------
    >>> id = temporal_id()
    >>> print(id)
    'a1b2c3d'
    """
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    hash_object = hashlib.sha256(time_str.encode())
    return hash_object.hexdigest()[:7]


def extract_yaml_id(file_path):
    """
    Extract the integer identifier from a YAML file path.

    This function uses a regular expression to find and return the integer
    that precedes the ".yml" extension in a file path.

    Parameters
    ----------
    file_path : str
        The path to the YAML file.

    Returns
    -------
    int or None
        The extracted integer identifier, or None if not found.

    Examples
    --------
    >>> extract_yaml_id("universe_42.yml")
    42
    >>> extract_yaml_id("config.yml")
    None
    """
    match = re.search(r"_(\d+)\.yml$", file_path)
    if match:
        return int(match.group(1))
    return None


def file_id_sorter(file_name):
    """
    Sort files by their numeric identifiers.

    This function extracts the integer identifier from a filename and
    returns it, allowing for numerical sorting of files with similar names.

    Parameters
    ----------
    file_name : str
        The name of the file.

    Returns
    -------
    int or float
        The extracted integer identifier, or float('inf') if not found.

    Examples
    --------
    >>> file_id_sorter("universe_5.yml")
    5
    >>> file_id_sorter("other_file.txt")
    inf
    """
    match = re.search(r"universe_(\d+)\.yml", file_name)
    return int(match.group(1)) if match else float("inf")


def get_wandb_env() -> Dict[str, str]:
    """
    Load Weights & Biases (WANDB) configuration from environment variables.

    This function reads WANDB-related configuration from environment variables,
    providing default values if variables are not set.

    Returns
    -------
    dict
        A dictionary containing WANDB configuration with the following keys:
        - 'wandb_enabled': Whether WANDB is enabled (bool).
        - 'wandb_project': The WANDB project name (str).
        - 'wandb_entity': The WANDB entity name (str).
        - 'wandb_tag': A tag for WANDB runs (str).

    Examples
    --------
    >>> config = get_wandb_env()
    >>> print(config)
    {'wandb_enabled': True, 'wandb_project': 'my_project', 'wandb_entity': 'my_entity', 'wandb_tag': 'LSD'}
    """
    # Load environment variables from a .env file if present
    load_dotenv()

    # Fetch WANDB configurations with default values
    wandb_project = os.getenv("WANDB_PROJECT", None)
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    wandb_tag = os.getenv("WANDB_TAG", "LSD")

    wandb_enabled = (
        os.getenv("WANDB", "False").lower() == "true"
        and wandb_project
        and wandb_entity
    )

    # Return configuration as a dictionary
    return {
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_tag": wandb_tag,
    }


def test_wandb_connection(config: Dict[str, str]) -> bool:
    """
    Test the connection to Weights & Biases (WANDB).

    This function attempts to connect to WANDB using the provided configuration,
    returning True if successful, or False if the connection fails.

    Parameters
    ----------
    config : dict
        The configuration dictionary for WANDB, typically retrieved using `get_wandb_env()`.

    Returns
    -------
    bool
        True if the connection to WANDB is successful, False otherwise.

    Examples
    --------
    >>> config = get_wandb_env()
    >>> success = test_wandb_connection(config)
    >>> print(success)
    True
    """
    if not config["wandb_enabled"]:
        print("WANDB is not enabled or missing necessary configurations.")
        return False

    try:
        # Initialize WANDB with the provided configuration
        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            tags=[config["wandb_tag"]],
            mode="online",
        )
        return True
    except Exception as e:
        print(f"Failed to connect to WANDB: {e}")
        return False


def write_pkl(data, path):
    """
    Write data to a pickle file.

    This function serializes the given data and writes it to the specified
    file path in binary format.

    Parameters
    ----------
    data : any
        The data to be serialized and written to a file.
    path : str
        The path to the file where data will be written.

    Examples
    --------
    >>> my_data = {'a': 1, 'b': 2}
    >>> write_pkl(my_data, 'data.pkl')
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read(path):
    """
    Read data from a file.

    This function reads data from a file and returns it. It supports
    `.npz` and `.pkl` file formats.

    Parameters
    ----------
    path : str
        The path to the file to be read.

    Returns
    -------
    any
        The data read from the file.

    Raises
    ------
    NotImplementedError
        If the file type is not supported.

    Examples
    --------
    >>> data = read('data.pkl')
    >>> print(data)
    {'a': 1, 'b': 2}
    """
    if path.endswith(".npz"):
        return np.load(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise NotImplementedError(f"File type not supported: {path}")
