import hashlib
import os
import pickle
import re
from dataclasses import fields
from datetime import datetime
from typing import Dict

import numpy as np
from dotenv import load_dotenv


class LoadClass:
    """Unpack an input dictionary to load a class"""

    classFieldCache = {}

    @classmethod
    def instantiate(cls, classToInstantiate, argDict):
        if classToInstantiate not in cls.classFieldCache:
            cls.classFieldCache[classToInstantiate] = {
                f.name for f in fields(classToInstantiate) if f.init
            }

        fieldSet = cls.classFieldCache[classToInstantiate]
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return classToInstantiate(**filteredArgDict)


def temporal_id():
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
    hash_object = hashlib.sha256(time_str.encode())
    return hash_object.hexdigest()[:7]


def extract_yaml_id(file_path):
    # Use regex to find the integer just before ".yml"
    match = re.search(r"_(\d+)\.yml$", file_path)
    if match:
        return int(match.group(1))
    return None


def file_id_sorter(file_name):
    # Extract the number from the filename
    match = re.search(r"universe_(\d+)\.yml", file_name)
    return int(match.group(1)) if match else float("inf")


def get_wandb_env() -> Dict[str, str]:
    """
    Load WANDB configuration from environment variables,
    providing default values if variables are not set.

    Returns:
        Dict[str, str]: A dictionary containing WANDB configuration.
    """
    # Load environment variables from a .env file if present
    load_dotenv()

    # Fetch WANDB configurations with default values
    wandb_enabled = os.getenv("WANDB", "False").lower() == "true"
    wandb_project = os.getenv("WANDB_PROJECT", None)
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    wandb_tag = os.getenv("WANDB_TAG", None)

    # Return configuration as a dictionary
    return {
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_tag": wandb_tag,
    }


def write_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read(path):
    if path.endswith(".npz"):
        return np.load(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise NotImplementedError(f"File type not supported: {path}")
