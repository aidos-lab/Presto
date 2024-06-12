from dataclasses import fields
import hashlib
from datetime import datetime
import re


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
    return hash_object.hexdigest()


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
