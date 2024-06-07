import pytest
import omegaconf
from itertools import product

from lsd.lsd import LSD
from lsd import utils as ut
from lsd.config import AutoencoderMultiverse


def test_cfg():
    lsd = LSD("AutoencoderMultiverse")
    assert lsd.cfg == AutoencoderMultiverse

    with pytest.raises(ValueError):
        lsd = LSD("TestMultiverse")


def test_load_params(test_dict1, test_yaml1_file):
    lsd = LSD("AutoencoderMultiverse")

    # Manually Set config
    lsd.cfg.model_choices = test_yaml1_file
    lsd.cfg.data_choices = test_yaml1_file
    lsd.cfg.implementation_choices = test_yaml1_file

    D = lsd.load_params(lsd.cfg.data_choices)
    M = lsd.load_params(lsd.cfg.model_choices)
    I = lsd.load_params(lsd.cfg.implementation_choices)

    assert D == test_dict1
    assert M == test_dict1
    assert I == test_dict1

    with pytest.raises(AssertionError):
        lsd.load_params("non_existent_file.yoml")

    with pytest.raises(AssertionError):
        lsd.load_params("non_existent_file.yaml")


def test_filter_params(test_dict1, test_yaml1_file):
    lsd = LSD("AutoencoderMultiverse")

    lsd.cfg.model_choices = test_yaml1_file
    lsd.cfg.data_choices = test_yaml1_file
    lsd.cfg.implementation_choices = test_yaml1_file

    assert lsd.cfg.id_ == "Autoencoder"

    with pytest.raises(AttributeError):
        lsd.filter_params(
            path=lsd.cfg.data_choices,
            choices="NonExistent",
            id_="Autoencoder",
        )

    assert (
        lsd.filter_params(
            path=lsd.cfg.data_choices,
            choices="model_choices",
            id_="Autoencoder",
        )
        is None
    )

    D = lsd.filter_params(
        path=lsd.cfg.data_choices, choices="data_choices", id_=lsd.cfg.id_
    )
    I = lsd.filter_params(
        path=lsd.cfg.implementation_choices,
        choices="implementation_choices",
        id_=lsd.cfg.id_,
    )

    assert D == test_dict1["data_choices"]["Autoencoder"]
    assert I == test_dict1["implementation_choices"]["Autoencoder"]


def test_get_all_keys(test_nested_dict):

    assert set(ut.get_all_keys(test_nested_dict)) == set(
        [
            "Test1",
            "Test1.Test2",
            "Test1.Test3",
            "Test1.Test2.Test4",
        ]
    )


def test_load_multiverse(test_yaml1_file, test_yaml2_file):
    lsd = LSD("AutoencoderMultiverse")

    lsd.cfg.model_choices = test_yaml1_file
    lsd.cfg.data_choices = test_yaml1_file
    lsd.cfg.implementation_choices = test_yaml1_file

    with pytest.raises(AssertionError):
        lsd.load_multiverse()

    lsd.cfg.model_choices = test_yaml2_file

    multiverse = lsd.multiverse

    assert type(lsd.D) == omegaconf.dictconfig.DictConfig
    assert type(lsd.M) == omegaconf.dictconfig.DictConfig
    assert type(lsd.I) == omegaconf.dictconfig.DictConfig
