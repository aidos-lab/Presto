import pytest
import omegaconf
from itertools import product

from lsd.lsd import LSD
from lsd.generate.autoencoders.models.beta import BetaVAE

# from lsd.generate.dim_reductions.models.umap import UMAP
from lsd import utils as ut
from lsd.config import AutoencoderMultiverse


def test_cfg():
    lsd = LSD("AutoencoderMultiverse")
    assert lsd.cfg == AutoencoderMultiverse

    with pytest.raises(ValueError):
        lsd = LSD("TestMultiverse")

    with pytest.raises(AssertionError):
        lsd = LSD("CustomMultiverse")


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

    assert lsd.cfg.base == "Autoencoder"

    with pytest.raises(AttributeError):
        lsd.filter_params(
            path=lsd.cfg.data_choices,
            choices="NonExistent",
            base="Autoencoder",
        )

    assert (
        lsd.filter_params(
            path=lsd.cfg.data_choices,
            choices="model_choices",
            base="Autoencoder",
        )
        is None
    )

    D = lsd.filter_params(
        path=lsd.cfg.data_choices, choices="data_choices", base=lsd.cfg.base
    )
    I = lsd.filter_params(
        path=lsd.cfg.implementation_choices,
        choices="implementation_choices",
        base=lsd.cfg.base,
    )

    assert D == test_dict1["data_choices"]["Autoencoder"]
    assert I == test_dict1["implementation_choices"]["Autoencoder"]


def test_multiverse_getter_setter(
    test_dict1,
    test_dict2,
    test_nested_dict,
    test_yaml1_file,
    test_yaml2_file,
):
    lsd = LSD("AutoencoderMultiverse")

    lsd.multiverse = test_dict2

    assert lsd._multiverse is not None

    with pytest.raises(AssertionError):
        lsd.multiverse = test_nested_dict

    assert lsd.data_choices == test_dict2["data_choices"]
    assert lsd.model_choices == test_dict2["model_choices"]
    assert lsd.implementation_choices == test_dict2["implementation_choices"]

    lsd.cfg.model_choices = test_yaml1_file
    lsd.cfg.data_choices = test_yaml1_file
    lsd.cfg.implementation_choices = test_yaml1_file

    lsd.cfg.model_choices = test_yaml2_file

    assert lsd.multiverse is not None
    assert lsd.data_choices == test_dict1["data_choices"]
    assert lsd.model_choices != test_dict1["model_choices"]
    assert lsd.model_choices == test_dict2["model_choices"]
    assert lsd.implementation_choices == test_dict1["implementation_choices"]


def test_load_generator(test_yaml2_file):
    lsd = LSD("AutoencoderMultiverse")

    assert isinstance(
        lsd.load_generator("lsd.generate.autoencoders.models.beta"),
        type(BetaVAE),
    )

    with pytest.raises(ImportError):
        lsd.load_generator("non_existent_module")


def test_design(test_yaml2_file, test_dict3):
    ae_lsd = LSD("AutoencoderMultiverse")

    ae_lsd.cfg.model_choices = test_yaml2_file
    ae_lsd.cfg.data_choices = test_yaml2_file
    ae_lsd.cfg.implementation_choices = test_yaml2_file

    ae_lsd.design()

    dr_lsd = LSD("DimReductionMultiverse")

    dr_lsd.multiverse = test_dict3
    dr_lsd.design()
