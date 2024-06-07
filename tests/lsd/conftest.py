import pytest
import os
import tempfile
from omegaconf import OmegaConf


@pytest.fixture
def test_multiverse1():
    return """
    data_choices:
      Autoencoder:
        celebA:
          batch_size:
            - 64
            - 128
          train_test_split:
            - [0.6, 0.3, 0.1]
        MNIST:
          batch_size:
            - 64
            - 128
          train_test_split:
            - [0.6, 0.3, 0.1]

    model_choices:
      DimReduction:
        UMAP:
          nn:
            - 16
            - 20
          min_dist:
            - 0
        tSNE:
          perplexity:
            - 15
            - 30

    implementation_choices:
      Autoencoder:
        Adam:
          lr:
            - 0.1
            - 0.01
          Epochs:
            - 50
            - 100
    """


@pytest.fixture
def test_dict1(test_multiverse1):
    config = OmegaConf.create(test_multiverse1)
    return config


@pytest.fixture
def test_yaml1_file(test_multiverse1):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_multiverse1)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_multiverse2():
    return """
    data_choices:
      Autoencoder:
        celebA:
          batch_size:
            - 64
            - 128
          train_test_split:
            - [0.6, 0.3, 0.1]
        MNIST:
          batch_size:
            - 64
            - 128
          train_test_split:
            - [0.6, 0.3, 0.1]

    model_choices:
      Autoencoder:
        betaVAE:
          beta:
            - 0.1
            - 0.01
          gamma:
            - 0
            - 100
        infoVAE:
          alpha:
            - 0.1
            - 0.01
          beta:
            - 0
            - 100
    implementation_choices:
      Autoencoder:
        Adam:
          lr:
            - 0.1
            - 0.01
          Epochs:
            - 50
            - 100
    """


@pytest.fixture
def test_dict2(test_multiverse2):
    config = OmegaConf.create(test_multiverse2)
    return config


@pytest.fixture
def test_yaml2_file(test_multiverse2):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_multiverse2)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_nested_dict():
    return {"Test1": {"Test2": {"Test4": 1}, "Test3": 2}}


@pytest.fixture
def test_multiverse3():
    return """
    data_choices:
      DimReduction:
        MNIST:
          samples:
            - 1000
            - 10000

    model_choices:
      DimReduction:
        UMAP:
          nn:
            - 16
            - 20
          min_dist:
            - 0
        tSNE:
          perplexity:
            - 15
            - 30

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
            - -1
    """


@pytest.fixture
def test_dict3(test_multiverse3):
    config = OmegaConf.create(test_multiverse3)
    return config


@pytest.fixture
def test_yaml3_file(test_multiverse3):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_multiverse3)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)
