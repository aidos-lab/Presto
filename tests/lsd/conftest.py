import pytest
import os
import tempfile
from omegaconf import OmegaConf
from contextlib import contextmanager


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
          epochs:
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
          hidden_dims:
            - [8, 16]
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
          epochs:
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
          num_samples:
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


@pytest.fixture
def test_multiverse4():
    return """
    data_choices:
      Transformer:
        CNN:
          num_samples:
            - 1000
            - 10000

    model_choices:
      Transformer:
        Mistral:
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
    """


@pytest.fixture
def test_dict4(test_multiverse4):
    config = OmegaConf.create(test_multiverse4)
    return config


@pytest.fixture
def test_yaml4_file(test_multiverse4):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_multiverse4)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_multiverse5():
    return """
    data_choices:
      Custom:
        arXiv:
          batch_size:
            - 64
            - 128
          train_test_split:
            - [0.6, 0.3, 0.1]
        CNN:
          batch_size:
            - 64
            - 128
          train_test_split:
            - [0.6, 0.3, 0.1]
    model_choices:
    implementation_choices:
    """


@pytest.fixture
def test_dict5(test_multiverse5):
    config = OmegaConf.create(test_multiverse5)
    return config


@pytest.fixture
def test_yaml5_file(test_multiverse5):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_multiverse5)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_ae_beta_multiverse():
    return """
    data_choices:
      Autoencoder:
        MNIST:
          batch_size:
            - 64
          train_test_split:
            - [0.7, 0.2, 0.1]
          sample_size:
            - 0.001

    model_choices:
      Autoencoder:
        betaVAE:
          beta:
            - 0.1
          gamma:
            - 0
          max_capacity:
            - 20
          C_max_iter:
            - 1e4
          hidden_dims:
            - [4]
    implementation_choices:
      Autoencoder:
        Adam:
          lr:
            - 0.1
          epochs:
            - 2
    """


@pytest.fixture
def test_yaml_ae_beta_file(test_ae_beta_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_ae_beta_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_ae_info_multiverse():
    return """
    data_choices:
      Autoencoder:
        MNIST:
          batch_size:
            - 64
          train_test_split:
            - [0.6, 0.3, 0.1]
          sample_size:
            - 0.001

    model_choices:
      Autoencoder:
        infoVAE:
          alpha:
            - 0.5
          kernel:
            - imq
            - rbf
          hidden_dims:
            - [4]

    implementation_choices:
      Autoencoder:
        SGD:
          lr:
            - 0.01
          momentum:
            - 0.75
          epochs:
            - 1
    """


@pytest.fixture
def test_yaml_ae_info_file(test_ae_info_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_ae_info_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_ae_wae_multiverse():
    return """
    data_choices:
      Autoencoder:
        MNIST:
          batch_size:
            - 64
          train_test_split:
            - [0.6, 0.3, 0.1]
          sample_size:
            - 0.001

    model_choices:
      Autoencoder:
        WAE:
          z_var:
            - 0.5
          reg_weight:
            - 2.0
          kernel:
            - imq
            - rbf
          hidden_dims:
            - [4]

    implementation_choices:
      Autoencoder:
        SGD:
          lr:
            - 0.01
          momentum:
            - 0.75
          epochs:
            - 1
    """


@pytest.fixture
def test_yaml_ae_wae_file(test_ae_wae_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_ae_wae_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_ae_no_train_multiverse():
    return """
    data_choices:
      Autoencoder:
        MNIST:
          batch_size:
            - 64
          train_test_split:
            - [0.6, 0.3, 0.1]
          sample_size:
            - 0.001

    model_choices:
      Autoencoder:
        WAE:
          z_var:
            - 0.5
          reg_weight:
            - 2.0
          kernel:
            - imq
          hidden_dims:
            - [4]
          latent_dim:
            - 2

    implementation_choices:
      Autoencoder:
        SGD:
          lr:
            - 0.01
          momentum:
            - 0.75
          epochs:
            - 0
"""


@pytest.fixture
def test_yaml_ae_no_train_file(test_ae_no_train_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_ae_no_train_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_umap_multiverse():
    return """
    data_choices:
      DimReduction:
        wine:
          generator:
            - load_wine
          
    model_choices:
      DimReduction:
        UMAP:
          nn:
            - 16
          min_dist:
            - 0

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_umap_file(test_dr_umap_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_umap_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_tsne_multiverse():
    return """
    data_choices:
      DimReduction:
        breast_cancer:
          generator:
            - load_breast_cancer
          
    model_choices:
      DimReduction:
        tSNE:
          perplexity:
            - 15

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_tsne_file(test_dr_tsne_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_tsne_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_phate_multiverse():
    return """
    data_choices:
      DimReduction:
        iris:
          generator:
            - load_iris
          
    model_choices:
      DimReduction:
        Phate:
          k:
            - 5
          gamma:
            - 1.0
          decay:
            - 0.5
          t:
            - auto

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_phate_file(test_dr_phate_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_phate_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_isomap_multiverse():
    return """
    data_choices:
      DimReduction:
        iris:
          generator:
            - load_iris
          
    model_choices:
      DimReduction:
        Isomap:
          nn:
            - 30
          metric:
            - manhattan

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_isomap_file(test_dr_isomap_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_isomap_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_lle_multiverse():
    return """
    data_choices:
      DimReduction:
        iris:
          generator:
            - load_iris
          
    model_choices:
      DimReduction:
        LLE:
          nn:
            - 5
          reg:
            - 0.001

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_lle_file(test_dr_lle_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_lle_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_local_data_multiverse():
    return """
    data_choices:
      DimReduction:
        MNIST:
          num_samples:
            - 1000
          path:
            - /Users/jeremy.wayland/Downloads/mnist.npz
          seed:
            - 68
          
    model_choices:
      DimReduction:
        LLE:
          nn:
            - 5
          reg:
            - 0.001

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_local_data_file(test_dr_local_data_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_local_data_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_manifold_data_multiverse():
    return """
    data_choices:
      DimReduction:
        swiss_roll:
          generator:
            - swiss_roll
          num_samples:
            - 1000
          seed:
            - 68
        moons:
          generator:
            - moons
          num_samples:
            - 1000
          seed:
            - 68
        barbell:
          generator:
            - barbell
          num_samples:
            - 1000
          seed:
            - 68
          beta:
            - 0.67
        noisy_annulus:
          generator:
            - noisy_annulus
          num_samples:
            - 1000
          inner_radius:
            - 68
          outer_radius:
            - 100
          
    model_choices:
      DimReduction:
        LLE:
          nn:
            - 5
          reg:
            - 0.001

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    logging:
      wandb_logging: False
      out_file: True
      dev: False
    """


@pytest.fixture
def test_yaml_dr_manifold_data_file(test_dr_manifold_data_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_manifold_data_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@pytest.fixture
def test_dr_pca_training_multiverse():
    return """
    data_choices:
      DimReduction:
        MNIST:
          num_samples:
            - 100
          path:
            - /Users/jeremy.wayland/Downloads/mnist.npz
          seed:
            - 68
          
    model_choices:
      DimReduction:
        LLE:
          nn:
            - 5
          reg:
            - 0.001
          max_ambient_dim:
            - 20
            - 

    implementation_choices:
      DimReduction:
        Thread:
          n_jobs:
            - 1
    """


@pytest.fixture
def test_yaml_dr_pca_training_file(test_dr_pca_training_multiverse):
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".yaml"
    ) as temp_file:
        temp_file.write(test_dr_pca_training_multiverse)
        temp_file_path = temp_file.name

    yield temp_file_path

    os.remove(temp_file_path)


@contextmanager
def set_env_var(key: str, value: str):
    """
    A context manager to temporarily set an environment variable.

    Args:
        key (str): The name of the environment variable.
        value (str): The value to set for the environment variable.
    """
    old_value = os.getenv(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        else:
            del os.environ[key]
