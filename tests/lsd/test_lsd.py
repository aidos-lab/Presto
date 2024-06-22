import os
import pickle
import tempfile
from itertools import product
import numpy as np

import omegaconf
import pytest
from memory_profiler import profile
from omegaconf.errors import ConfigKeyError

from lsd import utils as ut
from lsd.config import AutoencoderMultiverse
from lsd.generate.autoencoders.models.beta import BetaVAE
from lsd.generate.autoencoders.models.wae import WAE
from lsd.lsd import LSD


def test_cfg():
    with pytest.raises(TypeError):
        LSD("AutoencoderMultiverse")
    with pytest.raises(AssertionError):
        LSD("AutoencoderMultiverse", outDir="some/path/to/dir")

    with tempfile.TemporaryDirectory() as tmp_dir:
        lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)
        assert lsd.cfg == AutoencoderMultiverse

        with pytest.raises(ValueError):
            lsd = LSD("TestMultiverse", outDir=tmp_dir)


def test_read_params(test_dict1, test_yaml1_file):
    with tempfile.TemporaryDirectory() as tmp_dir:
        lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        # Manually Set config
        lsd.cfg.model_choices = test_yaml1_file
        lsd.cfg.data_choices = test_yaml1_file
        lsd.cfg.implementation_choices = test_yaml1_file

        D = lsd.read_params(lsd.cfg.data_choices)
        M = lsd.read_params(lsd.cfg.model_choices)
        I = lsd.read_params(lsd.cfg.implementation_choices)

        assert D == test_dict1
        assert M == test_dict1
        assert I == test_dict1

        with pytest.raises(AssertionError):
            lsd.read_params("non_existent_file.yoml")

        with pytest.raises(AssertionError):
            lsd.read_params("non_existent_file.yaml")


def test_filter_params(test_dict1, test_yaml1_file):
    with tempfile.TemporaryDirectory() as tmp_dir:
        lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

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
    with tempfile.TemporaryDirectory() as tmp_dir:

        lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        lsd.multiverse = test_dict2

        assert lsd._multiverse is not None

        with pytest.raises(AssertionError):
            lsd.multiverse = test_nested_dict

        assert lsd.data_choices == test_dict2["data_choices"]
        assert lsd.model_choices == test_dict2["model_choices"]
        assert (
            lsd.implementation_choices == test_dict2["implementation_choices"]
        )

        lsd.cfg.model_choices = test_yaml1_file
        lsd.cfg.data_choices = test_yaml1_file
        lsd.cfg.implementation_choices = test_yaml1_file

        lsd.cfg.model_choices = test_yaml2_file

        assert lsd.multiverse is not None
        assert lsd.data_choices == test_dict1["data_choices"]
        assert lsd.model_choices != test_dict1["model_choices"]
        assert lsd.model_choices == test_dict2["model_choices"]
        assert (
            lsd.implementation_choices == test_dict1["implementation_choices"]
        )


def test_cartesian_product(test_yaml2_file, test_dict4):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_lsd = LSD("TransformerMultiverse", outDir=tmp_dir)

        tf_lsd.multiverse = test_dict4

        label = "data_choices"
        parameters = ["RIDICULOUS"]

        assert list(tf_lsd._cartesian_product(label, "MNIST", parameters)) == [
            ()
        ]
        assert list(
            tf_lsd._cartesian_product(label, "Ridiculous Model", parameters)
        ) == [()]

        label = "model_choices"
        generator = "Mistral"
        parameters = ["nn", "min_dist"]

        output = tf_lsd._cartesian_product(label, generator, parameters)

        assert isinstance(output, product)
        assert list(output) == [(16, 0), (20, 0)]

        ae_lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        ae_lsd.cfg.model_choices = test_yaml2_file
        ae_lsd.cfg.data_choices = test_yaml2_file
        ae_lsd.cfg.implementation_choices = test_yaml2_file

        label = "model_choices"
        generator = "betaVAE"
        parameters = ["beta", "gamma"]
        output = ae_lsd._cartesian_product(label, generator, parameters)

        assert isinstance(output, product)
        assert list(output) == [(0.1, 0), (0.1, 100), (0.01, 0), (0.01, 100)]


def test_design(
    test_yaml2_file,
    test_dict3,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Autoencoder
        ae_lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        ae_lsd.cfg.model_choices = test_yaml2_file
        ae_lsd.cfg.data_choices = test_yaml2_file
        ae_lsd.cfg.implementation_choices = test_yaml2_file

        ae_lsd.design()

        configs = os.path.join(tmp_dir, f"{ae_lsd.experimentName}/configs")
        assert os.path.exists(configs)
        for file in os.listdir(configs):
            assert file.startswith("universe_")
            assert file.endswith(".yml")

        U0 = os.path.join(configs, "universe_0.yml")
        cfg0 = omegaconf.OmegaConf.load(U0)

        assert cfg0.data_choices.name == "MNIST"
        assert cfg0.data_choices.batch_size == 64
        assert cfg0.data_choices.train_test_split == [0.6, 0.3, 0.1]
        assert cfg0.model_choices.name == "Beta Variational Autoencoder"
        assert cfg0.model_choices.beta == 0.1
        assert cfg0.implementation_choices.epochs == 50
        assert cfg0.implementation_choices.lr == 0.1

        U1 = os.path.join(configs, "universe_1.yml")
        cfg1 = omegaconf.OmegaConf.load(U1)

        assert cfg1.data_choices.name == "MNIST"
        assert cfg1.data_choices.batch_size == 64
        assert cfg1.data_choices.train_test_split == [0.6, 0.3, 0.1]
        assert cfg1.model_choices.name == "Beta Variational Autoencoder"
        assert cfg1.model_choices.beta == 0.1
        assert cfg1.implementation_choices.epochs == 50
        assert cfg1.implementation_choices.lr == 0.01

        # DimReduction
        dr_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)

        dr_lsd.multiverse = test_dict3
        dr_lsd.design()

        configs = os.path.join(tmp_dir, f"{dr_lsd.experimentName}/configs")
        assert os.path.exists(configs)
        for file in os.listdir(configs):
            assert file.startswith("universe_")
            assert file.endswith(".yml")

        U0 = os.path.join(configs, "universe_0.yml")
        cfg0 = omegaconf.OmegaConf.load(U0)

        assert cfg0.data_choices.name == "MNIST"
        assert cfg0.data_choices.num_samples == 1000
        assert (
            cfg0.model_choices.name
            == "Uniform Manifold Approximation and Projection"
        )
        assert cfg0.model_choices.nn == 16
        assert cfg0.model_choices.min_dist == 0
        assert cfg0.implementation_choices.n_jobs == 1

        U1 = os.path.join(configs, "universe_1.yml")
        cfg1 = omegaconf.OmegaConf.load(U1)

        assert cfg1.data_choices.name == "MNIST"
        assert cfg1.data_choices.num_samples == 1000
        assert (
            cfg1.model_choices.name
            == "Uniform Manifold Approximation and Projection"
        )
        assert cfg1.model_choices.nn == 16
        assert cfg1.model_choices.min_dist == 0
        assert cfg1.implementation_choices.n_jobs == -1


@profile
def test_ae_generation(
    test_yaml_ae_beta_file,
    test_yaml_ae_info_file,
    test_yaml_ae_wae_file,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        beta_lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        beta_lsd.cfg.model_choices = test_yaml_ae_beta_file
        beta_lsd.cfg.data_choices = test_yaml_ae_beta_file
        beta_lsd.cfg.implementation_choices = test_yaml_ae_beta_file

        beta_lsd.generate()

        info_lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        info_lsd.cfg.model_choices = test_yaml_ae_info_file
        info_lsd.cfg.data_choices = test_yaml_ae_info_file
        info_lsd.cfg.implementation_choices = test_yaml_ae_info_file

        info_lsd.generate()

        wae_lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        wae_lsd.cfg.model_choices = test_yaml_ae_wae_file
        wae_lsd.cfg.data_choices = test_yaml_ae_wae_file
        wae_lsd.cfg.implementation_choices = test_yaml_ae_wae_file

        wae_lsd.generate()


def test_dr_generation(
    test_yaml_dr_umap_file,
    test_yaml_dr_tsne_file,
    test_yaml_dr_isomap_file,
    test_yaml_dr_lle_file,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # UMAP
        umap_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)

        umap_lsd.cfg.model_choices = test_yaml_dr_umap_file
        umap_lsd.cfg.data_choices = test_yaml_dr_umap_file
        umap_lsd.cfg.implementation_choices = test_yaml_dr_umap_file

        umap_lsd.generate()

        # t-SNE
        tsne_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)

        tsne_lsd.cfg.model_choices = test_yaml_dr_tsne_file
        tsne_lsd.cfg.data_choices = test_yaml_dr_tsne_file
        tsne_lsd.cfg.implementation_choices = test_yaml_dr_tsne_file

        tsne_lsd.generate()

        # Isomap
        isomap_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)
        isomap_lsd.cfg.model_choices = test_yaml_dr_isomap_file
        isomap_lsd.cfg.data_choices = test_yaml_dr_isomap_file
        isomap_lsd.cfg.implementation_choices = test_yaml_dr_isomap_file

        isomap_lsd.generate()

        # LLE
        lle_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)
        lle_lsd.cfg.model_choices = test_yaml_dr_lle_file
        lle_lsd.cfg.data_choices = test_yaml_dr_lle_file
        lle_lsd.cfg.implementation_choices = test_yaml_dr_lle_file

        lle_lsd.generate()


def test_dr_data(
    test_yaml_dr_local_data_file,
    test_yaml_dr_manifold_data_file,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Local Data
        dr_local_data_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)

        dr_local_data_lsd.cfg.model_choices = test_yaml_dr_local_data_file
        dr_local_data_lsd.cfg.data_choices = test_yaml_dr_local_data_file
        dr_local_data_lsd.cfg.implementation_choices = (
            test_yaml_dr_local_data_file
        )

        dr_local_data_lsd.generate()

        dr_manifold_data_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)

        dr_manifold_data_lsd.cfg.model_choices = test_yaml_dr_manifold_data_file
        dr_manifold_data_lsd.cfg.data_choices = test_yaml_dr_manifold_data_file
        dr_manifold_data_lsd.cfg.implementation_choices = (
            test_yaml_dr_manifold_data_file
        )

        dr_manifold_data_lsd.generate()


def test_generate_io(
    test_yaml_dr_lle_file,
    test_yaml_ae_no_train_file,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dr_lsd = LSD("DimReductionMultiverse", outDir=tmp_dir)

        dr_lsd.cfg.model_choices = test_yaml_dr_lle_file
        dr_lsd.cfg.data_choices = test_yaml_dr_lle_file
        dr_lsd.cfg.implementation_choices = test_yaml_dr_lle_file

        dr_lsd.generate()

        latents = os.path.join(dr_lsd.outDir, "latent_spaces/")
        assert os.path.isdir(latents)
        pkl_file = os.path.join(latents, "universe_0.pkl")
        assert os.path.isfile(pkl_file)

        with open(pkl_file, "rb") as f:
            L = pickle.load(f)

        assert isinstance(L, np.ndarray)
        assert L.shape == (150, 2)

        assert len(os.listdir(latents)) == 1

        ae_lsd = LSD("AutoencoderMultiverse", outDir=tmp_dir)

        ae_lsd.cfg.model_choices = test_yaml_ae_no_train_file
        ae_lsd.cfg.data_choices = test_yaml_ae_no_train_file
        ae_lsd.cfg.implementation_choices = test_yaml_ae_no_train_file

        ae_lsd.generate()

        latents = os.path.join(ae_lsd.outDir, "latent_spaces/")
        assert os.path.isdir(latents)
        pkl_file = os.path.join(latents, "universe_0.pkl")
        assert os.path.isfile(pkl_file)

        with open(pkl_file, "rb") as f:
            L = pickle.load(f)

        assert isinstance(L, np.ndarray)
        assert L.shape == (42, 2)

        models = os.path.join(ae_lsd.outDir, "models/")
        assert os.path.isdir(models)
        model_file = os.path.join(models, "model_0.pkl")
        assert os.path.isfile(model_file)

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        assert isinstance(model, WAE)
        assert model.latent_dim == 2
        assert model.z_var == 0.5
        assert model.kernel_type == "imq"

        logs = os.path.join(ae_lsd.outDir, "logs/")
        assert os.path.isdir(logs)

        assert len(os.listdir(models)) == 1
        assert len(os.listdir(latents)) == 1
        assert len(os.listdir(logs)) == 1
