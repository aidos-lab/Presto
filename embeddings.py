from presto import Presto
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os
import pickle
import re
from multiprocessing import Pool, cpu_count
from itertools import product


def get_sample_data(data, dataset, n_samples):
    if dataset == "cnn_dailymail 3.0.0":
        return_data = data["train"][:n_samples]["highlights"]
    elif dataset == "big_patent a":
        return_data = data["train"]["abstract"][:n_samples]
    elif dataset == "gfissore/arxiv-abstracts-2021":
        return_data = data["train"]["abstract"][:n_samples]
    else:
        raise NotImplementedError(dataset)
    return return_data


def worker_init(M, D, S, DD, O):
    global models
    global datasets
    global max_samples
    global data_dir
    global overwrite
    models = M
    datasets = D
    max_samples = S
    data_dir = DD
    overwrite = O


def clean_dataset_name(dataset):
    return re.sub("/", "_-_", re.sub(' ', '___', dataset))


def embed(model, dataset):
    print(dataset, max_samples, model)
    filename = f"{data_dir}/embeddings_{clean_dataset_name(dataset)}_{max_samples}_{model}.pkl"
    if overwrite or not os.path.isfile(filename):
        embeddings = models[model].encode(get_sample_data(datasets[dataset], dataset, max_samples),
                                          show_progress_bar=True)
        with open(filename, "wb") as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    overwrite = False
    config = OmegaConf.load("config.yml")
    print(config)
    data_dir = "data"
    os.makedirs(os.path.join(os.curdir, data_dir), exist_ok=True)

    models = {m: SentenceTransformer(*m.split()) for m in config.models}
    datasets = {d: load_dataset(*d.split()) for d in config.datasets}
    max_samples = max(config.n_samples)
    # we can get the smaller ones simply by slicing
    all_combinations = list(product(models, datasets))

    with Pool(cpu_count() - 2, initializer=worker_init,
              initargs=(models, datasets, max_samples, data_dir, overwrite)) as p:
        p.starmap(embed, all_combinations)
