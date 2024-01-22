from datasets import load_dataset

from itertools import product

from omegaconf import OmegaConf

import os
import llm
import pickle
import re

import numpy as np


def get_sample_data(data, dataset, n_samples):
    if dataset == "cnn_dailymail 3.0.0":
        return_data = data["train"][:n_samples]["highlights"]
    elif dataset == "big_patent a":
        return_data = data["train"]["abstract"][:n_samples]
    elif dataset == "gfissore/arxiv-abstracts-2021":
        return_data = data["train"]["abstract"][:n_samples]
    elif dataset == "EdinburghNLP/xsum":
        return_data = data["train"]["summary"][:n_samples]
    else:
        raise NotImplementedError(dataset)
    return return_data


def clean_dataset_name(dataset):
    return re.sub("/", "_-_", re.sub(' ', '___', dataset))


def embed(model, dataset):
    print(dataset, max_samples, model)
    filename = f"{data_dir}/embeddings_{clean_dataset_name(dataset)}_{max_samples}_{model}.pkl"

    if overwrite or not os.path.isfile(filename):
        print("-->", filename)

        embedding_model = llm.get_embedding_model(model)
        data = get_sample_data(datasets[dataset], dataset, max_samples)
        embeddings = list(embedding_model.embed_multi(data))
        embeddings = np.asarray(embeddings)

        with open(filename, "wb") as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    overwrite = False
    config = OmegaConf.load("config.yml")
    print(config)
    data_dir = "data"
    os.makedirs(os.path.join(os.curdir, data_dir), exist_ok=True)

    models = ["ada-002", "mistral-embed"]
    max_samples = max(config.n_samples)
    datasets = {d: load_dataset(*d.split()) for d in config.datasets}

    # we can get the smaller ones simply by slicing
    all_combinations = list(product(models, datasets))

    for combination in all_combinations:
        model, dataset = combination
        embed(model, dataset)
