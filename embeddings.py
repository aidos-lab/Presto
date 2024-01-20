from presto import Presto
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os
import pickle
import re


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


if __name__ == "__main__":
    overwrite = False
    config = OmegaConf.load("config.yml")
    print(config)
    data_dir = "data"
    os.makedirs(os.path.join(os.curdir, data_dir), exist_ok=True)

    models = {m: SentenceTransformer(*m.split()) for m in config.models}
    datasets = {d: load_dataset(*d.split()) for d in config.datasets}

    for dataset in config.datasets:
        for n_samples in config.n_samples:
            for model in models:
                print(dataset, n_samples, model)
                filename = f"{data_dir}/embeddings_{re.sub(' ', '___', dataset)}_{n_samples}_{model}.pkl"
                if overwrite or not os.path.isfile(filename):
                    embeddings = models[model].encode(get_sample_data(datasets[dataset], dataset, n_samples))
                    with open(filename, "wb") as f:
                        pickle.dump(embeddings, f)
