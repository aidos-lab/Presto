import itertools

from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import re
import pickle
import os
from itertools import combinations
from presto import Presto

if __name__ == "__main__":
    overwrite = False
    config = OmegaConf.load("config.yml")
    models = {m: SentenceTransformer(*m.split()) for m in config.models}
    datasets = {d: load_dataset(*d.split()) for d in config.datasets}
    max_samples = max(config.n_samples)
    data_dir = "data"

    embedding_combinations = list(combinations(models, 2)) + [(m, m) for m in models]

    metric = Presto(max_homology_dim=config.max_homology_dim, n_components=config.n_components,
                    normalize=config.normalize)

    results = {d: {m: dict() for m in models} for d in datasets}
    for dataset, model, n_samples in itertools.product(datasets, models, config.n_samples):
        filename = f"{data_dir}/derivatives_{re.sub(' ', '___', dataset)}_{n_samples}_{model}.pkl"
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                results[dataset][model][n_samples] = pickle.load(f)

    with open(f"{data_dir}/combined_results.pkl", "wb") as f:
        pickle.dump(results, f)
    # ...now we can flexibly compute scores, varying the number of projections considered
    # and the number of samples used to create the projections and landscapes
