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
    sample_ns = config.n_samples
    max_samples = max(sample_ns)
    max_projections = config.n_projections.max()
    data_dir = "./data"

    embeddings = {d: dict() for d in config.datasets}
    for dataset in datasets:
        for model in models:
            filename = f"{data_dir}/embeddings_{re.sub(' ', '___', dataset)}_{max_samples}_{model}.pkl"
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    embeddings[dataset][model] = pickle.load(f)

    embedding_combinations = list(combinations(embeddings, 2)) + [(e, e) for e in embeddings]

    metric = Presto(max_homology_dim=config.max_homology_dim, n_components=config.n_components,
                    normalize=config.normalize)

    for dataset in datasets:
        for model, e in embeddings[dataset].items():
            filename = f"{data_dir}/derivatives_{re.sub(' ', '___', dataset)}_{max_samples}_{model}.pkl"
            if overwrite or not os.path.isfile(filename):
                print(model, dataset, max_projections)
                projections = metric.generate_projections(e, max_projections)
                landscapes = metric.generate_landscapes(projections)
                with open(filename, "wb") as f:
                    pickle.dump(dict(projections=projections, landscapes=landscapes), f)
