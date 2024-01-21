from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle
import os
from presto import Presto
from embeddings import clean_dataset_name

if __name__ == "__main__":
    overwrite = False
    config = OmegaConf.load("config.yml")
    models = {m: SentenceTransformer(*m.split()) for m in config.models}
    datasets = {d: load_dataset(*d.split()) for d in config.datasets}
    max_samples = max(config.n_samples)
    max_projections = max(config.n_projections)
    data_dir = "data"

    embeddings = {d: dict() for d in config.datasets}
    for dataset in datasets:
        for model in models:
            filename = f"{data_dir}/embeddings_{clean_dataset_name(dataset)}_{max_samples}_{model}.pkl"
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    embeddings[dataset][model] = pickle.load(f)

    metric = Presto(max_homology_dim=config.max_homology_dim, n_components=config.n_components,
                    normalize=config.normalize)

    for dataset in datasets:
        for model, e in embeddings[dataset].items():
            for n_samples in config.n_samples:
                filename = f"{data_dir}/derivatives_{clean_dataset_name(dataset)}_{n_samples}_{model}.pkl"
                if overwrite or not os.path.isfile(filename):
                    print(f"Working to create {filename}")
                    projections = metric.generate_projections(e[:n_samples, :], max_projections)
                    landscapes = metric.generate_landscapes(projections)
                    with open(filename, "wb") as f:
                        pickle.dump(dict(projections=projections, landscapes=landscapes), f)
