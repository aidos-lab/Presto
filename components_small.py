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

    # Let's try for one fixed config first
    dataset = "EdinburghNLP/xsum"
    model = "all-mpnet-base-v2"
    n_samples = 1024
    n_projections = 64

    component_ns = [2, 3, 4, 5]

    for n_components in component_ns:
        metric = Presto(max_homology_dim=config.max_homology_dim, n_components=n_components,
                        normalize=config.normalize)
        filename = f"{data_dir}/derivatives_{clean_dataset_name(dataset)}_{n_samples}_{model}_np-{n_projections}_nc-{n_components}.pkl"
        if overwrite or not os.path.isfile(filename):
            print(f"Working to create {filename}")
            e = embeddings[dataset][model]
            projections = metric.generate_projections(e[:n_samples, :], max_projections)
            landscapes = metric.generate_landscapes(projections)
            with open(filename, "wb") as f:
                pickle.dump(dict(projections=projections, landscapes=landscapes), f)
