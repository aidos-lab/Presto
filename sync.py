import wandb

artifact = wandb.Artifact(name="embeddings", type="dataset")
artifact.add_dir("data")

run = wandb.init(project="Multiverse", entity="aidos-labs", job_type="add-dataset")
run.log_artifact(artifact)
