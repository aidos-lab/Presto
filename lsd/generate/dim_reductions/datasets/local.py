import numpy as np


def generate_sampling_mask(N, num_samples, random_state):
    np.random.seed(random_state)
    mask = np.random.choice(range(N), num_samples)
    return mask


# Kaggle MNIST Version
def mnist(**kwargs):
    bundle = np.load(kwargs["path"])
    data = x_train = bundle["x_train"].reshape(-1, 28 * 28)
    labels = bundle["y_train"]
    mask = range(len(data))
    if kwargs["num_samples"]:
        mask = generate_sampling_mask(
            len(data), kwargs["num_samples"], kwargs["seed"]
        )
    return data[mask], labels[mask]
