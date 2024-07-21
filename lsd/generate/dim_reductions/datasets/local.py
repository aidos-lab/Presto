import numpy as np


def generate_sampling_mask(N: int, num_samples: int, random_state: int):
    """
    Generate a random sampling mask for selecting a subset of data.

    Parameters
    ----------
    N : int
        The total number of data points from which to sample.
    num_samples : int
        The number of samples to select.
    random_state : int
        The seed for the random number generator to ensure reproducibility.

    Returns
    -------
    mask : ndarray of shape (num_samples,)
        An array of indices representing the selected samples.

    Notes
    -----
    The function sets the random seed for reproducibility and uses the `np.random.choice`
    method to generate a random sampling of indices without replacement from the range [0, N).
    """
    np.random.seed(random_state)
    mask = np.random.choice(range(N), num_samples)
    return mask


def mnist(**kwargs):
    """
    Load and optionally subsample the Kaggle MNIST dataset.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - path : str
            Path to the .npz file containing the MNIST dataset.
            The file should contain 'x_train' and 'y_train' keys.
        - num_samples : int, optional
            Number of samples to return. If not specified, all samples are used.
        - seed : int, optional
            Random seed for reproducibility when subsampling. Default is None.

    Returns
    -------
    data : ndarray of shape (num_samples, 784)
        The MNIST data points, where each row corresponds to a flattened 28x28 image.
    labels : ndarray of shape (num_samples,)
        The labels for the MNIST data points, where each value corresponds to the digit class.

    Notes
    -----
    The function assumes that the Kaggle MNIST dataset is stored in a .npz file with 'x_train' and 'y_train' keys. The 'x_train' data is reshaped into a 2D array where each row is a flattened 28x28 image.
    """
    bundle = np.load(kwargs["path"])
    data = bundle["x_train"].reshape(-1, 28 * 28)
    labels = bundle["y_train"]
    mask = range(len(data))
    if kwargs["num_samples"]:
        mask = generate_sampling_mask(
            len(data), kwargs["num_samples"], kwargs["seed"]
        )
    return data[mask], labels[mask]
