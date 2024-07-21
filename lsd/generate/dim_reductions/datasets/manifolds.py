import numpy as np
from sklearn.datasets import (
    make_swiss_roll,
    make_blobs,
    make_moons,
    make_circles,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def swiss_roll(
    **kwargs,
):
    """
    Swiss Roll Generator.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - num_samples : int
            Number of samples to generate.
        - seed : int
            Random seed for reproducibility.
        - hole : bool
            If True, creates a hole in the Swiss Roll.
        - num_classes : int, optional
            Number of clusters to create. If not provided, all points are given the same label.

    Returns
    -------
    data : ndarray
        The Swiss Roll Embedding of shape (num_samples, 3).
    labels : ndarray
        KNN-generated cluster labels for the data points. Shape (num_samples,).
    """

    data, _ = make_swiss_roll(
        n_samples=kwargs["num_samples"],
        random_state=kwargs["seed"],
        hole=kwargs["hole"],
    )
    if kwargs["num_classes"]:
        labels = synthetic_manifold_labels(
            data,
            n_clusters=kwargs["num_classes"],
        )
    else:
        labels = np.ones(kwargs["num_samples"])

    return data, labels


def blobs(**kwargs):
    """
    Gaussian Blob Generator.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - num_samples : int
            Number of samples to generate.
        - seed : int
            Random seed for reproducibility.

    Returns
    -------
    data : ndarray
        The Gaussian blob embedding. Shape (num_samples, 2).
    labels : ndarray
        Labels for the blobs data points. Shape (num_samples,).
    """

    return make_blobs(kwargs["num_samples"], random_state=kwargs["seed"])


def moons(**kwargs):
    """
    Moons Generator.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - num_samples : int
            Number of samples to generate.
        - seed : int
            Random seed for reproducibility.

    Returns
    -------
    data : ndarray
        The Moons embedding. Shape (num_samples, 2).
    labels : ndarray
        Labels for the Moons data points. Shape (num_samples,).
    """

    return make_moons(kwargs["num_samples"], random_state=kwargs["seed"])


def nested_circles(**kwargs):
    """
    Nested Circles Generator.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - num_samples : int
            Number of samples to generate.
        - seed : int
            Random seed for reproducibility.

    Returns
    -------
    data : ndarray
        The Nested Circles embedding. Shape (num_samples, 2).
    labels : ndarray
        Labels for the Nested Circles data points. Shape (num_samples,).
    """

    return make_circles(kwargs["num_samples"], random_state=kwargs["seed"])


def barbell(**kwargs):
    """
    Barbell Generator.

    This function generates a uniformly-sampled 2-D barbelll with colours.


    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - num_samples : int
            Number of samples to generate.
        - seed : int
            Random seed for reproducibility.
        - beta : float
            Width of the barbell.
        - num_classes : int, optional
            Number of clusters to create. If not provided, all points are given the same label.

    Returns
    -------
    data : ndarray
        The Barbell embedding. Shape (num_samples, 2).
    labels : ndarray
        Labels for the Barbell data points. Shape (num_samples,).

    """
    if kwargs.get("seed"):
        np.random.seed(kwargs["seed"])
    beta = kwargs["beta"]

    X = []
    C = []
    k = 1

    while k <= kwargs["num_samples"]:
        x = (2 + beta / 2) * np.random.uniform()
        y = (2 + beta / 2) * np.random.uniform()

        n_prev = len(C)

        if (x - 0.5) ** 2 + (y - 0.5) ** 2 <= 0.25:
            C.append(0)

        elif abs(x - 1 - beta / 4) < beta / 4 and abs(y - 0.5) < 0.125:
            C.append(1)

        elif (x - 1.5 - beta / 2) ** 2 + (y - 0.5) ** 2 <= 0.25:
            C.append(2)

        if len(C) > n_prev:
            X.append((x, y))
            k += 1

    return np.asarray(X), np.asarray(C)


def noisy_annulus(**kwargs):
    """
    Noisy Annulus Generator.

    Sample points from noisy annulus, with points obstructing a clear H1 feature.

    This function samples `kwargs["num_samples"]` points from an annulus with inner radius `r`and outer radius `R`, and then adds f*kwargs["num_samples"] noisy points to the interior.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of keyword arguments.
        - num_samples : int
            Number of samples to generate.
        - seed : int
            Random seed for reproducibility.
        - inner_radius : float
            Inner radius of the annulus.
        - outer_radius : float
            Outer radius of the annulus.
        - noise : float
            Fraction of noisy points to add to the interior.
        - num_classes : int, optional
            Number of clusters to create. If not provided, all points are given the same label.

    Returns
    -------
    data : ndarray
        The Noisy Annulus embedding. Shape (num_samples, 2).
    labels : ndarray
        Labels for the Noisy Annulus data points. Shape (num_samples,).
    """
    r = kwargs["inner_radius"]
    R = kwargs["outer_radius"]
    f = kwargs["noise"]
    if r >= R:
        raise RuntimeError(
            "Inner radius must be less than or equal to outer radius"
        )

    if kwargs.get("seed"):
        np.random.seed(kwargs["seed"])

    # Take ceiling so we get at least one noisy point
    size = int(np.ceil(kwargs["num_samples"] * (1 + f)))
    thetas = np.random.uniform(0, 2 * np.pi, size)

    radii = np.sqrt(np.random.uniform(r**2, R**2, kwargs["num_samples"]))
    # Append noisy points
    radii = np.append(
        radii,
        np.sqrt(np.random.uniform(0, 0.5 * r**2, size - kwargs["num_samples"])),
    )

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))

    labels = synthetic_manifold_labels(X, n_clusters=kwargs["num_classes"])
    return X, labels


def synthetic_manifold_labels(data: np.ndarray, n_clusters: int, k: int = 10):
    """
    Assign cluster labels to a dataset using K-Nearest Neighbors (KNN) graphs and agglomerative clustering.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The input data for which to assign labels. Each row represents a data point, and each column represents a feature.

    n_clusters : int
        The number of clusters to form.

    k : int, optional, default=10
        The number of nearest neighbors to use for constructing the KNN graph.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The cluster labels assigned to each data point.

    Notes
    -----
    This function uses the KNN graph to represent the data structure and performs agglomerative clustering using the Ward linkage method to group the data into clusters.
    """

    connectivity = kneighbors_graph(data, n_neighbors=k, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    ).fit(data)
    labels = ward.labels_
    return labels
