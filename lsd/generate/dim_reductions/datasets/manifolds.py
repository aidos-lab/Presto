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
    """Generate Swiss Roll data set."""

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
    """Generate set of Gaussian blobs."""

    return make_blobs(kwargs["num_samples"], random_state=kwargs["seed"])


def moons(**kwargs):
    """Generate moons data set with labels."""

    return make_moons(kwargs["num_samples"], random_state=kwargs["seed"])


def nested_circles(**kwargs):
    """Generate nested circles with labels."""

    return make_circles(kwargs["num_samples"], random_state=kwargs["seed"])


def barbell(**kwargs):
    """Generate uniformly-sampled 2-D barbelll with colours."""
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
    """Sample points from noisy annulus,
    with points obstructing a clear H1 feature.

    This function samples `kwargs["num_samples"]` points from an annulus with inner radius `r`
    and outer radius `R`, and then adds f*kwargs["num_samples"] noisy points to the interior.

    Parameters
    ----------
    kwargs["num_samples"] : int
        kwargs["num_samples"]umber of points to sample

    r : float
        Inner radius of annulus

    R : float
        Outer radius of annulus

    f : float
        Fraction of noisy points to include.


    **kwargs:
        Optional keyword arguments, such as a fixed random state for the
        pseudo-random number generator.

    Returns
    -------
    Array of (x, y) coordinates.
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


def synthetic_manifold_labels(data, n_clusters, k=10):
    """Assign Labels for Sklearn Data Sets based on KNN Graphs"""

    connectivity = kneighbors_graph(data, n_neighbors=k, include_self=False)
    ward = AgglomerativeClustering(
        n_clusters=n_clusters, connectivity=connectivity, linkage="ward"
    ).fit(data)
    labels = ward.labels_
    return labels
