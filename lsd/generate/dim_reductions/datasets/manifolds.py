import numpy as np
from sklearn.datasets import (
    make_swiss_roll,
    make_blobs,
    make_moons,
    make_circles,
)


def swiss_roll(
    N: int = 1500,
    hole: bool = False,
    n_clusters: int = 6,
    **kwargs,
):
    """Generate Swiss Roll data set."""
    from sklearn.datasets import make_swiss_roll

    data, _ = make_swiss_roll(
        n_samples=N,
        random_state=0,
        hole=hole,
    )

    labels = assign_labels(
        data,
        n_clusters,
    )

    return data, labels


def blobs(N, **kwargs):
    """Generate set of Gaussian blobs."""
    from sklearn.datasets import make_blobs

    return make_blobs(N, random_state=kwargs["random_state"])


def moons(N, **kwargs):
    """Generate moons data set with labels."""
    from sklearn.datasets import make_moons

    return make_moons(N, random_state=kwargs["random_state"])


def nested_circles(N, **kwargs):
    """Generate nested circles with labels."""
    from sklearn.datasets import make_circles

    return make_circles(N, random_state=kwargs["random_state"])


def barbell(N, beta=1, **kwargs):
    """Generate uniformly-sampled 2-D barbelll with colours."""
    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    X = []
    C = []
    k = 1

    while k <= N:
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


def double_annulus(N, **kwargs):
    """Sample N points from a double annulus."""
    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    X = []
    C = []
    for i in range(N):
        while True:
            t = [
                np.random.uniform(-50, 50, 1)[0],
                np.random.uniform(-50, 140, 1)[0],
            ]

            d = np.sqrt(np.dot(t, t))
            if d <= 50 and d >= 20:
                X.append(t)
                C.append(0)
                break

            d = np.sqrt(t[0] ** 2 + (t[1] - 90) ** 2)
            if d <= 50 and d >= 40:
                X.append(t)
                C.append(1)
                break

    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return np.asarray(X), np.asarray(C)


def annulus(N, r, R, **kwargs):
    """Sample points from annulus.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`.

    Parameters
    ----------
    N : int
        Number of points to sample

    r : float
        Inner radius of annulus

    R : float
        Outer radius of annulus

    **kwargs:
        Optional keyword arguments, such as a fixed random state for the
        pseudo-random number generator.

    Returns
    -------
    Array of (x, y) coordinates.
    """
    if r >= R:
        raise RuntimeError(
            "Inner radius must be less than or equal to outer radius"
        )

    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    thetas = np.random.uniform(0, 2 * np.pi, N)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(np.random.uniform(r**2, R**2, N))

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    return X, np.linspace(0, 1, N)


def noisy_annulus(N, r=2, R=6, f=0.01, **kwargs):
    """Sample points from noisy annulus,
    with points obstructing a clear H1 feature.

    This function samples `N` points from an annulus with inner radius `r`
    and outer radius `R`, and then adds f*N noisy points to the interior.

    Parameters
    ----------
    N : int
        Number of points to sample

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
    if r >= R:
        raise RuntimeError(
            "Inner radius must be less than or equal to outer radius"
        )

    if kwargs.get("random_state"):
        np.random.seed(kwargs["random_state"])

    # Take ceiling so we get at least one noisy point
    size = int(np.ceil(N * (1 + f)))
    thetas = np.random.uniform(0, 2 * np.pi, size)

    # Need to sample based on squared radii to account for density
    # differences.
    radii = np.sqrt(np.random.uniform(r**2, R**2, N))
    # Append noisy points
    radii = np.append(
        radii, np.sqrt(np.random.uniform(0, 0.5 * r**2, size - N))
    )

    X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))

    labels = assign_labels(X, n_clusters=kwargs["n_clusters"])
    return X, labels
