def iris(**kwargs):
    data, labels = load_iris(return_X_y=True)
    mask = range(len(data))
    # if kwargs["N"] is not None:
    #     mask = generate_sampling_mask(len(data), kwargs["N"], kwargs["random_state"])
    return data[mask], labels[mask]


def diabetes(**kwargs):

    data, labels = load_diabetes(return_X_y=True)
    mask = range(len(data))
    # if kwargs["N"] is not None:
    #     mask = generate_sampling_mask(len(data), kwargs["N"], kwargs["random_state"])
    return data[mask], labels[mask]


def digits(**kwargs):

    return load_digits(return_X_y=True)


def linnerud(**kwargs):
    from sklearn.datasets import load_linnerud

    return load_linnerud(return_X_y=True)


def wine(**kwargs):
    from sklearn.datasets import load_wine

    return load_wine(return_X_y=True)


def breast_cancer(**kwargs):
    from sklearn.datasets import load_breast_cancer

    return load_breast_cancer(return_X_y=True)
