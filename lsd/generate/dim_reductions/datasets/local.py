def mnist(**kwargs):
    bundle = load_local_data("mnist")
    data, labels = bundle["data"], bundle["labels"]
    mask = range(len(data))
    # if kwargs["N"] is not None:
    #     mask = generate_sampling_mask(len(data), kwargs["N"], kwargs["random_state"])
    return data[mask], labels[mask]


def ipsc(**kwargs):
    data = load_local_data("ipsc")["data"]
    labels = np.zeros(len(data))
    mask = range(len(data))
    # if kwargs["N"] is not None:
    #     mask = generate_sampling_mask(len(data), kwargs["N"], kwargs["random_state"])
    return data[mask], labels[mask]
