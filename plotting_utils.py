import matplotlib.pyplot as plt


def get_short_name(dataset):
    if "patent" in dataset:
        return "patents"
    elif "arxiv" in dataset:
        return "arxiv"
    elif "cnn" in dataset:
        return "cnn"
    elif "xsum" in dataset:
        return "bbc"
    else:
        raise NotImplementedError(dataset)


def get_short_model_name(model):
    model_names = ['all-mpnet-base-v2',
                   'all-MiniLM-L6-v2',
                   'all-distilroberta-v1',
                   'multi-qa-distilbert-cos-v1']
    if model == model_names[0]:
        return "mpnet"
    elif model == model_names[1]:
        return "MiniLM"
    elif model == model_names[2]:
        return "distilroberta"
    elif model == model_names[3]:
        return "qa-distilbert"
    else:
        raise NotImplementedError(model)


def set_rcParams():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amssymb}\usepackage{amsmath}\usepackage{times}"
