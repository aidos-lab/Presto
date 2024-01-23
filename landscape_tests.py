"Testing `Presto` Scores on Random data"

import sys
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from presto import Presto


if __name__ == "__main__":
    
    n_components = 3
    n_projections = 1
    homology_dim = 1

    metric = Presto(n_components=n_components,normalize=True,max_homology_dim=homology_dim,projector=PCA)
    
    runs = []
    for _ in range(1000):
        X = np.random.random(size=(100, 1000))

        projections1 = metric.generate_projections(X, n_projections)
        projections2 = metric.generate_projections(X, n_projections)

        P1 = projections1[0]
        P2 = projections2[0]

        assert np.isclose(P1,P2).all(), "Projections are not equal"

        landscapes1 = metric.generate_landscapes(projections1)
        landscapes2 = metric.generate_landscapes(projections2)

        L1 = metric.average_landscape(landscapes1)
        L2 = metric.average_landscape(landscapes2)

        scores = metric.compute_presto_scores(L1,L2)

        runs.append(scores)


    colors = sns.color_palette('husl', n_colors=2)
    for dim in range(homology_dim+1):
        print(dim)
        scores = []
        for run in runs:
            scores.append(run[dim])
        
        sns.histplot(scores, bins=50, kde=False, color=colors[dim], edgecolor='black',label=f"Homology Dimension {dim}")

    plt.xlabel("Landscape Distance (L,L')")
    plt.ylabel('Frequency')
    plt.title("Landscapes distance between L(X) and L'(X)")
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig('normalized_histogram.png')


