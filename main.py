import time

import numpy as np

from pesto import PESTO

if __name__ == "__main__":
    start_time = time.time()

    metric = PESTO(max_homology_dim=2, normalize=True)

    X = np.random.random(size=(15000, 100))
    Y = np.random.random(size=(10000, 256))
    N = 10
    print("--------------------------")
    print(f"Embedding 1: {X.shape}")
    print(f"Embedding 2: {Y.shape}")
    print("--------------------------")

    print()
    print(f"Aggregating Landsacpes for {N} Random Projections...")
    score = metric.fit_transform(X, Y, N=N)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Execution time: {:.2f} seconds".format(elapsed_time))
    print("PESTO Score: {:.4f}".format(score))
