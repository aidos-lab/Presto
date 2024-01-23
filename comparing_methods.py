"Compare `Presto` to RTD (https://github.com/IlyaTrofimov/RTD) and CKA (https://github.com/jayroxis/CKA-similarity)"

import sys
import math
import torch
import numpy as np
sys.path.append("../CKA-similarity/")


import rtd
from CKA import CKA,CudaCKA

from sklearn.decomposition import PCA

from presto import Presto


if __name__ == "__main__":
    
    n_components = 2
    n_projections = 1000

    device = "cuda"
    np_cka = CudaCKA(device)
    presto = Presto(n_components=n_components,normalize=True,max_homology_dim=0)
    
    X = torch.randn(1000, 10)
    Y = torch.randn(1000, 10)

    
    print('Linear CKA, between X and Y: {:4f}'.format(np_cka.linear_CKA(X.to(device), Y.to(device))))
    print('Linear CKA, between X and X: {:4f}'.format(np_cka.linear_CKA(X.to(device), X.to(device))))

    print('RBF Kernel CKA, between X and Y: {:4f}'.format(np_cka.kernel_CKA(X.to(device), Y.to(device))))
    print('RBF Kernel CKA, between X and X: {:4f}'.format(np_cka.kernel_CKA(X.to(device), X.to(device))))

    X,Y = X.numpy(), Y.numpy()
    print('RTD, between X and Y: {:4f}'.format(rtd.rtd(X, Y)))
    print('RTD, between X and X: {:4f}'.format(rtd.rtd(X, X)))

    print('`Presto`, between X and Y: {:4f}'.format(presto.fit_transform(X, Y, n_projections)))
    print('`Presto`, between X and X: {:4f}'.format(presto.fit_transform(X, X, n_projections)))
