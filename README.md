# Presto 🎶

## Confidently and efficiently navigate the multiverse 🧭🚀

The world of machine learning research is riddled with small decisions, from data collection, cleaning, into model selection and parameter tuning. Each combination of data, implementation, and modeling decisions leads to a potential universe where we can analyze and interpret results. Together, these form a multiverse! 🌌

In this project, we focus on mapping out the multiverse of latent representations, specifically targeting machine learning tasks that produce embeddings. In our [ICML 2024 paper](https://arxiv.org/abs/2402.01514), we develop topological tools to efficiently measure the structural variation between representations that arise from different choices in a machine learning workflow. Our main contribution is a custom score developed specifically for multiverse analysis:

`Presto`, our *Pr*ojected *E*mbedding *S*imilarity based on *T*opological *O*verlays. 🔍✨

If you happen to find `Presto` useful, please cite:

```
@inproceedings{wayland2024mapping,
	title={Mapping the multiverse of latent representations},
	author={Wayland, Jeremy and Coupette†, Corinna and Rieck†, Bastian},
	booktitle={International Conference on Machine Learning (ICML)},
	pages={to appear},
	year={2024}
}
```

## Installation

You can install Presto using pip:

`pip install presto-multiverse`

## Basic Usage

### Presto

```python
import numpy as np
from presto import Presto
from sklearn.random_projection import GaussianRandomProjection as Gauss

#Compare Two Embeddings X,Y based on a collection of low-dimensional random embeddings

X = np.random.rand(1000,10)
Y = np.random.rand(1000,12)

presto = Presto(projector=Gauss)

dist = presto.fit_transform(X,Y,n_projections = 20,n_components=2,normalize=True)

print(f"Presto Distance between X & Y : {dist}")
```

### Atom

We also provide an `Atom` class that supports *A*pproximate *T*opological *O*perations in the *M*ultiverse.
When considering a collection of representations that arise from a multiverse, we provide functionality for producing Multiverse Metric Spaces (MMS) that we introduce in our work. Given a collection of embeddings, an MMS encodes the pairwise distance between topological descriptors as computed by `Presto`.

```python
import numpy as np
from presto import Atom

# A list of embeddings in the multiverse
data = [np.random.rand(100, 10) for _ in range(3)]

atom = Atom(data)
atom.compute_MMS(parallelize=True)
print(atom.MMS)
```

### LSD

In order to generate a **latent-space multiverse** we provide a subpacakge called `LSD`: Latent Space Designer. Using the `LSD` class you can catalog embeddings that are generated from Variational Autoencoders, Dimensionality Reduction, and Transformer models and analyze them using `Presto`. Explore your own custom multiverses or recreate our results from our paper!

This is a simple example of how to use the `LSD` class to generate a multiverse of embeddings for `scikit-learn`'s [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) dataset using t-SNE as the dimensionality reduction technique.

```python
import omegaconf
from lsd import LSD

# Define a multiverse config consisting of data, model, and implementation choices
sample_content = """data_choices:
  DimReduction:
    breast_cancer:
      generator:
        - load_breast_cancer
model_choices:
  DimReduction:
    tSNE:
      perplexity:
        - 15
        - 30

implementation_choices:
  DimReduction:
    DimReduction:
      tSNETrainer:
        n_jobs:
          - 1
   """

my_cfg = omegaconf.OmegaConf.create(sample_content)

# Create an instance of the Latent Space Designer
lsd = LSD(multiverseConfig="DimReduction",output_dir="./",experimentName="sample_dr_multiverse")

# Set the multiverse configuration
lsd.multiverse = my_cfg

# Desgin the model configurations based on the multiverse config
lsd.design()

# Generate embeddings from the multiverse
lsd.generate()
```

See our documentation for more examples and detailed usage instructions on using `LSD` and designing multuverse configurations!

## Features

### Normalization & Projection

Presto provides a method to normalize spaces by approximating their diameter. This ensures the embeddings are scaled appropriately before further processing. This class also allows a user to you to project high-dimensional embeddings into a lower-dimensional space using your method of choice! We recommend methods like PCA or Gaussian Random Projection. When using random projections, we encourage the user to produce many random embeddings– `Presto` uses the distrubution of these low-dimensional representations to build topological descriptors of each space.

### Presto Distance

Presto first fits a topological descriptor to each high dimensional embedding: in particular we build persistence landscapes based on the projections of the embedding. When dealing with a distribution of projections, we fit landscapes to each projection and aggregate the topological information into an average landscape, a well defined notion thanks to the great work by [Bubenik et al.](https://arxiv.org/abs/1207.6437) The presto distance between embeddings X,Y is then computed as the landscape distance between the fitted topological descriptors.

### Presto Sensitivity

Presto also offers methods to evaluate the sensitivity of the embeddings' topological structure. By using the statistical properties afforded by persistence landscapes, we compute variance-based scores to evaluate the sensitivity and stability of embeddings with respect to different data, implementation, and modeling choices. `Presto` supports functionality to calculate various versions of sensitivity within a multiverse.

### Clustering and Compression

Using the `Atom` class, you can compute Multiverse Metric Spaces (MMS) that encode pairwise Presto distances between embeddings. Once you have an MMS, there are lots of cool things to do in this space: we provide functionality for clustering your embeddings with Scikit-learn's [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) and compressing your embedding space using a greedy approximation of a set cover algorithm.

### Generating Embeddings

Using the `LSD` class, you can generate embeddings from a variety of models and data choices. This class is designed to help you catalog embeddings that arise from different data, implementation, and modeling choices for a variety of generative modeling tasks. At the moment, we provide support for generating embeddings that arise from Variational Autoencoders, Dimensionality Reduction, and Transformer models– we provide example configuration files in the `lsd.design` module:

```bash
├── design
│   ├── ae.yaml
│   ├── dr.yaml
│   └── tf.yaml
```

See our full documentation for more information on how to design and generate embeddings from your own multiverse!

## License

Presto is licensed under the BSD-3 License. This permissive license allows you to use, modify, and distribute the software with minimal restrictions. You are free to incorporate Presto into your projects, whether they are open-source or proprietary, as long as you cite us! Please include the original license and copyright notice in any distributions of the software. For more detailed information, please refer to the LICENSE file included in the repository.

## Contributing

We welcome contributions and suggestions for our Presto package! Here are some basic guidelines for contributing:

### How to Submit an Issue

1. **Check Existing Issues**: Before submitting a new issue, please check if it has already been reported.

2. **Open a New Issue**: If your issue is new, open a new issue in the repository. Provide a clear and detailed description of the problem, including steps to reproduce the issue if applicable.

3. **Include Relevant Information**: Include any relevant information, such as system details, version numbers, and screenshots, to help us understand and resolve the issue more efficiently.

### How to Contribute

If you're unfamiliar with contributing to open source repositories, here is a basic roadmap:

1. **Fork the Repository**: Start by forking the repository to your own GitHub account.

2. **Clone the Repository**: Clone the forked repository to your local machine.

   ```sh
   git clone https://github.com/your-username/presto.git
   ```

````

3. **Create a Branch**: Create a new branch for your feature or bug fix.

   ```sh
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Implement your changes in the new branch.

5. **Commit Changes**: Commit your changes with a descriptive commit message.

   ```sh
   git commit -m "Description of your changes"
   ```

6. **Push Changes**: Push the changes to your forked repository.

   ```sh
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**: Open a pull request to the main repository with a clear description of your changes and the purpose of the contribution.

### Need Help?

If you need any help or have questions, feel free to reach out to the authors or submit a pull request. We appreciate your contributions and are happy to assist!
````
