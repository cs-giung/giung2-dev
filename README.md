# giung2-dev

## Getting Started

See [`projects/`](./projects/) for some research projects that are built on `giung2`.
* [`projects/residual-networks`](./projects/residual-networks/) contains basics for image classification using the family of residual networks [(He et al., 2016)](https://arxiv.org/abs/1512.03385).
* [`projects/bnn-sgmcmc`](./projects/bnn-sgmcmc/) implements Stochastic Gradient Markov Chain Monte Carlo (SG-MCMC; [Welling and Teh, 2011](https://icml.cc/Conferences/2011/papers/398_icmlpaper.pdf)) methods for Bayesian deep learning.

### Required Libraries
`giung2` requires the following libraries, which are available under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* [JAX: Autograd and XLA, brought together for high-performance machine learning research.](https://github.com/google/jax)
* [Flax: a neural network library and ecosystem for JAX designed for flexibility.](https://github.com/google/flax)
* [Optax: a gradient processing and optimization library for JAX.](https://github.com/deepmind/optax)

### Built-in Datasets
See [`data/README.md`](./data/README.md) for details on the built-in datasets.
