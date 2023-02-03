# SG-MCMC for Bayesian Neural Networks

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results

**R20-FRN-Swish.**
Following [Izmailov et al. (2021)](https://arxiv.org/abs/2104.14421), we use residual networks with a depth of 20 and projection shortcuts [(He et al., 2016)](https://arxiv.org/abs/1512.03385) modified by the followings; 1) replace ReLU [(Fukushima, 1969)](https://ieeexplore.ieee.org/document/4082265) with Swish [(Hendrycks and Gimpel, 2016)](https://arxiv.org/abs/1606.08415), 2) replace Batch Normalization [(Ioffe and Szegedy, 2015)](https://arxiv.org/abs/1502.03167) with Filter Response Normalization [(Singh and Krishnan, 2020)](https://arxiv.org/abs/1911.09737), and 3) use bias for convolutional layers even though normalization layers already deal with it.

**CIFAR-10/100.**
We split the official training dataset into 40,960 training examples and 9,040 validation examples throughout the experiments. All images are standardized before being passed to the model, i.e., subtracting the per-channel mean of (0.49, 0.48, 0.44) and dividing the result by the per-channel standard deviation (0.2, 0.2, 0.2). We do not apply any data augmentation strategy unless specified.

### Test performance with varying prior precision
| Method | Precision | Epoch | CIFAR-10      | CIFAR-100     | Logs |
| :-:    | :-:       | :-:   | :-:           | :-:           | :-   |
| SGDM   | 2.5       | 500   | 84.62 / 0.637 | 49.21 / 2.847 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-00/42/20230203091638.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.99_pr-00/42/20230203200438.log)
|        | 5.0       | 500   | 84.78 / 0.602 | 49.76 / 1.889 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-01/42/20230203092801.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.99_pr-01/42/20230203200442.log)
|        | 10.0      | 500   | 84.86 / 0.606 | 52.27 / 2.228 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203075155.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203194716.log)
|        | 20.0      | 500   | 84.96 / 0.626 | 50.27 / 2.141 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203090836.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203205925.log)
|        | 40.0      | 500   | 85.15 / 0.639 | 51.31 / 1.885 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203084421.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203204656.log)
|        | 80.0      | 500   | 85.04 / 0.671 | 46.86 / 2.018 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000001_mo-0.97_pr-05/42/20230203042552.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.70_pr-05/42/20230203190726.log)

**SGDM** : We consider the following command line arguments:
```bash
python scripts/SGDM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --optim_lr {3e-6, 1e-6, 3e-7, 1e-7}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --prior_precision {80.0, 40.0, 20.0, 10.0, 5.0, 2.5}
    --seed 42
```

**SGHMC** : We consider the following command line arguments:
```bash
(TBD)
```
