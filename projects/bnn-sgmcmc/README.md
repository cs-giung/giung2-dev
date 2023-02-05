# SG-MCMC for Bayesian Neural Networks

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results

### Preliminaries

**R20-FRN-Swish.**
Following [Izmailov et al. (2021)](https://arxiv.org/abs/2104.14421), we use residual networks with a depth of 20 and projection shortcuts [(He et al., 2016)](https://arxiv.org/abs/1512.03385) modified by the followings; 1) replace ReLU [(Fukushima, 1969)](https://ieeexplore.ieee.org/document/4082265) with Swish [(Hendrycks and Gimpel, 2016)](https://arxiv.org/abs/1606.08415), 2) replace Batch Normalization [(Ioffe and Szegedy, 2015)](https://arxiv.org/abs/1502.03167) with Filter Response Normalization [(Singh and Krishnan, 2020)](https://arxiv.org/abs/1911.09737), and 3) use bias for convolutional layers even though normalization layers already deal with it.

**CIFAR-10/100.**
We split the official training dataset into 40,960 training examples and 9,040 validation examples throughout the experiments. All images are standardized before being passed to the model, i.e., subtracting the per-channel mean of (0.49, 0.48, 0.44) and dividing the result by the per-channel standard deviation (0.2, 0.2, 0.2). We do not apply any data augmentation strategy unless specified.

### Stochastic Gradient Descent with Momentum (SGDM; [Polyak, 1964](https://www.sciencedirect.com/science/article/abs/pii/0041555364901375))

The classical training of neural network is finding a single Maximum A Posteriori (MAP) solution, and SGDM is one of the most common ways to accomplish this.

We consider the following command line arguments,
```bash
python scripts/SGDM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --optim_lr {3e-6, 1e-6, 3e-7, 1e-7}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --prior_precision {80.0, 40.0, 20.0, 10.0, 5.0, 2.5}
    --seed 42
```
and obtain the following test performance with varying prior precision,
| Method | Epoch | Precision | CIFAR-10      | CIFAR-100     | Logs |
| :-:    | :-:   | :-:       | :-:           | :-:           | :-   |
| SGDM   | 500   | 2.5       | 84.62 / 0.637 | 49.21 / 2.847 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-00/42/20230203091638.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.99_pr-00/42/20230203200438.log)
|        | 500   | 5.0       | 84.78 / 0.602 | 49.76 / 1.889 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-01/42/20230203092801.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.99_pr-01/42/20230203200442.log)
|        | 500   | 10.0      | 84.86 / 0.606 | 52.27 / 2.228 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203075155.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203194716.log)
|        | 500   | 20.0      | 84.96 / 0.626 | 50.27 / 2.141 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203090836.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203205925.log)
|        | 500   | 40.0      | 85.15 / 0.639 | 51.31 / 1.885 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203084421.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203204656.log)
|        | 500   | 80.0      | 85.04 / 0.671 | 46.86 / 2.018 | [`CIFAR-10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000001_mo-0.97_pr-05/42/20230203042552.log), [`CIFAR-100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.70_pr-05/42/20230203190726.log)

### Stochastic Gradient Hamiltonian Monte Carlo (SGHMC; [Chen et al., 2014](https://arxiv.org/abs/1402.4102))

In Bayesian inference, we do not want to optimize the parameters, but instead sample it from the posterior. The well-known approach to accomplish this in practice is SGHMC, which implements Hamiltonian Monte Carlo (HMC; [Duane et al., 1987](https://www.sciencedirect.com/science/article/abs/pii/037026938791197X), [Neal et al., 2011](https://arxiv.org/abs/1206.1901)) using stochastic gradients.

We consider the following command line arguments, where `num_cycles`, `num_epochs_quiet`, and `num_epochs_noisy` specify a cyclical learning rate schedule for the better exploration of the MCMC chain ([Zhang et al., 2020](https://arxiv.org/abs/1902.03932)),
```bash
python scripts/SGHMC.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --optim_lr {3e-6, 1e-6, 3e-7, 1e-7}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --prior_precision {80.0, 40.0, 20.0, 10.0, 5.0, 2.5}
    --seed 42

    # schedule: 0045-0005
    --num_cycles 30 --num_epochs_quiet  45 --num_epochs_noisy   5

    # schedule: 0000-0050
    --num_cycles 30 --num_epochs_quiet   0 --num_epochs_noisy  50

    # schedule: 0090-0010
    --num_cycles 30 --num_epochs_quiet  90 --num_epochs_noisy  10

    # schedule: 0000-0100
    --num_cycles 30 --num_epochs_quiet   0 --num_epochs_noisy 100
```
and obtain the following test performance with varying prior precision,
| Method | Epoch / Cycle | Schedule  | Precision | CIFAR-10      | CIFAR-100     | Logs |
| :-:    | :-:           | :-:       | :-:       | :-:           | :-:           | :-   |
| SGHMC  | 1500 / 30     | 0045-0005 | 2.5       | 88.05 / 0.370 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000030_mo-0.97_pr-00/42/20230204172708.log)
|        |               |           | 5.0       | 88.43 / 0.360 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000030_mo-0.90_pr-01/42/20230204164156.log)
|        |               |           | 10.0      | 87.51 / 0.377 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000003_mo-0.99_pr-02/42/20230204081042.log)
|        |               |           | 20.0      | 88.77 / 0.352 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000030_mo-0.70_pr-03/42/20230204152125.log)
|        |               |           | 40.0      | 87.91 / 0.378 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000010_mo-0.90_pr-04/42/20230204113545.log)
|        |               |           | 80.0      | 88.56 / 0.369 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000003_mo-0.90_pr-05/42/20230204062313.log)
|        |               | 0000-0050 | 2.5       | 85.93 / 0.456 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000003_mo-0.99_pr-00/42/20230205061339.log)
|        |               |           | 5.0       | 86.21 / 0.427 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000010_mo-0.90_pr-01/42/20230205084120.log)
|        |               |           | 10.0      | 86.06 / 0.428 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000010_mo-0.90_pr-02/42/20230205093834.log)
|        |               |           | 20.0      | 86.46 / 0.425 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000003_mo-0.90_pr-03/42/20230205043014.log)
|        |               |           | 40.0      | 86.03 / 0.439 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000001_mo-0.90_pr-04/42/20230204232015.log)
|        |               |           | 80.0      | 85.27 / 0.464 |               | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000003_mo-0.70_pr-05/42/20230205031644.log)
