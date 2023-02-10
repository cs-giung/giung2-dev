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
    --model_width {1, 4}
    --optim_ne 500
    --optim_lr {3e-6, 1e-6, 3e-7, 1e-7}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --prior_precision {80.0, 40.0, 20.0, 10.0, 5.0, 2.5}
    --seed 42
```
and obtain the following test performance with varying prior precision,

**R20-FRN-Swish on CIFAR-10/100**
| Precision | CIFAR-10      | CIFAR-100     | Logs |
| :-:       | :-:           | :-:           | :-   |
| 2.5       | 84.62 / 0.637 | 49.21 / 2.847 | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-00/42/20230203091638.log), [`C100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.99_pr-00/42/20230203200438.log)
| 5.0       | 84.78 / 0.602 | 49.76 / 1.889 | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-01/42/20230203092801.log), [`C100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.99_pr-01/42/20230203200442.log)
| 10.0      | 84.86 / 0.606 | 52.27 / 2.228 | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203075155.log), [`C100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203194716.log)
| 20.0      | 84.96 / 0.626 | 50.27 / 2.141 | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203090836.log), [`C100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203205925.log)
| 40.0      | 85.15 / 0.639 | 51.31 / 1.885 | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203084421.log), [`C100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203204656.log)
| 80.0      | 85.04 / 0.671 | 46.86 / 2.018 | [`C10`](./save/CIFAR10_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000001_mo-0.97_pr-05/42/20230203042552.log), [`C100`](./save/CIFAR100_x32/R20-FRN-Swish/SGDM/bs-0080_ne-0500_lr-0.0000010_mo-0.70_pr-05/42/20230203190726.log)

**R20x4-FRN-Swish on CIFAR-10/100 (8 TPUv2 cores)**
| Precision | CIFAR-10      | CIFAR-100     | Logs |
| :-:       | :-:           | :-:           | :-   |
| 2.5       | 87.17 / 0.487 | 00.00 / 0.000 | [`C10`](./save/CIFAR10_x32/R20x4-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-00/42/20230205152826.log)
| 5.0       | 87.09 / 0.563 |               | [`C10`](./save/CIFAR10_x32/R20x4-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-01/42/20230205171157.log)
| 10.0      | 87.21 / 0.619 |               | [`C10`](./save/CIFAR10_x32/R20x4-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.90_pr-02/42/20230205084003.log)
| 20.0      | 87.74 / 0.584 |               | [`C10`](./save/CIFAR10_x32/R20x4-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-03/42/20230205001428.log)
| 40.0      | 87.48 / 0.578 |               | [`C10`](./save/CIFAR10_x32/R20x4-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230205015539.log)
| 80.0      | 87.34 / 0.596 |               | [`C10`](./save/CIFAR10_x32/R20x4-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-05/42/20230205033517.log)

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

    # 0000-0050 schedule for SGHMC
    --num_cycles 30 --num_epochs_quiet   0 --num_epochs_noisy  50

    # 0045-0005 schedule for SGHMC with exploration (Zhang et al., 2020)
    --num_cycles 30 --num_epochs_quiet  45 --num_epochs_noisy   5

    # 0050-0000 schedule for SSE (Huang et al., 2017)
    --num_cycles 30 --num_epochs_quiet  50 --num_epochs_noisy   0
```
and obtain the following test performance with varying prior precision and schedule,

**(CIFAR-10) 1500 epochs / 30 cycles**
| Precision | `0000-0050`   | `0045-0005`   | `0050-0000`   | Logs |
| :-:       | :-:           | :-:           | :-:           | :-   |
| 2.5       | 85.93 / 0.456 | 88.05 / 0.370 | 88.28 / 0.354 | [`C10:0000-0050`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000003_mo-0.99_pr-00/42/20230205061339.log), [`C10:0045-0005`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000030_mo-0.97_pr-00/42/20230204172708.log), [`C10:0050-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0050-0000_lr-0.0000030_mo-0.97_pr-00/42/20230206124722.log) |
| 5.0       | 86.21 / 0.427 | 88.43 / 0.360 | 88.78 / 0.341 | [`C10:0000-0050`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000010_mo-0.90_pr-01/42/20230205084120.log), [`C10:0045-0005`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000030_mo-0.90_pr-01/42/20230204164156.log), [`C10:0050-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0050-0000_lr-0.0000030_mo-0.97_pr-01/42/20230206130909.log) |
| 10.0      | 86.06 / 0.428 | 87.51 / 0.377 | 88.17 / 0.361 | [`C10:0000-0050`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000010_mo-0.90_pr-02/42/20230205093834.log), [`C10:0045-0005`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000003_mo-0.99_pr-02/42/20230204081042.log), [`C10:0050-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0050-0000_lr-0.0000003_mo-0.99_pr-02/42/20230205192703.log) |
| 20.0      | 86.46 / 0.425 | 88.77 / 0.352 | 88.73 / 0.344 | [`C10:0000-0050`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000003_mo-0.90_pr-03/42/20230205043014.log), [`C10:0045-0005`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000030_mo-0.70_pr-03/42/20230204152125.log), [`C10:0050-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0050-0000_lr-0.0000030_mo-0.70_pr-03/42/20230206080228.log) |
| 40.0      | 86.03 / 0.439 | 87.91 / 0.378 | 88.15 / 0.356 | [`C10:0000-0050`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000001_mo-0.90_pr-04/42/20230204232015.log), [`C10:0045-0005`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000010_mo-0.90_pr-04/42/20230204113545.log), [`C10:0050-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0050-0000_lr-0.0000003_mo-0.97_pr-04/42/20230205174208.log) |
| 80.0      | 85.27 / 0.464 | 88.56 / 0.369 | 88.86 / 0.338 | [`C10:0000-0050`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0050_lr-0.0000003_mo-0.70_pr-05/42/20230205031644.log), [`C10:0045-0005`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0045-0005_lr-0.0000003_mo-0.90_pr-05/42/20230204062313.log), [`C10:0050-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0050-0000_lr-0.0000003_mo-0.90_pr-05/42/20230205142337.log) |

**(CIFAR-10) 3000 epochs / 30 cycles**
| Precision | `0000-0100`   | `0090-0010`   | `0100-0000`   | Logs |
| :-:       | :-:           | :-:           | :-:           | :-   |
| 2.5       | 86.75 / 0.403 | 87.83 / 0.385 | 88.73 / 0.346 | [`C10:0000-0100`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0100_lr-0.0000010_mo-0.90_pr-00/42/20230205173647.log), [`C10:0090-0010`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0090-0010_lr-0.0000030_mo-0.90_pr-00/42/20230206104045.log), [`C10:0100-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0100-0000_lr-0.0000030_mo-0.97_pr-00/42/20230209203828.log) |
| 5.0       | 86.81 / 0.410 | 87.67 / 0.389 | 88.31 / 0.361 | [`C10:0000-0100`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0100_lr-0.0000001_mo-0.99_pr-01/42/20230205000401.log), [`C10:0090-0010`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0090-0010_lr-0.0000010_mo-0.97_pr-01/42/20230206003459.log), [`C10:0100-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0100-0000_lr-0.0000010_mo-0.97_pr-01/42/20230209000448.log) |
| 10.0      | 86.76 / 0.406 | 88.48 / 0.365 | 88.21 / 0.358 | [`C10:0000-0100`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0100_lr-0.0000001_mo-0.97_pr-02/42/20230204194159.log), [`C10:0090-0010`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0090-0010_lr-0.0000010_mo-0.90_pr-02/42/20230205211510.log), [`C10:0100-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0100-0000_lr-0.0000010_mo-0.90_pr-02/42/20230208202351.log) |
| 20.0      | 86.55 / 0.414 | 88.46 / 0.367 | 88.35 / 0.363 | [`C10:0000-0100`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0100_lr-0.0000001_mo-0.90_pr-03/42/20230204192552.log), [`C10:0090-0010`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0090-0010_lr-0.0000030_mo-0.70_pr-03/42/20230206064051.log), [`C10:0100-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0100-0000_lr-0.0000030_mo-0.70_pr-03/42/20230209120233.log) |
| 40.0      | 86.58 / 0.413 | 87.91 / 0.387 | 88.68 / 0.350 | [`C10:0000-0100`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0100_lr-0.0000003_mo-0.70_pr-04/42/20230205043039.log), [`C10:0090-0010`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0090-0010_lr-0.0000010_mo-0.90_pr-04/42/20230205215032.log), [`C10:0100-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0100-0000_lr-0.0000010_mo-0.70_pr-04/42/20230208163139.log) |
| 80.0      | 86.65 / 0.427 | 88.27 / 0.375 | 88.52 / 0.349 | [`C10:0000-0100`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0100_lr-0.0000001_mo-0.70_pr-05/42/20230204151945.log), [`C10:0090-0010`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0090-0010_lr-0.0000003_mo-0.90_pr-05/42/20230205085239.log), [`C10:0100-0000`](./save/CIFAR10_x32/R20-FRN-Swish/SSE/bs-0080_nc-0030_ne-0100-0000_lr-0.0000001_mo-0.97_pr-05/42/20230207083816.log) |

**(CIFAR-10) 6000 epochs / 30 cycles**
| Precision | `0000-0200`   | `0180-0020`   | `0200-0000`   | Logs |
| :-:       | :-:           | :-:           | :-:           | :-   |
| 2.5       | 87.15 / 0.387 | 87.20 / 0.399 | 00.00 / 0.000 | [`C10:0000-0200`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0200_lr-0.0000003_mo-0.97_pr-00/42/20230207061602.log), [`C10:0180-0020`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0180-0020_lr-0.0000003_mo-0.99_pr-00/42/20230207130749.log) |
| 5.0       | 87.47 / 0.384 | 87.62 / 0.386 |               | [`C10:0000-0200`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0200_lr-0.0000003_mo-0.97_pr-01/42/20230207063727.log), [`C10:0180-0020`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0180-0020_lr-0.0000010_mo-0.90_pr-01/42/20230207221229.log) |
| 10.0      | 87.56 / 0.392 | 87.67 / 0.393 |               | [`C10:0000-0200`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0200_lr-0.0000001_mo-0.97_pr-02/42/20230206173300.log), [`C10:0180-0020`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0180-0020_lr-0.0000030_mo-0.70_pr-02/42/20230208135405.log) |
| 20.0      | 87.14 / 0.390 | 88.01 / 0.397 |               | [`C10:0000-0200`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0200_lr-0.0000001_mo-0.90_pr-03/42/20230206095558.log), [`C10:0180-0020`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0180-0020_lr-0.0000003_mo-0.97_pr-03/42/20230207082959.log) |
| 40.0      | 87.39 / 0.393 | 87.30 / 0.406 |               | [`C10:0000-0200`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0200_lr-0.0000001_mo-0.70_pr-04/42/20230206030759.log), [`C10:0180-0020`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0180-0020_lr-0.0000003_mo-0.90_pr-04/42/20230207060608.log) |
| 80.0      | 87.33 / 0.401 | 88.23 / 0.386 |               | [`C10:0000-0200`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0000-0200_lr-0.0000001_mo-0.70_pr-05/42/20230206030829.log), [`C10:0180-0020`](./save/CIFAR10_x32/R20-FRN-Swish/SGHMC/bs-0080_nc-0030_ne-0180-0020_lr-0.0000003_mo-0.70_pr-05/42/20230207041121.log) |
