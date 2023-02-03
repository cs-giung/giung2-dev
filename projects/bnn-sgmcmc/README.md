# SG-MCMC for Bayesian Neural Networks

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results
Following [Izmailov et al. (2021)](https://arxiv.org/abs/2104.14421), we use residual networks with a depth of 20 and projection shortcuts [(He et al., 2016)](https://arxiv.org/abs/1512.03385) modified by the followings; 1) replace ReLU [(Fukushima, 1969)](https://ieeexplore.ieee.org/document/4082265) with Swish [(Hendrycks and Gimpel, 2016)](https://arxiv.org/abs/1606.08415), and 2) replace Batch Normalization [(Ioffe and Szegedy, 2015)](https://arxiv.org/abs/1502.03167) with Filter Response Normalization [(Singh and Krishnan, 2020)](https://arxiv.org/abs/1911.09737).

### Test performance with varying prior precision
| Method | Precision | Epoch | CIFAR-10      | CIFAR-100 | Logs |
| :-:    | :-:       | :-:   | :-:           | :-:       | :-   |
| SGDM   | 2.5       | 500   | 84.62 / 0.637 | (TBD)     | [C10](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-00/42/20230203091638.log)
|        | 5.0       | 500   | 84.78 / 0.602 | (TBD)     | [C10](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.97_pr-01/42/20230203092801.log)
|        | 10.0      | 500   | 84.86 / 0.606 | (TBD)     | [C10](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000010_mo-0.97_pr-02/42/20230203075155.log)
|        | 20.0      | 500   | 84.96 / 0.626 | (TBD)     | [C10](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.90_pr-03/42/20230203090836.log)
|        | 40.0      | 500   | 85.15 / 0.639 | (TBD)     | [C10](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000030_mo-0.70_pr-04/42/20230203084421.log)
|        | 80.0      | 500   | 85.04 / 0.671 | (TBD)     | [C10](./save/CIFAR10_x32/R20-FRN-Swish/bs-0080_ne-0500_lr-0.0000001_mo-0.97_pr-05/42/20230203042552.log)

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
