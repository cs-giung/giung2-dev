# Residual Networks

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results

### Preliminaries

**Residual Networks.**
We use residual networks with projection shortcuts [(He et al., 2016)](https://arxiv.org/abs/1512.03385). It is worth noting that the use of projection shortcuts may lead to different performance compared to the commonly used identity shortcuts in the literature for CIFAR-10. Our implementation also includes the option to adjust the number of convolutional filters using a widen factor [(Zagoruyko and Komodakis, 2016)](https://arxiv.org/abs/1605.07146).

**CIFAR-10/100.**
We split the official training dataset into 40,960 training examples and 9,040 validation examples throughout the experiments. It is important to mention that using 40,960 train examples may result in different performance compared to the commonly misused 50,000 train examples as reported in the literature for CIFAR-10. Before passing the images to the model, we standardized them by subtracting the per-channel mean of (0.49, 0.48, 0.44) and dividing the result by the per-channel standard deviation of (0.2, 0.2, 0.2). Unless specified otherwise, we apply the standard training data augmentation strategy, which includes random cropping and random horizontal flipping.

### Test performance with varying depth
```bash
python scripts/SGDM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --model_name FlaxResNet
    --model_depth {20, 32, 44, 56}
    --model_width 1
    --optim_ne {200, 500}
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --seed 42
```
| Depth | Width | Epoch | CIFAR-10      | CIFAR-100     | Logs |
| :-:   | :-:   | :-:   | :-:           | :-:           | :-   |
| 20    | 1     | 200   | 91.64 / 0.306 | 65.94 / 1.405 | [`C10`](./save/CIFAR10_x32/R20-BN-ReLU/bs-0256_ne-0200_lr-0.30_mo-0.90_wd-0.0003_fp32/42/20230203010253.log), [`C100`](./save/CIFAR100_x32/R20-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.99_wd-0.0003_fp32/42/20230203021457.log)
|       |       | 500   | 92.82 / 0.275 | 66.50 / 1.335 | [`C10`](./save/CIFAR10_x32/R20-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230203041033.log), [`C100`](./save/CIFAR100_x32/R20-BN-ReLU/bs-0256_ne-0500_lr-0.03_mo-0.97_wd-0.0010_fp32/42/20230203053707.log)
| 32    | 1     | 200   | 92.49 / 0.290 | 68.81 / 1.283 | [`C10`](./save/CIFAR10_x32/R32-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230203075646.log), [`C100`](./save/CIFAR100_x32/R32-BN-ReLU/bs-0256_ne-0200_lr-0.30_mo-0.70_wd-0.0010_fp32/42/20230203071742.log)
|       |       | 500   | 93.22 / 0.276 | 67.77 / 1.392 | [`C10`](./save/CIFAR10_x32/R32-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230203114824.log), [`C100`](./save/CIFAR100_x32/R32-BN-ReLU/bs-0256_ne-0500_lr-0.30_mo-0.70_wd-0.0010_fp32/42/20230203101038.log)
| 44    | 1     | 200   | 93.58 / 0.270 | 68.35 / 1.355 | [`C10`](./save/CIFAR10_x32/R44-BN-ReLU/bs-0256_ne-0200_lr-0.30_mo-0.70_wd-0.0010_fp32/42/20230203165805.log), [`C100`](./save/CIFAR100_x32/R44-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.97_wd-0.0010_fp32/42/20230203193005.log)
|       |       | 500   | 93.55 / 0.298 | 69.64 / 1.285 | [`C10`](./save/CIFAR10_x32/R44-BN-ReLU/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0010_fp32/42/20230204020313.log), [`C100`](./save/CIFAR100_x32/R44-BN-ReLU/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_fp32/42/20230204021223.log)
| 56    | 1     | 200   | 93.40 / 0.267 | 70.07 / 1.265 | [`C10`](./save/CIFAR10_x32/R56-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.90_wd-0.0030_fp32/42/20230204085020.log), [`C100`](./save/CIFAR100_x32/R56-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230204073051.log)
|       |       | 500   | 93.83 / 0.261 | 70.58 / 1.336 | [`C10`](./save/CIFAR10_x32/R56-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230204134931.log), [`C100`](./save/CIFAR100_x32/R56-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230204144755.log)

### Test performance with varying width
```bash
python scripts/SGDM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --model_name FlaxResNet
    --model_depth 20
    --model_width {1, 2, 4}
    --optim_ne {200, 500}
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --seed 42
```
| Depth | Width | Epoch | CIFAR-10      | CIFAR-100     | Logs |
| :-:   | :-:   | :-:   | :-:           | :-:           | :-   |
| 20    | 1     | 200   | 91.64 / 0.306 | 65.94 / 1.405 | [`C10`](./save/CIFAR10_x32/R20-BN-ReLU/bs-0256_ne-0200_lr-0.30_mo-0.90_wd-0.0003_fp32/42/20230203010253.log), [`C100`](./save/CIFAR100_x32/R20-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.99_wd-0.0003_fp32/42/20230203021457.log)
|       |       | 500   | 92.82 / 0.275 | 66.50 / 1.335 | [`C10`](./save/CIFAR10_x32/R20-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230203041033.log), [`C100`](./save/CIFAR100_x32/R20-BN-ReLU/bs-0256_ne-0500_lr-0.03_mo-0.97_wd-0.0010_fp32/42/20230203053707.log)
| 20    | 2     | 200   | 94.11 / 0.206 | 72.01 / 1.110 | [`C10`](./save/CIFAR10_x32/R20x2-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230203015047.log), [`C100`](./save/CIFAR100_x32/R20x2-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230203024338.log)
|       |       | 500   | 94.41 / 0.203 | 72.53 / 1.153 | [`C10`](./save/CIFAR10_x32/R20x2-BN-ReLU/bs-0256_ne-0500_lr-0.30_mo-0.70_wd-0.0010_fp32/42/20230203040358.log), [`C100`](./save/CIFAR100_x32/R20x2-BN-ReLU/bs-0256_ne-0500_lr-0.30_mo-0.70_wd-0.0010_fp32/42/20230203045354.log)
| 20    | 4     | 200   | 95.02 / 0.171 | 77.36 / 0.930 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.90_wd-0.0030_fp32/42/20230203142955.log), [`C100`](./save/CIFAR100_x32/R20x4-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230203133254.log)
|       |       | 500   | 95.29 / 0.168 | 76.29 / 0.991 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/bs-0256_ne-0500_lr-0.01_mo-0.97_wd-0.0030_fp32/42/20230204062900.log), [`C100`](./save/CIFAR100_x32/R20x4-BN-ReLU/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_fp32/42/20230204020715.log)

### Test performance of Wide Residual Networks (WRNs)
```bash
python scripts/SGDM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --model_name FlaxPreResNet
    --model_depth {16, 28, 40}
    --model_width {4, 8, 10}
    --optim_ne 200
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --seed 42
```
| Depth | Width | Epoch | CIFAR-10      | CIFAR-100     | Logs |
| :-:   | :-:   | :-:   | :-:           | :-:           | :-   |
| 40    | 4     | 200   | 95.18 / 0.203 | 76.90 / 0.993 | [`C10`](./save/CIFAR10_x32/WRN40x4-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.90_wd-0.0030_fp32/42/20230408153805.log), [`C100`](./save/CIFAR100_x32/WRN40x4-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230408185317.log)
| 16    | 8     | 200   | 95.18 / 0.163 | 78.62 / 1.065 | [`C10`](./save/CIFAR10_x32/WRN16x8-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230408043410.log), [`C100`](./save/CIFAR100_x32/WRN16x8-BN-ReLU/bs-0256_ne-0200_lr-0.10_mo-0.70_wd-0.0030_fp32/42/20230408091344.log)
| 28    | 10    | 200   | 95.20 / 0.187 | 79.44 / 0.891 | [`C10`](./save/CIFAR10_x32/WRN28x10-BN-ReLU/bs-0256_ne-0200_lr-0.01_mo-0.97_wd-0.0030_fp32/42/20230407080102.log), [`C100`](./save/CIFAR100_x32/WRN28x10-BN-ReLU/bs-0256_ne-0200_lr-0.03_mo-0.97_wd-0.0010_fp32/42/20230407233930.log)
