# Sharpness-Aware Minimization

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results

| Method   | CIFAR-10      | Logs |
| :-:      | :-:           | :-   |
| SGDM     | 95.35 / 0.167 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/SGDM/bs-0256_ne-0200_lr-0.03_mo-0.90_wd-0.0030_rho-0.0000_fp32/42/20230207084455.log)
| SAM-SGDM | 96.20 / 0.119 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/SAM/bs-0256_ne-0200_lr-0.10_mo-0.90_wd-0.0010_rho-0.3000_fp32/42/20230207023915.log)
| Adam     | 93.53 / 0.316 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/Adam/bs-0256_ne-0200_lr-0.0010_b1-0.9700_b2-0.9990_wd-0.0001_rho-0.0000_fp32/42/20230208152048.log)
| SAM-Adam | 95.49 / 0.140 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/SAM-Adam/bs-0256_ne-0200_lr-0.0003_b1-0.7000_b2-0.9990_wd-0.0010_rho-0.3000_fp32/42/20230208082743.log)
