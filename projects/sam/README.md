# Sharpness-Aware Minimization

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results

| Method   | CIFAR-10      | CIFAR-100     | Logs |
| :-:      | :-:           | :-:           | :-   |
| SGDM     | 95.35 / 0.167 | 00.00 / 0.000 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/SGDM/bs-0256_ne-0200_lr-0.03_mo-0.90_wd-0.0030_rho-0.0000_fp32/42/20230207084455.log)
| SAM-SGDM | 96.20 / 0.119 |               | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/SAM/bs-0256_ne-0200_lr-0.10_mo-0.90_wd-0.0010_rho-0.3000_fp32/42/20230207023915.log)
| Adam     | 93.53 / 0.316 |               | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/Adam/bs-0256_ne-0200_lr-0.0010_b1-0.9700_b2-0.9990_wd-0.0001_rho-0.0000_fp32/42/20230208152048.log)
| SAM-Adam | 95.49 / 0.140 |               | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/SAM-Adam/bs-0256_ne-0200_lr-0.0003_b1-0.7000_b2-0.9990_wd-0.0010_rho-0.3000_fp32/42/20230208082743.log)
| BayesSAM | 96.45 / 0.107 |               | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/BayesSAM/bs-0256_ne-0200_lr-0.30_b1-0.900_b2-0.999_wd-0.0003_eps-1e-1_rho-0.1000_factor-1.0_fp32/42/20230210182542.log)

**SGDM / SAM-SGDM ([Polyak, 1964](https://www.sciencedirect.com/science/article/abs/pii/0041555364901375); [Foret et al., 2021](https://arxiv.org/abs/2010.01412))**
```
python scripts/SAM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_momentum {0.99, 0.97, 0.90, 0.70}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --rho {0.0, 0.3, 0.1, 0.03, 0.01}
    --seed 42
```

**Adam / SAM-Adam ([Kingma and Ba, 2015](https://arxiv.org/abs/1412.6980); [Foret et al., 2021](https://arxiv.org/abs/2010.01412))**
```
python scripts/SAM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --optim_lr {0.003, 0.001, 0.0003, 0.0001}
    --optim_b1 {0.99, 0.97, 0.90, 0.70}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --rho {0.0, 0.3, 0.1, 0.03, 0.01}
    --seed 42
```

**BayesSAM ([Möllenhoff and Khan, 2023](https://arxiv.org/abs/2210.01620))**
```
python scripts/BayesSAM.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --optim_lr {1.0, 0.3, 0.1, 0.03, 0.01}
    --optim_b1 {0.99, 0.97, 0.90, 0.70}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --rho {0.3, 0.1, 0.03, 0.01}
    --seed 42
```
