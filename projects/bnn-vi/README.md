# Variational Inference for Bayesian Neural Networks

## Getting Started
```bash
ln -s ../../giung2
ln -s ../../data
```

## Results

### R20x1-BN-ReLU
| Method                 | Epoch | CIFAR-10      | CIFAR-100     | Logs |
| :-                     | :-:   | :-:           | :-:           | :-   |
| Deterministic          | 500   | 92.82 / 0.275 | 66.59 / 1.332 | [`C10`](/projects/residual-networks/save/CIFAR10_x32/R20-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230203041033.log), [`C100`](/projects/residual-networks/save/CIFAR100_x32/R20-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230203041342.log)
| Dropout                | 500   | 93.01 / 0.240 | 69.08 / 1.177 | [`C10`](./save/CIFAR10_x32/R20x1-BN-ReLU/Dropout/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_drop-0.01_fp32/42/20230216064433.log), [`C100`](./save/CIFAR100_x32/R20x1-BN-ReLU/Dropout/bs-0256_ne-0500_lr-0.30_mo-0.90_wd-0.0001_drop-0.03_fp32/42/20230216091002.log)
| BatchEnsemble (M=4)    | 500   | 92.04 / 0.423 | 66.26 / 1.427 | [`C10`](./save/CIFAR10_x32/R20x1-BN-ReLU/BatchEnsemble/bs-0256_ne-0500_lr-0.30_mo-0.90_wd-0.0001_ens-4_fp32/42/20230215142311.log), [`C100`](./save/CIFAR100_x32/R20x1-BN-ReLU/BatchEnsemble/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_ens-4_fp32/42/20230215182058.log)
| NormalRankOneBNN (M=4) | 500   | 92.74 / 0.293 | 67.16 / 1.379 | [`C10`](./save/CIFAR10_x32/R20x1-BN-ReLU/NormalRankOneBNN/bs-0256_ne-0500_lr-0.30_mo-0.90_wd-0.0003_ens-4_std-0.0100_fp32/42/20230215154532.log), [`C100`](./save/CIFAR100_x32/R20x1-BN-ReLU/NormalRankOneBNN/bs-0256_ne-0500_lr-0.30_mo-0.90_wd-0.0003_ens-4_std-0.0100_fp32/42/20230215233017.log)

### R20x4-BN-ReLU
| Method                 | Epoch | CIFAR-10      | CIFAR-100     | Logs |
| :-                     | :-:   | :-:           | :-:           | :-   |
| Deterministic          | 500   | 95.11 / 0.171 | 77.17 / 1.022 | [`C10`](/projects/residual-networks/save/CIFAR10_x32/R20x4-BN-ReLU/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_fp32/42/20230204011718.log), [`C100`](/projects/residual-networks/save/CIFAR100_x32/R20x4-BN-ReLU/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_fp32/42/20230203221028.log)
| Dropout                | 500   | 95.35 / 0.166 | 78.40 / 0.863 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/Dropout/bs-0256_ne-0500_lr-0.01_mo-0.90_wd-0.0030_drop-0.01_fp32/42/20230216113735.log), [`C100`](./save/CIFAR100_x32/R20x4-BN-ReLU/Dropout/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_drop-0.03_fp32/42/20230216145050.log)
| BatchEnsemble (M=4)    | 500   | 95.36 / 0.165 | 76.78 / 0.946 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/BatchEnsemble/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_ens-4_fp32/42/20230215010039.log), [`C100`](./save/CIFAR100_x32/R20x4-BN-ReLU/BatchEnsemble/bs-0256_ne-0500_lr-0.03_mo-0.90_wd-0.0030_ens-4_fp32/42/20230215181920.log)
| NormalRankOneBNN (M=4) | 500   | 95.53 / 0.159 | 77.95 / 0.933 | [`C10`](./save/CIFAR10_x32/R20x4-BN-ReLU/NormalRankOneBNN/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_ens-4_std-0.0100_fp32/42/20230215085242.log), [`C100`](./save/CIFAR100_x32/R20x4-BN-ReLU/NormalRankOneBNN/bs-0256_ne-0500_lr-0.10_mo-0.90_wd-0.0010_ens-4_std-0.0100_fp32/42/20230216055323.log)

**Dropout**
```
python scripts/Dropout.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --model_depth 20
    --model_width {1, 4}
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --drop_rate {0.1, 0.03, 0.01}
    --seed 42
```

**BatchEnsemble (M=4)**
```
python scripts/BatchEnsemble.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --model_depth 20
    --model_width {1, 4}
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --ensemble_size 4
    --seed 42
```

**NormalRankOneBNN (M=4)**
```
python scripts/NormalRankOneBNN.py
    --data_name {CIFAR10_x32, CIFAR100_x32}
    --model_depth 20
    --model_width {1, 4}
    --optim_lr {0.3, 0.1, 0.03, 0.01}
    --optim_weight_decay {0.003, 0.001, 0.0003, 0.0001}
    --ensemble_size 4
    --seed 42
```
