# Residual Networks

### CIFAR10_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (i.e., 300 epochs).
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc           |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-             | :- |
|    20 |     1 |   0.27 M |   0.923 |   0.924 |   0.278 |                | [`*.log`](./save/CIFAR10_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530211340.log)
|    20 |     2 |   1.08 M |   0.947 |   0.944 |   0.212 |                | [`*.log`](./save/CIFAR10_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530211338.log)
|    20 |     4 |   4.30 M |   0.954 |   0.948 |   0.193 |                | [`*.log`](./save/CIFAR10_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530211348.log)
|    32 |     1 |   0.47 M |   0.932 |   0.931 |   0.289 |                | [`*.log`](./save/CIFAR10_x32/resnet_32x1-iter_48k-wd_0.0010/42/20230530211352.log)
|    44 |     1 |   0.66 M |   0.936 |   0.933 |   0.297 |                | [`*.log`](./save/CIFAR10_x32/resnet_44x1-iter_48k-wd_0.0010/42/20230530212503.log)
|    56 |     1 |   0.86 M |   0.937 |   0.934 |   0.304 |                | [`*.log`](./save/CIFAR10_x32/resnet_56x1-iter_48k-wd_0.0010/42/20230530213027.log)
|    18 |     1 |  11.17 M |   0.952 |   0.948 |   0.199 | fp32 (0.6 hrs) | [`*.log`](./save/CIFAR10_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530225741.log)
|       |       |          |   0.953 |   0.951 |   0.193 | fp16 (0.4 hrs) | [`*.log`](./save/CIFAR10_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004307.log)

### CIFAR100_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (i.e., 300 epochs).
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc           |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-             | :- |
|    20 |     1 |   0.28 M |   0.663 |   0.662 |   1.258 |                | [`*.log`](./save/CIFAR100_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530213318.log)
|    20 |     2 |   1.09 M |   0.720 |   0.722 |   1.170 |                | [`*.log`](./save/CIFAR100_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530214440.log)
|    20 |     4 |   4.32 M |   0.767 |   0.767 |   0.962 |                | [`*.log`](./save/CIFAR100_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530214656.log)
|    32 |     1 |   0.47 M |   0.679 |   0.690 |   1.299 |                | [`*.log`](./save/CIFAR100_x32/resnet_32x1-iter_48k-wd_0.0010/42/20230530215534.log)
|    44 |     1 |   0.67 M |   0.691 |   0.692 |   1.380 |                | [`*.log`](./save/CIFAR100_x32/resnet_44x1-iter_48k-wd_0.0010/42/20230530215810.log)
|    56 |     1 |   0.86 M |   0.700 |   0.703 |   1.364 |                | [`*.log`](./save/CIFAR100_x32/resnet_56x1-iter_48k-wd_0.0010/42/20230530220433.log)
|    18 |     1 |  11.22 M |   0.772 |   0.768 |   0.961 | fp32 (0.6 hrs) | [`*.log`](./save/CIFAR100_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530225740.log)
|       |       |          |   0.774 |   0.768 |   0.957 | fp16 (0.4 hrs) | [`*.log`](./save/CIFAR100_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004337.log)

### TinyImageNet200_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (i.e., 150 epochs).
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc           |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-             | :- |
|    20 |     1 |   0.28 M |   0.461 |   0.468 |   2.229 |                | [`*.log`](./save/TinyImageNet200_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530223310.log)
|    20 |     2 |   1.10 M |   0.511 |   0.511 |   2.137 |                | [`*.log`](./save/TinyImageNet200_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530223313.log)
|    20 |     4 |   4.35 M |   0.541 |   0.545 |   2.186 |                | [`*.log`](./save/TinyImageNet200_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530223333.log)
|    18 |     1 |  11.27 M |   0.559 |   0.565 |   2.032 | fp32 (0.6 hrs) | [`*.log`](./save/TinyImageNet200_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530233419.log)
|       |       |          |   0.565 |   0.559 |   2.037 | fp16 (0.4 hrs) | [`*.log`](./save/TinyImageNet200_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004231.log)

### TinyImageNet200_x64
* The optimization terminates after 48k iterations with a mini-batch size of 256 (i.e., 150 epochs).
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc           |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-             | :- |
|    20 |     1 |   0.28 M |   0.506 |   0.508 |   2.011 |                | [`*.log`](./save/TinyImageNet200_x64/resnet_20x1-iter_48k-wd_0.0010/42/20230530214921.log)
|    20 |     2 |   1.10 M |   0.576 |   0.570 |   1.775 |                | [`*.log`](./save/TinyImageNet200_x64/resnet_20x2-iter_48k-wd_0.0010/42/20230530214951.log)
|    20 |     4 |   4.35 M |   0.611 |   0.598 |   1.731 |                | [`*.log`](./save/TinyImageNet200_x64/resnet_20x4-iter_48k-wd_0.0010/42/20230530222243.log)
|    18 |     1 |  11.27 M |   0.648 |   0.647 |   1.635 | fp32 (2.1 hrs) | [`*.log`](./save/TinyImageNet200_x64/resnet_18x1-iter_48k-wd_0.0010/42/20230530225807.log)
|       |       |          |   0.643 |   0.642 |   1.631 | fp16 (1.1 hrs) | [`*.log`](./save/TinyImageNet200_x64/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230530233139.log)
