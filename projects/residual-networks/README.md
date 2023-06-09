# Residual Networks

## Results for 32x32 images

### CIFAR10_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (~300 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with a single GeForce RTX 3090

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                    |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                      | :- |
|    20 |     1 |   0.27 M |   0.923 |   0.924 |   0.278 | `28.4 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530211340.log)
|    20 |     2 |   1.08 M |   0.947 |   0.944 |   0.212 | `15.2 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530211338.log)
|    20 |     4 |   4.30 M |   0.954 |   0.948 |   0.193 | ` 7.1 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530211348.log)
|    32 |     1 |   0.47 M |   0.932 |   0.931 |   0.289 | `18.4 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_32x1-iter_48k-wd_0.0010/42/20230530211352.log)
|    44 |     1 |   0.66 M |   0.936 |   0.933 |   0.297 | `14.4 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_44x1-iter_48k-wd_0.0010/42/20230530212503.log)
|    56 |     1 |   0.86 M |   0.937 |   0.934 |   0.304 | `11.5 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_56x1-iter_48k-wd_0.0010/42/20230530213027.log)
|    18 |     1 |  11.17 M |   0.952 |   0.948 |   0.199 | ` 8.3 epoch/min`        | [`*.log`](./save/CIFAR10_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530225741.log)
|       |       |          |   0.953 |   0.951 |   0.193 | `14.2 epoch/min` `fp16` | [`*.log`](./save/CIFAR10_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004307.log)

### CIFAR100_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (~300 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                    |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                      | :- |
|    20 |     1 |   0.28 M |   0.663 |   0.662 |   1.258 |                         | [`*.log`](./save/CIFAR100_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530213318.log)
|    20 |     2 |   1.09 M |   0.720 |   0.722 |   1.170 |                         | [`*.log`](./save/CIFAR100_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530214440.log)
|    20 |     4 |   4.32 M |   0.767 |   0.767 |   0.962 |                         | [`*.log`](./save/CIFAR100_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530214656.log)
|    32 |     1 |   0.47 M |   0.679 |   0.690 |   1.299 |                         | [`*.log`](./save/CIFAR100_x32/resnet_32x1-iter_48k-wd_0.0010/42/20230530215534.log)
|    44 |     1 |   0.67 M |   0.691 |   0.692 |   1.380 |                         | [`*.log`](./save/CIFAR100_x32/resnet_44x1-iter_48k-wd_0.0010/42/20230530215810.log)
|    56 |     1 |   0.86 M |   0.700 |   0.703 |   1.364 |                         | [`*.log`](./save/CIFAR100_x32/resnet_56x1-iter_48k-wd_0.0010/42/20230530220433.log)
|    18 |     1 |  11.22 M |   0.772 |   0.768 |   0.961 | ` 8.3 epoch/min`        | [`*.log`](./save/CIFAR100_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530225740.log)
|       |       |          |   0.774 |   0.768 |   0.957 | `14.1 epoch/min` `fp16` | [`*.log`](./save/CIFAR100_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004337.log)

### TinyImageNet200_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (~150 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                   |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                     | :- |
|    20 |     1 |   0.28 M |   0.461 |   0.468 |   2.229 |                        | [`*.log`](./save/TinyImageNet200_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530223310.log)
|    20 |     2 |   1.10 M |   0.511 |   0.511 |   2.137 |                        | [`*.log`](./save/TinyImageNet200_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530223313.log)
|    20 |     4 |   4.35 M |   0.541 |   0.545 |   2.186 |                        | [`*.log`](./save/TinyImageNet200_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530223333.log)
|    18 |     1 |  11.27 M |   0.559 |   0.565 |   2.032 | `4.0 epoch/min`        | [`*.log`](./save/TinyImageNet200_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530233419.log)
|       |       |          |   0.565 |   0.559 |   2.037 | `7.2 epoch/min` `fp16` | [`*.log`](./save/TinyImageNet200_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004231.log)

### ImageNet1k_x32
* The optimization terminates after 128k iterations with a mini-batch size of 1024 (~102 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with two GeForce RTX 3090.

| Depth | Width | # Params | val/acc | misc                                |    |
|    -: |    -: |       -: |     :-: | :-                                  | :- |
|    18 |     1 |  11.68 M |   0.546 | `0.81 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_18x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601025851.log)
|    34 |     1 |  21.79 M |   0.573 | `0.44 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_34x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531221315.log)
|    50 |     1 |  25.55 M |   0.613 | `0.30 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_50x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531184114.log)

## Results for 64x64 images

### TinyImageNet200_x64
* The optimization terminates after `48k` iterations with a mini-batch size of 256 (~150 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with a single GeForce RTX 3090.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                   |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                     | :- |
|    20 |     1 |   0.28 M |   0.506 |   0.508 |   2.011 | `4.6 epoch/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_20x1-iter_48k-wd_0.0010/42/20230530214921.log)
|    20 |     2 |   1.10 M |   0.576 |   0.570 |   1.775 | `2.3 epoch/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_20x2-iter_48k-wd_0.0010/42/20230530214951.log)
|    20 |     4 |   4.35 M |   0.611 |   0.598 |   1.731 | `1.0 epoch/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_20x4-iter_48k-wd_0.0010/42/20230530222243.log)
|    18 |     1 |  11.27 M |   0.648 |   0.647 |   1.635 | `1.2 epoch/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_18x1-iter_48k-wd_0.0010/42/20230530225807.log)
|       |       |          |   0.643 |   0.642 |   1.631 | `2.3 epoch/min` `fp16` | [`*.log`](./save/TinyImageNet200_x64/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230530233139.log)

### ImageNet1k_x64
* The optimization terminates after 128k iterations with a mini-batch size of 1024 (~102 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with four GeForce RTX 3090.

| Depth | Width | # Params | val/acc | misc                              |    |
|    -: |    -: |       -: |     :-: | :-                                | :- |
|    18 |     1 |  11.68 M |   0.654 | `0.46 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_18x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230602124852.log)
|    34 |     1 |  21.79 M |   0.681 | `0.25 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_34x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230602163612.log)
|    50 |     1 |  25.55 M |   0.707 | `0.16 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_50x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601214123.log)

## Results for 224x224 images

### imagenet2012
* The optimization terminates after approximately 102 epochs with varying mini-batch sizes.
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | IN    | IN-V2 |  IN-R |  IN-A |  IN-S | misc                  |    |
|    -: |    -: |       -: |     :-: | :-:   |   :-: |   :-: |   :-: |   :-: | :-                    | :- |
|    18 |     1 |  11.69 M |   0.705 | 0.703 | 0.570 | 0.304 | 0.012 | 0.187 | `0.32 epoch/min` `fp16` `b2048-64k` `4RTX3090`  | [`*.log`](./save/imagenet2012/R18x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16/20230609151625.log)
|    34 |     1 |  21.80 M |   0.738 | 0.738 | 0.609 | 0.346 | 0.019 | 0.229 | `0.27 epoch/min` `fp16` `b2048-64k` `4RTX3090`  | [`*.log`](./save/imagenet2012/R34x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16/20230609204742.log)
|    50 |     1 |  25.56 M |   0.767 |       |       |       |       |       | `0.24 epoch/min` `fp16` `b2048-64k` `8RTX3090`  | [`*.log`](./save/imagenet2012/resnet_50x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-fp16/42/20230531040107.log)
|       |       |          |   0.767 | 0.765 | 0.641 |       |       |       | `0.23 epoch/min` `fp16` `b2048-64k` `8TPUv3`    | [`*.log`](./save/imagenet2012/resnet_50x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-tpuv3-fp16/42/20230530191949.log)
|       |       |          |   0.766 | 0.765 | 0.641 |       |       |       | `0.27 epoch/min` `fp16` `b4096-32k` `8TPUv3`    | [`*.log`](./save/imagenet2012/resnet_50x1-batch_4096-iter_32k-lr_1.6-wd_0.0001-tpuv3-fp16/42/20230601090151.log)
|   101 |     1 |  44.55 M |   0.784 | 0.782 | 0.663 |       |       |       | `0.19 epoch/min` `fp16` `b2048-64k` `8RTX3090`  | [`*.log`](./save/imagenet2012/resnet_101x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-fp16/42/20230531152838.log)
|   152 |     1 |  60.19 M |   0.792 | 0.790 | 0.673 |       |       |       | `0.09 epoch/min` `fp16` `b1024-128k` `8RTX3090` | [`*.log`](./save/imagenet2012/resnet_152x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601024251.log)
