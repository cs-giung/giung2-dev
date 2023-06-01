# Residual Networks

## Results for 32x32 images

### CIFAR10_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (~300 epochs).
* All training runs are done with a single GeForce RTX 3090
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                   |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                     | :- |
|    20 |     1 |   0.27 M |   0.923 |   0.924 |   0.278 | `4545 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530211340.log)
|    20 |     2 |   1.08 M |   0.947 |   0.944 |   0.212 | `2439 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530211338.log)
|    20 |     4 |   4.30 M |   0.954 |   0.948 |   0.193 | `1132 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530211348.log)
|    32 |     1 |   0.47 M |   0.932 |   0.931 |   0.289 | `2941 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_32x1-iter_48k-wd_0.0010/42/20230530211352.log)
|    44 |     1 |   0.66 M |   0.936 |   0.933 |   0.297 | `2308 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_44x1-iter_48k-wd_0.0010/42/20230530212503.log)
|    56 |     1 |   0.86 M |   0.937 |   0.934 |   0.304 | `1840 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_56x1-iter_48k-wd_0.0010/42/20230530213027.log)
|    18 |     1 |  11.17 M |   0.952 |   0.948 |   0.199 | `1327 iter/min`        | [`*.log`](./save/CIFAR10_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530225741.log)
|       |       |          |   0.953 |   0.951 |   0.193 | `2273 iter/min` `fp16` | [`*.log`](./save/CIFAR10_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004307.log)

### CIFAR100_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (~300 epochs).
* All training runs are done with a single GeForce RTX 3090.
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                   |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                     | :- |
|    20 |     1 |   0.28 M |   0.663 |   0.662 |   1.258 |                        | [`*.log`](./save/CIFAR100_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530213318.log)
|    20 |     2 |   1.09 M |   0.720 |   0.722 |   1.170 |                        | [`*.log`](./save/CIFAR100_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530214440.log)
|    20 |     4 |   4.32 M |   0.767 |   0.767 |   0.962 |                        | [`*.log`](./save/CIFAR100_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530214656.log)
|    32 |     1 |   0.47 M |   0.679 |   0.690 |   1.299 |                        | [`*.log`](./save/CIFAR100_x32/resnet_32x1-iter_48k-wd_0.0010/42/20230530215534.log)
|    44 |     1 |   0.67 M |   0.691 |   0.692 |   1.380 |                        | [`*.log`](./save/CIFAR100_x32/resnet_44x1-iter_48k-wd_0.0010/42/20230530215810.log)
|    56 |     1 |   0.86 M |   0.700 |   0.703 |   1.364 |                        | [`*.log`](./save/CIFAR100_x32/resnet_56x1-iter_48k-wd_0.0010/42/20230530220433.log)
|    18 |     1 |  11.22 M |   0.772 |   0.768 |   0.961 | `1322 iter/min`        | [`*.log`](./save/CIFAR100_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530225740.log)
|       |       |          |   0.774 |   0.768 |   0.957 | `2256 iter/min` `fp16` | [`*.log`](./save/CIFAR100_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004337.log)

### TinyImageNet200_x32
* The optimization terminates after 48k iterations with a mini-batch size of 256 (~150 epochs).
* All training runs are done with a single GeForce RTX 3090.
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                   |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                     | :- |
|    20 |     1 |   0.28 M |   0.461 |   0.468 |   2.229 |                        | [`*.log`](./save/TinyImageNet200_x32/resnet_20x1-iter_48k-wd_0.0010/42/20230530223310.log)
|    20 |     2 |   1.10 M |   0.511 |   0.511 |   2.137 |                        | [`*.log`](./save/TinyImageNet200_x32/resnet_20x2-iter_48k-wd_0.0010/42/20230530223313.log)
|    20 |     4 |   4.35 M |   0.541 |   0.545 |   2.186 |                        | [`*.log`](./save/TinyImageNet200_x32/resnet_20x4-iter_48k-wd_0.0010/42/20230530223333.log)
|    18 |     1 |  11.27 M |   0.559 |   0.565 |   2.032 | `1293 iter/min`        | [`*.log`](./save/TinyImageNet200_x32/resnet_18x1-iter_48k-wd_0.0010/42/20230530233419.log)
|       |       |          |   0.565 |   0.559 |   2.037 | `2290 iter/min` `fp16` | [`*.log`](./save/TinyImageNet200_x32/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230531004231.log)

### ImageNet1k_x32
* The optimization terminates after 128k iterations with a mini-batch size of 1024 (~102 epochs).
* All training runs are done with two GeForce RTX 3090.
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | misc                               |    |
|    -: |    -: |       -: |     :-: | :-                                 | :- |
|    18 |     1 |  11.68 M |   0.546 | `1017 iter/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_18x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601025851.log)
|    34 |     1 |  21.79 M |   0.573 | ` 547 iter/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_34x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531221315.log)
|    50 |     1 |  25.55 M |   0.613 | ` 375 iter/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_50x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531184114.log)

## Results for 64x64 images

### TinyImageNet200_x64
* The optimization terminates after `48k` iterations with a mini-batch size of 256 (~150 epochs).
* All training runs are done with a single GeForce RTX 3090.
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | tst/acc | tst/nll | misc                   |    |
|    -: |    -: |       -: |     :-: |     :-: |     :-: | :-                     | :- |
|    20 |     1 |   0.28 M |   0.506 |   0.508 |   2.011 | `1471 iter/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_20x1-iter_48k-wd_0.0010/42/20230530214921.log)
|    20 |     2 |   1.10 M |   0.576 |   0.570 |   1.775 | ` 735 iter/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_20x2-iter_48k-wd_0.0010/42/20230530214951.log)
|    20 |     4 |   4.35 M |   0.611 |   0.598 |   1.731 | ` 322 iter/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_20x4-iter_48k-wd_0.0010/42/20230530222243.log)
|    18 |     1 |  11.27 M |   0.648 |   0.647 |   1.635 | ` 385 iter/min`        | [`*.log`](./save/TinyImageNet200_x64/resnet_18x1-iter_48k-wd_0.0010/42/20230530225807.log)
|       |       |          |   0.643 |   0.642 |   1.631 | ` 730 iter/min` `fp16` | [`*.log`](./save/TinyImageNet200_x64/resnet_18x1-iter_48k-wd_0.0010-fp16/42/20230530233139.log)

### ImageNet1k_x64
* The optimization terminates after 128k iterations with a mini-batch size of 1024 (~102 epochs).
* All training runs are done with four GeForce RTX 3090.
* The calculation of throughputs is based on the last 5k iterations.

| Depth | Width | # Params | val/acc | misc                              |    |
|    -: |    -: |       -: |     :-: | :-                                | :- |
|    50 |     1 |  25.55 M |   0.711 | `206 iter/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_50x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531044821.log)

## Results for 224x224 images

### imagenet2012

| Depth | Width | # Params | val/acc | IN    | IN-V2 | misc                  |    |
|    -: |    -: |       -: |     :-: | :-:   |   :-: | :-                    | :- |
|    18 |     1 |  11.69 M |   0.702 |       |       | `198 iter/min` `fp16` `b2048-64k` `4RTX3090`  | [`*.log`](./save/imagenet2012/resnet_18x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-fp16/42/20230601213542.log)
|    34 |     1 |  21.80 M |   0.740 | 0.737 | 0.609 | `137 iter/min` `fp16` `b2048-64k` `4RTX3090`  | [`*.log`](./save/imagenet2012/resnet_34x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-fp16/42/20230531132545.log)
|    50 |     1 |  25.56 M |   0.767 |       |       | `152 iter/min` `fp16` `b2048-64k` `8RTX3090`  | [`*.log`](./save/imagenet2012/resnet_50x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-fp16/42/20230531040107.log)
|       |       |          |   0.767 | 0.765 | 0.641 | `141 iter/min` `fp16` `b2048-64k` `8TPUv3`    | [`*.log`](./save/imagenet2012/resnet_50x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-tpuv3-fp16/42/20230530191949.log)
|       |       |          |   0.764 | 0.763 | 0.639 | ` 84 iter/min` `fp16` `b4096-32k` `8TPUv3`    | [`*.log`](./save/imagenet2012/resnet_50x1-batch_4096-iter_32k-lr_1.6-wd_0.0001-tpuv3-fp16/42/20230531093438.log)
|   101 |     1 |  44.55 M |   0.784 | 0.782 | 0.663 | `118 iter/min` `fp16` `b2048-64k` `8RTX3090`  | [`*.log`](./save/imagenet2012/resnet_101x1-batch_2048-iter_64k-lr_0.8-wd_0.0001-fp16/42/20230531152838.log)
|   152 |     1 |  60.19 M |   0.792 | 0.790 | 0.673 | `112 iter/min` `fp16` `b1024-128k` `8RTX3090` | [`*.log`](./save/imagenet2012/resnet_152x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601024251.log)
