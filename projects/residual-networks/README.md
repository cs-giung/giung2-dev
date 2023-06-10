# Residual Networks for Image Classification

## Abstracts
[Deep residual learning for image recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385)
> Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

[Wide residual networks (BMVC 2016)](https://arxiv.org/abs/1605.07146)
> Deep residual networks were shown to be able to scale up to thousands of layers and still have improving performance. However, each fraction of a percent of improved accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of diminishing feature reuse, which makes these networks very slow to train. To tackle these problems, in this paper we conduct a detailed experimental study on the architecture of ResNet blocks, based on which we propose a novel architecture where we decrease depth and increase width of residual networks. We call the resulting network structures wide residual networks (WRNs) and show that these are far superior over their commonly used thin and very deep counterparts. For example, we demonstrate that even a simple 16-layer-deep wide residual network outperforms in accuracy and efficiency all previous deep residual networks, including thousand-layer-deep networks, achieving new state-of-the-art results on CIFAR, SVHN, COCO, and significant improvements on ImageNet. Our code and models are available at this https URL

## Usage examples
Train R20x1 on CIFAR10_x32:
```
python scripts/train.py
    --data_name=CIFAR10_x32 --resnet_depth=20 --resnet_width=1
    --batch_size=256 --optim_ni=48000 --optim_lr=0.1 --optim_momentum=0.9 --optim_weight_decay=0.001
    --seed=42 --mixed_precision=false --save=./save/CIFAR10_x32/R20x1_b256_i48k_lr0.1-wd0.001-s42/
```
Train R50x1 on imagenet2012 (using mixed precision training):
```
export TFDS_DATA_DIR=/path/to/tensorflow_datasets/

python scripts_tfds/train.py
    --data_name=imagenet2012 --resnet_depth=50 --resnet_width=1
    --batch_size=2048 --optim_ni=64000 --optim_lr=0.8 --optim_momentum=0.9 --optim_weight_decay=0.0001
    --seed=42 --mixed_precision=true --save=./save/imagenet2012/R18x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16/
```

## Results for in-built datasets

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

### ImageNet1k_x32
* The optimization terminates after 128k iterations with a mini-batch size of 1024 (~102 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with two GeForce RTX 3090.

| Depth | Width | # Params | val/acc | misc                                |    |
|    -: |    -: |       -: |     :-: | :-                                  | :- |
|    18 |     1 |  11.68 M |   0.546 | `0.81 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_18x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601025851.log)
|    34 |     1 |  21.79 M |   0.573 | `0.44 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_34x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531221315.log)
|    50 |     1 |  25.55 M |   0.613 | `0.30 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x32/resnet_50x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230531184114.log)

### ImageNet1k_x64
* The optimization terminates after 128k iterations with a mini-batch size of 1024 (~102 epochs).
* The calculation of throughputs is based on the last 5k iterations.
* All training runs are done with four GeForce RTX 3090.

| Depth | Width | # Params | val/acc | misc                              |    |
|    -: |    -: |       -: |     :-: | :-                                | :- |
|    18 |     1 |  11.68 M |   0.654 | `0.46 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_18x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230602124852.log)
|    34 |     1 |  21.79 M |   0.681 | `0.25 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_34x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230602163612.log)
|    50 |     1 |  25.55 M |   0.707 | `0.16 epoch/min` `fp16` `nan-grads` | [`*.log`](./save/ImageNet1k_x64/resnet_50x1-batch_1024-iter_128k-lr_0.4-wd_0.0001-fp16/42/20230601214123.log)

## Results for imagenet2012

* The optimization terminates after approximately 102 epochs with varying mini-batch sizes.
* The calculation of throughputs is based on the last 5k iterations.
* The following training runs are done with four GeForce RTX 3090.

| Depth | Width | # Params | IN    | V2    | R     | A     | S     | misc                  |    |
|    -: |    -: |       -: | :-:   | :-:   | :-:   | :-:   | :-:   | :-                    | :- |
|    18 |     1 |  11.69 M | 0.703 | 0.570 | 0.304 | 0.012 | 0.187 | `0.32 epoch/min` `fp16` `b2048-64k`  | [`*.log`](https://www.dropbox.com/s/cgxxwn9imk2xe7h/R18x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16.log) [`*.ckpt`](https://www.dropbox.com/s/2ge8jj6si2bcthn/R18x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16.ckpt)
|    34 |     1 |  21.80 M | 0.738 | 0.609 | 0.346 | 0.019 | 0.229 | `0.27 epoch/min` `fp16` `b2048-64k`  | [`*.log`](https://www.dropbox.com/s/n7nkugzrsll2x2c/R34x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16.log) [`*.ckpt`](https://www.dropbox.com/s/srtj13opp67ff34/R34x1_b2048_i64k_lr0.8-wd0.0001-s42-fp16.ckpt)
|    50 |     1 |  25.56 M | 0.764 | 0.638 | 0.353 | 0.020 | 0.239 | `0.16 epoch/min` `fp16` `b1024-128k` | [`*.log`](https://www.dropbox.com/s/l8cww0f0fqm60q6/R50x1_b1024_i128k_lr0.4-wd0.0001-s42-fp16.log) [`*.ckpt`](https://www.dropbox.com/s/5savumjgfofoh15/R50x1_b1024_i128k_lr0.4-wd0.0001-s42-fp16.ckpt)

* The following training runs are done with eight GeForce RTX 3090.

| Depth | Width | # Params | IN    | V2    | R     | A     | S     | misc                  |
|    -: |    -: |       -: | :-:   | :-:   | :-:   | :-:   | :-:   | :-                    |
|    50 |     1 |  25.56 M | 0.767 | (N/A) | (N/A) | (N/A) | (N/A) | `0.24 epoch/min` `fp16` `b2048-64k`  |
|   152 |     1 |  60.19 M | 0.790 | 0.673 | (N/A) | (N/A) | (N/A) | `0.09 epoch/min` `fp16` `b1024-128k` |

* The following training runs are done with eight TPUv3 cores.

| Depth | Width | # Params | IN    | V2    | R     | A     | S     | misc                  |
|    -: |    -: |       -: | :-:   | :-:   | :-:   | :-:   | :-:   | :-                    |
|    50 |     1 |  25.56 M | 0.765 | 0.641 | (N/A) | (N/A) | (N/A) | `0.27 epoch/min` `fp16` `b4096-32k`  |
|       |       |          | 0.765 | 0.641 | (N/A) | (N/A) | (N/A) | `0.23 epoch/min` `fp16` `b2048-64k`  |
|   152 |     1 |  60.19 M |       |       |       |       |       | `     epoch/min` `fp16` `b2048-32k ` |
|       |       |          |       |       |       |       |       | `     epoch/min` `fp16` `b1024-128k` |
