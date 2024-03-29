# SimCLR

## Abstracts
[A simple framework for contrastive learning of visual representations (ICML 2020)](https://arxiv.org/abs/2002.05709)
> This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100X fewer labels.

## Usage examples
```
python scripts/train.py
    --resnet_depth=50 --resnet_width=1
    --batch_size=2048 --projection_dim=128 --temperature=0.1
    --optim_ni=512000 --optim_lr=0.075 --optim_weight_decay=1e-06
    --seed=42 --mixed_precision=true --save=/path/to/directory
```

## Results for imagenet2012

### SimCLRv1
During linear evaluation, we added a linear classifier on the top of the base encoder. To prevent the label information from influencing the encoder, we utilized the `jax.lax.stop_gradient` on the input to the linear classifier. Throughout the training process, both the base encoder and the linear classifier were updated simultaneously using the LARS optimizer, employing the same set of hyperparameters. All training runs were conducted using a mini-batch size of 2048 and mixed precision training on eight TPUv3 cores.

| Depth | Width | # Params |     64k |    128k |    512k |
|    -: |    -: |       -: |     :-: |     :-: |     :-: |
|    18 |     1 |  11.69 M | 49.82 % | 51.31 % | 53.14 % |
|    34 |     1 |  21.80 M | 52.94 % | 54.78 % | 57.32 % |
|    50 |     1 |  25.56 M | 63.01 % | 64.59 % | 67.14 % |

### SimCLRv0
The results of the initial trials are summarized in the following table. During these trials, we did not use synchronized batch normalization layers and did not exclude batch normalization layers and biases from the LARS optimization. All training runs were conducted using a mini-batch size of 2048 and mixed precision training on eight TPUv3 cores.


| Depth | Width | # Params |     64k |    128k |    512k |
|    -: |    -: |       -: |     :-: |     :-: |     :-: |
|    18 |     1 |  11.69 M | 48.93 % | 50.82 % | 52.32 % |
|    50 |     1 |  25.56 M | 62.20 % | 63.89 % |   (N/A) |
