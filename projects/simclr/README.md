# SimCLR

## Abstracts
[A simple framework for contrastive learning of visual representations (ICML 2020)](https://arxiv.org/abs/2002.05709)
> This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100X fewer labels.

## Usage examples

## Results for imagenet2012

### SimCLRv1
During linear evaluation, which is represented as "Valid ACC" in the tables, we added a linear classifier on the top of the base encoder. To prevent the label information from influencing the encoder, we utilized the `jax.lax.stop_gradient` on the input to the linear classifier. Throughout the training process, both the base encoder and the linear classifier were updated simultaneously using the LARS optimizer, employing the same set of hyperparameters.

| Depth | Width | # Params | Valid ACC | misc |
|    -: |    -: |       -: |        -: | :-   |
|    18 |     1 |  11.69 M |   49.82 % | `b2048-64k` `fp16`  |
|       |       |          |     (TBD) | `b2048-128k` `fp16` |
|       |       |          |     (TBD) | `b2048-512k` `fp16` |
|    50 |     1 |  25.56 M |   63.01 % | `b2048-64k` `fp16`  |
|       |       |          |   64.59 % | `b2048-128k` `fp16` |
|       |       |          |     (TBD) | `b2048-512k` `fp16` |

### SimCLRv0
The results of the initial trials are summarized in the following table. During these trials, we did not use synchronized batch normalization layers and did not exclude batch normalization layers and biases from the LARS optimization.

| Depth | Width | # Params | Valid ACC | misc |
|    -: |    -: |       -: |        -: | :-   |
|    18 |     1 |  11.69 M |   48.93 % | `b2048-64k` `fp16`  |
|       |       |          |   50.82 % | `b2048-128k` `fp16` |
|       |       |          |   52.32 % | `b2048-512k` `fp16` |
|    50 |     1 |  25.56 M |   62.20 % | `b2048-64k` `fp16`  |
|       |       |          |   63.89 % | `b2048-128k` `fp16` |
|       |       |          |     (N/A) | `b2048-512k` `fp16` |
