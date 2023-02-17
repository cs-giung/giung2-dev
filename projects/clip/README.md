# CLIP-ViT

## Getting Started
```bash
ln -s ../../giung2
```

### Required Libraries
This project additionally requires the following libraries.
* [Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers)
* [PyTorch and torchvision: an open source machine learning framework.](https://github.com/pytorch/pytorch)
* [timm: a library containing computer vision models, augmentations, and etc.](https://github.com/rwightman/pytorch-image-models)

### Datasets
For evaluation, the following datasets should be manually downloaded.
* `~/data/imagenet`: ImageNet (IN; Russakovsky et al., 2015), 
* `~/data/imagenetv2-matched-frequency`: ImageNet-V2 (IN-V2; Recht et al., 2019), 
* `~/data/imagenet-r`: ImageNet-R (IN-R; Hendrycks et al., 2021), 
* `~/data/imagenet-a`: ImageNet-A (IN-A; Hendrycks et al., 2021), 
* `~/data/sketch`: ImageNet-Sketch (IN-S; Wang et al., 2019), 

## Results

### CLIP-ViT-B/32

| Method   | IN            | IN-V2         | IN-R          | IN-A          | IN-S          | Logs |
| :-       | :-:           | :-:           | :-:           | :-:           | :-:           | :-:  |
| Zeroshot | 63.35 / 1.398 | 55.91 / 1.764 | 69.34 / 1.201 | 31.60 / 2.942 | 42.23 / 2.507 |
| AdamW-LP | 75.45 / 0.901 | 63.86 / 1.445 | 60.00 / 1.773 | 26.27 / 3.412 | 39.79 / 2.832 | [log](./save/clip-vit-base-patch32/AdamW-LP/bs-0256_ne-0010_lr-0.100000_wd-0.0010/0/20230214211003.log)
| AdamW-FT | 79.47 / 0.765 | 68.24 / 1.301 | 61.95 / 1.777 | 23.69 / 3.875 | 43.54 / 2.867 | [log](./save/clip-vit-base-patch32/AdamW-FT/bs-0256_ne-0010_lr-0.000010_wd-0.0001/0/20230214100736.log) 

### CLIP-ViT-B/16

| Method   | IN            | IN-V2         | IN-R          | IN-A          | IN-S          | Logs |
| :-       | :-:           | :-:           | :-:           | :-:           | :-:           | :-:  |
| Zeroshot | 68.34 / 1.168 | 61.94 / 1.477 | 77.64 / 0.862 | 49.92 / 1.974 | 48.16 / 2.163 | 
| AdamW-LP | 79.77 / 0.720 | 69.64 / 1.169 | 72.74 / 1.098 | 46.89 / 2.175 | 47.73 / 2.315 | [log](./save/clip-vit-base-patch16/AdamW-LP/bs-0256_ne-0010_lr-0.010000_wd-0.0001/0/20230216165726.log)
| AdamW-FT | 83.62 / 0.601 | 73.54 / 1.082 | 70.90 / 1.316 | 42.80 / 2.544 | 50.15 / 2.442 | [log](./save/clip-vit-base-patch16/AdamW-FT/bs-0256_ne-0010_lr-0.000010_wd-0.1000/0/20230216010404.log)

**AdamW-LP**
```
python scripts/AdamW-LP.py
    --clip_name {openai/clip-vit-base-patch32, openai/clip-vit-base-patch16}
    --optim_lr {0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001}
    --optim_weight_decay {0.0000, 0.0001, 0.0010, 0.0100, 0.1000}
```

**AdamW-FT**
```
python scripts/AdamW-FT.py
    --clip_name {openai/clip-vit-base-patch32, openai/clip-vit-base-patch16}
    --optim_lr {0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003, 0.000001}
    --optim_weight_decay {0.0000, 0.0001, 0.0010, 0.0100, 0.1000}
```
