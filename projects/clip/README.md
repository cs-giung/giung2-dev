# CLIP-ViT

## Getting Started
```bash
ln -s ../../giung2
```

### Required Libraries
This project additionally requires the following libraries.
* [Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers)
* [PyTorch and torchvision: an open source machine learning framework.](https://github.com/pytorch/pytorch)

### Datasets
(TBD)

## Results

| Method   | IN            | IN-V2         | IN-R          | IN-A          | IN-Sketch     | Logs |
| :-:      | :-:           | :-:           | :-:           | :-:           | :-:           | :-:  |
| Zeroshot | 68.34 / 1.168 | 61.94 / 1.477 | 77.64 / 0.862 | 49.92 / 1.974 | 48.16 / 2.163 |
| AdamW-LP | 79.82 / 0.711 | 69.97 / 1.152 | 73.32 / 1.057 | 48.99 / 2.074 | 48.41 / 2.074 | [log](./save/clip-vit-base-patch16/AdamW-LP/bs-0256_ne-0010_lr-0.010000_wd-0.0010/0/20230208081404.log)
| AdamW-FT | 81.88 / 0.847 | 72.22 / 1.497 | 71.42 / 1.438 | 44.44 / 3.093 | 49.70 / 3.107 | [log](./save/clip-vit-base-patch16/AdamW-FT/bs-0256_ne-0010_lr-0.000010_wd-0.0001/0/20230207234539.log)

| Method   | IN            | IN-V2         | IN-R          | IN-A          | IN-Sketch     | Logs |
| :-:      | :-:           | :-:           | :-:           | :-:           | :-:           | :-:  |
| AdamW-LP | 79.75 / 0.718 | 69.95 / 1.162 | 71.18 / 1.184 | 47.89 / 2.129 | 46.98 / 2.360 | [log](./save/clip-vit-base-patch16-zero-head/AdamW-LP/bs-0256_ne-0010_lr-0.010000_wd-0.0010/0/20230211012559.log)
| AdamW-FP | 71.14 / 1.496 | 59.29 / 2.062 | 35.78 / 3.544 | 13.40 / 4.600 | 25.07 / 4.854 | [log](./save/clip-vit-base-patch16-zero-head/AdamW-FT/bs-0256_ne-0010_lr-0.000100_wd-0.0010/0/20230211180406.log)
