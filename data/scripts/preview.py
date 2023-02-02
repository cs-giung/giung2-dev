import os
import numpy as np
import matplotlib.pyplot as plt


NAMES = np.array([
    'Birds200_x32',
    'CIFAR10_x32',
    'CIFAR100_x32',
    'Dogs120_x32',
    'Food101_x32',
    'Pets37_x32',
    'TinyImageNet200_x32',
    'ImageNet1k_x32',
])

EXISTS = np.array([os.path.exists(e) for e in NAMES])
NAMES = NAMES[EXISTS]

fig, axes = plt.subplots(nrows=sum(EXISTS), ncols=10, figsize=(20, sum(EXISTS)*2))
for row_idx in range(axes.shape[0]):
    images = np.load(os.path.join(NAMES[row_idx], 'test_images.npy'))
    print(NAMES[row_idx], images.shape)
    axes[row_idx][0].set_ylabel(NAMES[row_idx])
    for col_idx in range(axes.shape[1]):
        axes[row_idx][col_idx].imshow(images[col_idx], cmap='gray' if images.shape[3] == 1 else None)
        axes[row_idx][col_idx].set_xlim([0, images.shape[1] - 1])
        axes[row_idx][col_idx].set_xticks([])
        axes[row_idx][col_idx].xaxis.tick_top()
        axes[row_idx][col_idx].set_ylim([images.shape[2] - 1, 0])
        axes[row_idx][col_idx].set_yticks([])

plt.tight_layout()
plt.savefig('preview.png')
