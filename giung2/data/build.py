import os
import math
import numpy as np
from collections import namedtuple

import jax
import jax.numpy as jnp


__all__ = [
    'load_data',
    'build_dataloader',
]


def load_data(data_root, data_name):

    data = namedtuple('data', [
        'trn_images', 'trn_labels', 'val_images', 'val_labels',
        'tst_images', 'tst_labels', 'image_shape', 'num_classes'])
    
    trn_images = np.load(
        os.path.join(data_root, f'{data_name}/train_images.npy'))
    trn_labels = np.load(
        os.path.join(data_root, f'{data_name}/train_labels.npy'))
    tst_images = np.load(
        os.path.join(data_root, f'{data_name}/test_images.npy'))
    tst_labels = np.load(
        os.path.join(data_root, f'{data_name}/test_labels.npy'))

    if data_name == 'Birds200_x32':
        #  5120 /   874 /  5794
        trn_images, val_images = trn_images[: 5120], trn_images[ 5120:]
        trn_labels, val_labels = trn_labels[: 5120], trn_labels[ 5120:]
        image_shape = (1, 32, 32, 3)
        num_classes = 200

    if data_name == 'CIFAR10_x32':
        # 40960 /  9040 / 10000
        trn_images, val_images = trn_images[:40960], trn_images[40960:]
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        image_shape = (1, 32, 32, 3)
        num_classes = 10

    if data_name == 'CIFAR100_x32':
        # 40960 /  9040 / 10000
        trn_images, val_images = trn_images[:40960], trn_images[40960:]
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        image_shape = (1, 32, 32, 3)
        num_classes = 100

    if data_name == 'Dogs120_x32':
        # 10240 /  1760 /  8580
        trn_images, val_images = trn_images[:10240], trn_images[10240:]
        trn_labels, val_labels = trn_labels[:10240], trn_labels[10240:]
        image_shape = (1, 32, 32, 3)
        num_classes = 120

    if data_name == 'Food101_x32':
        # 61440 / 14310 / 25250
        trn_images, val_images = trn_images[:61440], trn_images[61440:]
        trn_labels, val_labels = trn_labels[:61440], trn_labels[61440:]
        image_shape = (1, 32, 32, 3)
        num_classes = 101

    if data_name == 'Pets37_x32':
        #  2560 /  1120 /  3669
        trn_images, val_images = trn_images[: 2560], trn_images[ 2560:]
        trn_labels, val_labels = trn_labels[: 2560], trn_labels[ 2560:]
        image_shape = (1, 32, 32, 3)
        num_classes = 37

    if data_name == 'TinyImageNet200_x32':
        # 81920 / 18080 / 10000
        trn_images, val_images = trn_images[:81920], trn_images[81920:]
        trn_labels, val_labels = trn_labels[:81920], trn_labels[81920:]
        image_shape = (1, 32, 32, 3)
        num_classes = 200

    if data_name == 'TinyImageNet200_x64':
        # 81920 / 18080 / 10000
        trn_images, val_images = trn_images[:81920], trn_images[81920:]
        trn_labels, val_labels = trn_labels[:81920], trn_labels[81920:]
        image_shape = (1, 64, 64, 3)
        num_classes = 200

    if data_name == 'ImageNet1k_x32':
        trn_images, val_images = trn_images, tst_images
        trn_labels, val_labels = trn_labels, tst_labels
        image_shape = (1, 32, 32, 3)
        num_classes = 1000

    if data_name == 'ImageNet1k_x64':
        trn_images, val_images = trn_images, tst_images
        trn_labels, val_labels = trn_labels, tst_labels
        image_shape = (1, 64, 64, 3)
        num_classes = 1000

    return data(trn_images, trn_labels, val_images, val_labels,
                tst_images, tst_labels, image_shape, num_classes)


def build_dataloader(images, labels, batch_size,
                     rng=None, shuffle=False, transform=None):

    # shuffle the entire dataset, if specified
    if shuffle:
        _shuffled = jax.random.permutation(rng, len(images))
    else:
        _shuffled = jnp.arange(len(images))
    images = images[_shuffled]
    labels = labels[_shuffled]

    # add padding to process the entire dataset
    marker = np.ones([len(images),], dtype=bool)
    num_batches = math.ceil(len(marker) / batch_size)
    padded_images = np.concatenate([
        images, np.zeros([
            num_batches*batch_size - len(images), *images.shape[1:]
        ], images.dtype)])
    padded_labels = np.concatenate([
        labels, np.zeros([
            num_batches*batch_size - len(labels), *labels.shape[1:]
        ], labels.dtype)])
    padded_marker = np.concatenate([
        marker, np.zeros([
            num_batches*batch_size - len(images), *marker.shape[1:]
        ], marker.dtype)])

    # define generator using yield
    batch_indices = jnp.arange(len(padded_images))
    batch_indices = batch_indices.reshape((num_batches, batch_size))
    for batch_idx in batch_indices:
        batch = {'images': jnp.array(padded_images[batch_idx]),
                 'labels': jnp.array(padded_labels[batch_idx]),
                 'marker': jnp.array(padded_marker[batch_idx]),}
        if transform is not None:
            if rng is not None:
                _, rng = jax.random.split(rng)
            sub_rng = None if rng is None else jax.random.split(
                rng, batch['images'].shape[0])
            batch['images'] = transform(sub_rng, batch['images'])
        yield batch
