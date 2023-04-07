import os
import math
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from giung2.data import image_processing


__all__ = [
    'build_dataloaders',
]



def _build_dataloader(images, labels, batch_size, rng=None, shuffle=False, transform=None):

    # shuffle the entire dataset, if specified...
    _shuffled = jax.random.permutation(rng, len(images)) if shuffle else jnp.arange(len(images))
    images = images[_shuffled]
    labels = labels[_shuffled]

    # add padding to process the entire dataset...
    marker = np.ones([len(images),], dtype=bool)
    num_batches = math.ceil(len(images) / batch_size)

    padded_images = np.concatenate([
        images, np.zeros([num_batches*batch_size - len(images), *images.shape[1:]], images.dtype)])
    padded_labels = np.concatenate([
        labels, np.zeros([num_batches*batch_size - len(labels), *labels.shape[1:]], labels.dtype)])
    padded_marker = np.concatenate([
        marker, np.zeros([num_batches*batch_size - len(images), *marker.shape[1:]], marker.dtype)])

    # define generator using yield...
    local_device_count = jax.local_device_count()
    batch_indices = jnp.arange(len(padded_images)).reshape((num_batches, batch_size))
    for batch_idx in batch_indices:
        batch = {'images': jnp.array(padded_images[batch_idx]),
                 'labels': jnp.array(padded_labels[batch_idx]),
                 'marker': jnp.array(padded_marker[batch_idx]),}
        if transform is not None:
            if rng is not None:
                _, rng = jax.random.split(rng)
            sub_rng = None if rng is None else jax.random.split(rng, batch['images'].shape[0])
            batch['images'] = transform(sub_rng, batch['images'])
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), batch)
        yield batch


def _set_augmentation(name):

    if name == 'none':
        return jax.jit(jax.vmap(image_processing.ToTensorTransform()))
    
    if name == 'dequantized_none':
        return jax.jit(jax.vmap(image_processing.TransformChain([
            image_processing.RandomDequantizationTransform(),
            image_processing.ToTensorTransform()])))
    
    if name == 'standard':
        return jax.jit(jax.vmap(image_processing.TransformChain([
            image_processing.RandomCropTransform(size=32, padding=4),
            image_processing.RandomHFlipTransform(prob=0.5),
            image_processing.ToTensorTransform()])))
    
    if name == 'dequantized_standard':
        return jax.jit(jax.vmap(image_processing.TransformChain([
            image_processing.RandomDequantizationTransform(),
            image_processing.RandomCropTransform(size=32, padding=4),
            image_processing.RandomHFlipTransform(prob=0.5),
            image_processing.ToTensorTransform()])))


def build_dataloaders(config):
    """
    Args:
        config.data_root (str) : root directory containing datasets (e.g., ./data/).
        config.data_name (str) : name of the dataset (e.g., CIFAR10_x32).
        config.data_augmentation (str) : preset name for the train data augmentation (e.g., standard).
        config.data_proportional (float) : ratio value for the proportional train data (e.g., 100pct).
    
    Return:
        dataloaders (dict) : it contains dataloader, trn_loader, val_loader, and tst_loader.
    """
    trn_images = np.load(os.path.join(config.data_root, f'{config.data_name}/train_images.npy'))
    trn_labels = np.load(os.path.join(config.data_root, f'{config.data_name}/train_labels.npy'))
    tst_images = np.load(os.path.join(config.data_root, f'{config.data_name}/test_images.npy'))
    tst_labels = np.load(os.path.join(config.data_root, f'{config.data_name}/test_labels.npy'))

    if config.data_name == 'Birds200_x32':
        trn_images, val_images = trn_images[: 5120], trn_images[ 5120:] #  5120 /   874 /  5794
        trn_labels, val_labels = trn_labels[: 5120], trn_labels[ 5120:]
        image_shape = (1, 32, 32, 3)
        num_classes = 200

    if config.data_name == 'CIFAR10_x32':
        trn_images, val_images = trn_images[:40960], trn_images[40960:] # 40960 /  9040 / 10000
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        image_shape = (1, 32, 32, 3)
        num_classes = 10

    if config.data_name == 'CIFAR100_x32':
        trn_images, val_images = trn_images[:40960], trn_images[40960:] # 40960 /  9040 / 10000
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        image_shape = (1, 32, 32, 3)
        num_classes = 100

    if config.data_name == 'Dogs120_x32':
        trn_images, val_images = trn_images[:10240], trn_images[10240:] # 10240 /  1760 /  8580
        trn_labels, val_labels = trn_labels[:10240], trn_labels[10240:]
        image_shape = (1, 32, 32, 3)
        num_classes = 120

    if config.data_name == 'Food101_x32':
        trn_images, val_images = trn_images[:61440], trn_images[61440:] # 61440 / 14310 / 25250
        trn_labels, val_labels = trn_labels[:61440], trn_labels[61440:]
        image_shape = (1, 32, 32, 3)
        num_classes = 101

    if config.data_name == 'Pets37_x32':
        trn_images, val_images = trn_images[: 2560], trn_images[ 2560:] #  2560 /  1120 /  3669
        trn_labels, val_labels = trn_labels[: 2560], trn_labels[ 2560:]
        image_shape = (1, 32, 32, 3)
        num_classes = 37

    if config.data_name == 'TinyImageNet200_x32':
        trn_images, val_images = trn_images[:81920], trn_images[81920:] # 81920 / 18080 / 10000
        trn_labels, val_labels = trn_labels[:81920], trn_labels[81920:]
        image_shape = (1, 32, 32, 3)
        num_classes = 200

    if config.data_name == 'TinyImageNet200_x64':
        trn_images, val_images = trn_images[:81920], trn_images[81920:] # 81920 / 18080 / 10000
        trn_labels, val_labels = trn_labels[:81920], trn_labels[81920:]
        image_shape = (1, 64, 64, 3)
        num_classes = 200

    if config.data_name == 'ImageNet1k_x32':
        trn_images, val_images = trn_images, tst_images
        trn_labels, val_labels = trn_labels, tst_labels
        image_shape = (1, 32, 32, 3)
        num_classes = 1000

    if config.data_name == 'ImageNet1k_x64':
        trn_images, val_images = trn_images, tst_images
        trn_labels, val_labels = trn_labels, tst_labels
        image_shape = (1, 64, 64, 3)
        num_classes = 1000

    # proportional train data
    trn_images = trn_images[:int(len(trn_images) * config.data_proportional)]
    trn_labels = trn_labels[:int(len(trn_labels) * config.data_proportional)]

    # transforms
    trn_transform = _set_augmentation(config.data_augmentation)
    val_transform = jax.jit(jax.vmap(image_processing.ToTensorTransform()))

    dataloaders = dict()
    dataloaders['dataloader'] = partial(
        _build_dataloader,
        images     = trn_images,
        labels     = trn_labels,
        batch_size = config.optim_bs,
        shuffle    = True,
        transform  = trn_transform)
    dataloaders['trn_loader'] = partial(
        _build_dataloader,
        images     = trn_images,
        labels     = trn_labels,
        batch_size = config.optim_bs,
        shuffle    = False,
        transform  = val_transform)
    dataloaders['val_loader'] = partial(
        _build_dataloader,
        images     = val_images,
        labels     = val_labels,
        batch_size = config.optim_bs,
        shuffle    = False,
        transform  = val_transform)
    dataloaders['tst_loader'] = partial(
        _build_dataloader,
        images     = tst_images,
        labels     = tst_labels,
        batch_size = config.optim_bs,
        shuffle    = False,
        transform  = val_transform)
    dataloaders['trn_steps_per_epoch'] = math.ceil(len(trn_images) / config.optim_bs)
    dataloaders['val_steps_per_epoch'] = math.ceil(len(val_images) / config.optim_bs)
    dataloaders['tst_steps_per_epoch'] = math.ceil(len(tst_images) / config.optim_bs)
    dataloaders['image_shape'] = image_shape
    dataloaders['num_classes'] = num_classes
    dataloaders['num_data'] = len(trn_images)

    return dataloaders
