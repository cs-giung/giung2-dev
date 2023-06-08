import jax
import jax.numpy as jnp
from typing import List
from abc import ABCMeta, abstractmethod


__all__ = [
    'Transform',
    'TransformChain',
    'ToTensorTransform',
    'RandomDequantizationTransform',
    'RandomHFlipTransform',
    'RandomCropTransform',
]


class Transform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, rng, image):
        """
        Apply the transform on an image.
        """


class TransformChain(Transform):

    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, rng, image):
        for _t in self.transforms:
            image = _t(rng, image)
        return image


class ToTensorTransform(Transform):

    def __init__(self):
        """
        Change the range of [0, 255] into [0., 1.].
        """
    
    def __call__(self, rng, image):
        return image / 255.0


class RandomDequantizationTransform(Transform):

    def __init__(self):
        """
        Convert discrete [0, 255] to continuous [-0.5, 255.5].
        """

    def __call__(self, rng, image):
        return image + jax.random.uniform(
            rng, image.shape, minval=-0.5, maxval=0.5)


class RandomHFlipTransform(Transform):

    def __init__(self, prob=0.5):
        """
        Flip the image horizontally with the given probability.
        Args:
            prob: probability of the flip.
        """
        self.prob = prob

    def __call__(self, rng, image):
        is_flip = jax.random.bernoulli(rng, self.prob)
        return jnp.where(is_flip, jnp.flip(image, axis=1), image)


class RandomCropTransform(Transform):

    def __init__(self, size, padding):
        """
        Crop the image at a random location with given size and padding.
        Args:
            size (int): desired output size of the crop.
            padding (int): padding on each border of the image before cropping.
        """
        self.size = size
        self.padding = padding

    def __call__(self, rng, image):
        rngs = jax.random.split(rng, 2)
        pad_width = (
            (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        image = jnp.pad(
            image, pad_width=pad_width, mode='constant', constant_values=0)
        h0 = jax.random.randint(
            rngs[0], shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        w0 = jax.random.randint(
            rngs[1], shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        image = jax.lax.dynamic_slice(
            image, start_indices=(h0, w0, 0),
            slice_sizes=(self.size, self.size, image.shape[2]))
        return image
