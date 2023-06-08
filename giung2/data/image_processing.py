import jax
import jax.numpy as jnp
from typing import List
from abc import ABCMeta, abstractmethod

from giung2.data import color_conversion


__all__ = [
    'Transform',
    'TransformChain',
    'ToTensorTransform',
    'RandomDequantizationTransform',
    'RandomHFlipTransform',
    'RandomCropTransform',
    'RandomGrayscaleTransform',
    'RandomBrightnessTransform',
    'RandomContrastTransform',
    'RandomSaturationTransform',
    'RandomHueTransform',
    'RandomGaussianBlurTransform',
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


class RandomGrayscaleTransform(Transform):

    def __init__(self, prob=0.5):
        """
        Convert the image to grayscale with the given probability.
        Args:
            prob: probability of the conversion.
        """
        self.prob = prob

    def __call__(self, rng, image):
        is_gray = jax.random.bernoulli(rng, self.prob)
        jmage = (255.0 * color_conversion.rgb_to_grayscale(
            image / 255.0, keep_dims=True)).astype(image.dtype)
        return jnp.where(is_gray, jnp.clip(jmage, 0, 255), image)


class RandomBrightnessTransform(Transform):

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        factor = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        image = (image * factor).astype(image.dtype)
        return jnp.clip(image, 0, 255)


class RandomContrastTransform(Transform):

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, rng, image):
        factor = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        mean = jnp.mean(image, axis=(0, 1), keepdims=True)
        image = (mean + (image - mean) * factor).astype(image.dtype)
        return jnp.clip(image, 0, 255)


class RandomSaturationTransform(Transform):

    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    
    def __call__(self, rng, image):
        factor = jax.random.uniform(
            rng, shape=(1,), minval=self.lower, maxval=self.upper)
        rgb = color_conversion.split_channels(image / 255.0, 2)
        hue, sat, val = color_conversion.rgb_planes_to_hsv_planes(*rgb)
        rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes(
            hue, jnp.clip(sat * factor, 0., 1.), val)
        image = (jnp.stack(rgb_adjusted, axis=2) * 255.0).astype(image.dtype)
        return jnp.clip(image, 0, 255)


class RandomHueTransform(Transform):

    def __init__(self, delta=0.5):
        self.delta = delta

    def __call__(self, rng, image):
        factor = jax.random.uniform(
            rng, shape=(1,), minval=-self.delta, maxval=self.delta)
        rgb = color_conversion.split_channels(image / 255.0, 2)
        hue, sat, val = color_conversion.rgb_planes_to_hsv_planes(*rgb)
        rgb_adjusted = color_conversion.hsv_planes_to_rgb_planes(
            (hue + factor) % 1.0, sat, val)
        image = (jnp.stack(rgb_adjusted, axis=2) * 255.0).astype(image.dtype)
        return jnp.clip(image, 0, 255)


class RandomGaussianBlurTransform(Transform):

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, rng, image):
        sigma = jax.random.uniform(
            rng, shape=(1,), minval=self.sigma[0], maxval=self.sigma[1])
        radius = int(self.kernel_size / 2)
        kernel_size_ = 2 * radius + 1
        x = jnp.arange(-radius, radius + 1).astype(jnp.float32)
        blur_filter = jnp.exp(-x**2 / (2. * sigma**2))
        blur_filter = blur_filter / jnp.sum(blur_filter)
        blur_v = jnp.tile(
            jnp.reshape(blur_filter, [kernel_size_, 1, 1, 1]), [1, 1, 1, 3])
        blur_h = jnp.tile(
            jnp.reshape(blur_filter, [1, kernel_size_, 1, 1]), [1, 1, 1, 3])
        blurred = image[jnp.newaxis, ...]
        blurred = jax.lax.conv_general_dilated(
            blurred, blur_h, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=blurred.shape[3])
        blurred = jax.lax.conv_general_dilated(
            blurred, blur_v, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=blurred.shape[3])
        return jnp.squeeze(blurred, axis=0)
