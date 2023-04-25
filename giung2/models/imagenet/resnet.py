import inspect
import functools
from typing import Any, Tuple, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxResNet(nn.Module):
    depth:        int        = 50
    widen_factor: int        = 1
    dtype:        Any        = jnp.float32
    pixel_mean:   Tuple[int] = (0.0, 0.0, 0.0)
    pixel_std:    Tuple[int] = (1.0, 1.0, 1.0)
    num_classes:  int        = None
    conv:         nn.Module  = functools.partial(nn.Conv, use_bias=False,
                                                 kernel_init=jax.nn.initializers.he_normal(),
                                                 bias_init=jax.nn.initializers.zeros)
    norm:         nn.Module  = functools.partial(nn.BatchNorm, momentum=0.9, epsilon=1e-5, use_bias=True, use_scale=True,
                                                 scale_init=jax.nn.initializers.ones,
                                                 bias_init=jax.nn.initializers.zeros)
    relu:         Callable   = nn.relu
    fc:           nn.Module  = functools.partial(nn.Dense, use_bias=True,
                                                 kernel_init=jax.nn.initializers.he_normal(),
                                                 bias_init=jax.nn.initializers.zeros)

    @nn.compact
    def __call__(self, x, **kwargs):

        # NOTE: it should be False during training, if we use batch normalization...
        use_running_average = kwargs.pop('use_running_average', True)
        if 'use_running_average' in inspect.signature(self.norm).parameters:
            self.norm.keywords['use_running_average'] = use_running_average

        # standardize input images...
        m = self.variable('image_stats', 'm', lambda _: jnp.array(self.pixel_mean, dtype=jnp.float32), (x.shape[-1],))
        s = self.variable('image_stats', 's', lambda _: jnp.array(self.pixel_std , dtype=jnp.float32), (x.shape[-1],))
        x = x - jnp.reshape(m.value, (1, 1, 1, -1))
        x = x / jnp.reshape(s.value, (1, 1, 1, -1))

        # specify block structure and widen factor...
        widen_factor = self.widen_factor
        num_planes   = 64
        num_blocks   = {
             50: [3, 4,  6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[self.depth]

        # define the first layer...
        y = self.conv(
            features    = num_planes,
            kernel_size = (7, 7),
            strides     = (2, 2),
            padding     = (3, 3),
            dtype       = self.dtype,
        )(x)
        y = self.norm(dtype=self.dtype)(y)
        y = self.relu(y)
        y = nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')
        self.sow('intermediates', 'feature.layer0', y)
        
        # define intermediate layers...
        for layer_idx, num_block in enumerate(num_blocks):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            for _stride_idx, _stride in enumerate(_strides, start=1):
                _channel = num_planes * (2 ** layer_idx)
                residual = y

                # Following Zagoruyko and Komodakis (2016) and torchvision implementaion,
                # we do not wide the last convolutional layer for each block.
                y = self.conv(
                    features    = _channel * widen_factor,
                    kernel_size = (1, 1),
                    strides     = (1, 1),
                    padding     = (0, 0),
                    dtype       = self.dtype,
                )(y)
                y = self.norm(dtype=self.dtype)(y)
                y = self.relu(y)
                y = self.conv(
                    features    = _channel * widen_factor,
                    kernel_size = (3, 3),
                    strides     = (_stride, _stride),
                    padding     = (1, 1),
                    dtype       = self.dtype,
                )(y)
                y = self.norm(dtype=self.dtype)(y)
                y = self.relu(y)
                y = self.conv(
                    features    = _channel * 4,
                    kernel_size = (1, 1),
                    strides     = (1, 1),
                    padding     = (0, 0),
                    dtype       = self.dtype,
                )(y)
                y = self.norm(dtype=self.dtype)(y)
                if residual.shape != y.shape:
                    residual = self.conv(
                        features    = _channel * 4,
                        kernel_size = (1, 1),
                        strides     = (_stride, _stride),
                        padding     = (0, 0),
                        dtype       = self.dtype,
                    )(residual)
                    residual = self.norm(dtype=self.dtype)(residual)
                
                if _stride_idx == len(_strides):
                    self.sow('intermediates', f'pre_relu_feature.layer{layer_idx + 1}', y)
                
                y = self.relu(y + residual)
                
                if _stride_idx == len(_strides):
                    self.sow('intermediates', f'feature.layer{layer_idx + 1}', y)
        
        y = jnp.mean(y, axis=(1, 2))
        self.sow('intermediates', 'feature.vector', y)

        # return logits if possible
        if self.num_classes:
            y = self.fc(features=self.num_classes, dtype=self.dtype)(y)
            self.sow('intermediates', 'cls.logit', y)

        return y
