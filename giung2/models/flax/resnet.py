import inspect
import functools
from typing import Any, Tuple, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxResNetModule(nn.Module):
    image_size: int = 32
    num_planes: int = 16
    block_type: str = 'Basic'
    num_blocks: Tuple[int] = (3, 3, 3)
    widen_factor: int = 1
    dtype: Any = jnp.float32
    pixel_mean: Tuple[int] = (0.0, 0.0, 0.0)
    pixel_std: Tuple[int] = (1.0, 1.0, 1.0)
    num_classes: int = None
    conv: nn.Module = functools.partial(
        nn.Conv,
        use_bias=False,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros)
    norm: nn.Module = functools.partial(
        nn.BatchNorm,
        momentum=0.9,
        epsilon=1e-5,
        use_bias=True,
        use_scale=True,
        scale_init=jax.nn.initializers.ones,
        bias_init=jax.nn.initializers.zeros)
    relu: Callable = nn.relu
    fc: nn.Module = functools.partial(
        nn.Dense,
        use_bias=True,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros)

    @nn.compact
    def __call__(self, x, **kwargs):

        use_running_average = kwargs.pop('use_running_average', True)
        if 'use_running_average' in inspect.signature(self.norm).parameters:
            self.norm.keywords['use_running_average'] = use_running_average

        deterministic = kwargs.pop('deterministic', True)
        if 'deterministic' in inspect.signature(self.conv).parameters:
            self.conv.keywords['deterministic'] = deterministic
        if 'deterministic' in inspect.signature(self.fc).parameters:
            self.fc.keywords['deterministic'] = deterministic

        # standardize input images
        m = self.pixel_mean
        m = self.variable(
            'image_stats', 'm',
            lambda _: jnp.array(m, dtype=jnp.float32), (x.shape[-1],))
        x = x - jnp.reshape(m.value, (1, 1, 1, -1))

        s = self.pixel_std
        s = self.variable(
            'image_stats', 's',
            lambda _: jnp.array(s, dtype=jnp.float32), (x.shape[-1],))
        x = x / jnp.reshape(s.value, (1, 1, 1, -1))

        # define the first layer
        if self.image_size in [32, 64]:
            y = self.conv(
                features    = self.num_planes,
                kernel_size = (3, 3),
                strides     = (1, 1),
                padding     = 'SAME',
                dtype       = self.dtype,
            )(x)
            y = self.norm(dtype=self.dtype)(y)
            y = self.relu(y)
        
        elif self.image_size == 224:
            y = self.conv(
                features    = self.num_planes,
                kernel_size = (7, 7),
                strides     = (2, 2),
                padding     = 'SAME',
                dtype       = self.dtype,
            )(x)
            y = self.norm(dtype=self.dtype)(y)
            y = self.relu(y)
            y = nn.max_pool(y, (3, 3), strides=(2, 2), padding='SAME')
        
        else:
            raise NotImplementedError(
                f'Unknown image_size={self.image_size} for FlaxResNet')
        
        self.sow('intermediates', 'feature.layer0', y)
        
        # define intermediate layers
        for layer_idx, num_block in enumerate(self.num_blocks):
            _strides = (1,) if layer_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            
            for _stride_idx, _stride in enumerate(_strides, start=1):
                _channel = self.num_planes * (2 ** layer_idx)
                residual = y

                if self.block_type == 'Basic':
                    y = self.conv(
                        features    = _channel * self.widen_factor,
                        kernel_size = (3, 3),
                        strides     = (_stride, _stride),
                        padding     = 'SAME',
                        dtype       = self.dtype,
                    )(y)
                    y = self.norm(dtype=self.dtype)(y)
                    y = self.relu(y)
                    y = self.conv(
                        features    = _channel * self.widen_factor,
                        kernel_size = (3, 3),
                        strides     = (1, 1),
                        padding     = 'SAME',
                        dtype       = self.dtype,
                    )(y)
                    y = self.norm(dtype=self.dtype,
                                  scale_init=jax.nn.initializers.zeros)(y)

                if self.block_type == 'Bottleneck':
                    y = self.conv(
                        features    = _channel * self.widen_factor,
                        kernel_size = (1, 1),
                        strides     = (1, 1),
                        padding     = 'SAME',
                        dtype       = self.dtype,
                    )(y)
                    y = self.norm(dtype=self.dtype)(y)
                    y = self.relu(y)
                    y = self.conv(
                        features    = _channel * self.widen_factor,
                        kernel_size = (3, 3),
                        strides     = (_stride, _stride),
                        padding     = 'SAME',
                        dtype       = self.dtype,
                    )(y)
                    y = self.norm(dtype=self.dtype)(y)
                    y = self.relu(y)
                    y = self.conv(
                        features    = _channel * 4,
                        kernel_size = (1, 1),
                        strides     = (1, 1),
                        padding     = 'SAME',
                        dtype       = self.dtype,
                    )(y)
                    y = self.norm(dtype=self.dtype,
                                  scale_init=jax.nn.initializers.zeros)(y)
                
                if residual.shape != y.shape:
                    residual = self.conv(
                        features    = y.shape[-1],
                        kernel_size = (1, 1),
                        strides     = (_stride, _stride),
                        padding     = 'SAME',
                        dtype       = self.dtype,
                    )(residual)
                    residual = self.norm(dtype=self.dtype)(residual)

                y = self.relu(y + residual)
                
                if _stride_idx == len(_strides):
                    self.sow(
                        'intermediates', f'feature.layer{layer_idx + 1}', y)
        
        y = jnp.mean(y, axis=(1, 2))
        self.sow('intermediates', 'feature.vector', y)

        if self.num_classes:
            y = self.fc(features=self.num_classes, dtype=self.dtype)(y)
            self.sow('intermediates', 'cls.logit', y)

        return y
