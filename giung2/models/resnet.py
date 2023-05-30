import functools
from typing import Any, Tuple, Callable

import jax
import flax
from giung2.models.flax.resnet import FlaxResNetModule


def FlaxResNet(
    image_size: int = 32,
    depth: int = 20,
    widen_factor: int = 1,
    dtype: Any = jax.numpy.float32,
    pixel_mean: Tuple[int] = (0.0, 0.0, 0.0),
    pixel_std: Tuple[int] = (1.0, 1.0, 1.0),
    num_classes: int = None,
    conv: flax.linen.Module = functools.partial(
        flax.linen.Conv,
        use_bias=False,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros),
    norm: flax.linen.Module = functools.partial(
        flax.linen.BatchNorm,
        momentum=0.9,
        epsilon=1e-5,
        use_bias=True,
        use_scale=True,
        scale_init=jax.nn.initializers.ones,
        bias_init=jax.nn.initializers.zeros),
    relu: Callable = flax.linen.relu,
    fc: flax.linen.Module = functools.partial(
        flax.linen.Dense,
        use_bias=True,
        kernel_init=jax.nn.initializers.he_normal(),
        bias_init=jax.nn.initializers.zeros)):

    if depth in [20, 32, 44, 56, 110]:
        num_planes = 16
        block_type = 'Basic'
        num_blocks = [(depth - 2) // 6 for _ in range(3)]

    elif depth in [18, 34, 50, 101, 152]:
        num_planes = 64
        block_type = {
             18: 'Basic',
             34: 'Basic',
             50: 'Bottleneck',
            101: 'Bottleneck',
            152: 'Bottleneck',
        }[depth]
        num_blocks = {
             18: [2, 2,  2, 2],
             34: [3, 4,  6, 3],
             50: [3, 4,  6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]

    else:
        raise NotImplementedError(
            f'Unknown depth={depth} for FlaxResNetModule')

    return FlaxResNetModule(
        image_size, num_planes, block_type, num_blocks, widen_factor,
        dtype, pixel_mean, pixel_std, num_classes, conv, norm, relu, fc)
