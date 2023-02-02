from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


class FilterResponseNorm(nn.Module):
    epsilon: float = 1e-6
    dtype:   Any   = jnp.float32

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', jax.nn.initializers.ones,  (x.shape[-1],), self.dtype)
        beta  = self.param('beta',  jax.nn.initializers.zeros, (x.shape[-1],), self.dtype)
        tau   = self.param('tau',   jax.nn.initializers.zeros, (x.shape[-1],), self.dtype)
        nu2   = jnp.mean(jnp.square(x), axis=(1, 2), keepdims=True)
        y     = x * jax.lax.rsqrt(nu2 + self.epsilon)
        y     = gamma.reshape(1, 1, 1, -1) * y + beta.reshape(1, 1, 1, -1)
        z     = jnp.maximum(y, tau.reshape(1, 1, 1, -1))
        return z
