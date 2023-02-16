import math
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class Identity(nn.Module):
    
    @nn.compact
    def __call__(self, inputs):
        return inputs


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


class DenseBatchEnsemble(nn.Dense):
    use_bias: bool = False
    ensemble_size: int = 1
    use_ensemble_bias: bool = True
    r_base: float = 1.0
    s_base: float = 1.0
    r_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    s_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        r = self.param('batch_ensemble_r', self.r_init, (self.ensemble_size, inputs.shape[-1]), self.param_dtype)
        s = self.param('batch_ensemble_s', self.s_init, (self.ensemble_size,    self.features), self.param_dtype)
        b = self.param('ensemble_bias', self.bias_init, (self.features,), self.param_dtype) if self.use_ensemble_bias else None

        x = jnp.reshape(inputs, (self.ensemble_size, -1) + inputs.shape[1:])
        x, r = nn.dtypes.promote_dtype(x, r, dtype=self.dtype)
        x = jnp.multiply(x, jnp.reshape(r + self.r_base, (self.ensemble_size,) + (1,) * (x.ndim - 2) + (-1,)))
        y = super().__call__(x)
        y, s = nn.dtypes.promote_dtype(y, s, dtype=self.dtype)
        y = jnp.multiply(y, jnp.reshape(s + self.s_base, (self.ensemble_size,) + (1,) * (y.ndim - 2) + (-1,)))
        if b is not None:
            y += jnp.reshape(b, (1,) * (y.ndim - 1) + (-1,))
        return y.reshape((-1,) + y.shape[2:])


class ConvBatchEnsemble(nn.Conv):
    use_bias: bool = False
    ensemble_size: int = 1
    use_ensemble_bias: bool = True
    r_base: float = 1.0
    s_base: float = 1.0
    r_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    s_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        r = self.param('batch_ensemble_r', self.r_init, (self.ensemble_size, inputs.shape[-1]), self.param_dtype)
        s = self.param('batch_ensemble_s', self.s_init, (self.ensemble_size,    self.features), self.param_dtype)
        b = self.param('ensemble_bias', self.bias_init, (self.features,), self.param_dtype) if self.use_ensemble_bias else None

        x = jnp.reshape(inputs, (self.ensemble_size, -1) + inputs.shape[1:])
        x, r = nn.dtypes.promote_dtype(x, r, dtype=self.dtype)
        x = jnp.multiply(x, jnp.reshape(r + self.r_base, (self.ensemble_size,) + (1,) * (x.ndim - 2) + (-1,)))
        y = super().__call__(x)
        y, s = nn.dtypes.promote_dtype(y, s, dtype=self.dtype)
        y = jnp.multiply(y, jnp.reshape(s + self.s_base, (self.ensemble_size,) + (1,) * (y.ndim - 2) + (-1,)))
        if b is not None:
            y += jnp.reshape(b, (1,) * (y.ndim - 1) + (-1,))
        return y.reshape((-1,) + y.shape[2:])


class DenseNormalRankOneBNN(nn.Dense):
    use_bias: bool = False
    ensemble_size: int = 1
    use_ensemble_bias: bool = True
    deterministic: Optional[bool] = None
    rng_collection: str = 'rank_one_bnn'
    r_base: float = 1.0
    s_base: float = 1.0
    r_mean_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    s_mean_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    r_rawstd_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.constant(math.log(math.exp(0.01) - 1.0))
    s_rawstd_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.constant(math.log(math.exp(0.01) - 1.0))

    @nn.compact
    def __call__(self, inputs: Array, deterministic: Optional[bool] = None) -> Array:
        r_mean   = self.param('rank_one_bnn_r_mean',   self.r_mean_init,   (self.ensemble_size, inputs.shape[-1]), self.param_dtype)
        s_mean   = self.param('rank_one_bnn_s_mean',   self.s_mean_init,   (self.ensemble_size,    self.features), self.param_dtype)
        r_rawstd = self.param('rank_one_bnn_r_rawstd', self.r_rawstd_init, (self.ensemble_size, inputs.shape[-1]), self.param_dtype)
        s_rawstd = self.param('rank_one_bnn_s_rawstd', self.s_rawstd_init, (self.ensemble_size,    self.features), self.param_dtype)
        b = self.param('ensemble_bias', self.bias_init, (self.features,), self.param_dtype) if self.use_ensemble_bias else None
        
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        if deterministic:
            r = r_mean
            s = s_mean
        else:
            rngs = jax.random.split(self.make_rng(self.rng_collection))
            r = r_mean + jax.nn.softplus(r_rawstd) * jax.random.normal(rngs[0], r_mean.shape)
            s = s_mean + jax.nn.softplus(s_rawstd) * jax.random.normal(rngs[1], s_mean.shape)

        x = jnp.reshape(inputs, (self.ensemble_size, -1) + inputs.shape[1:])
        x, r = nn.dtypes.promote_dtype(x, r, dtype=self.dtype)
        x = jnp.multiply(x, jnp.reshape(r + self.r_base, (self.ensemble_size,) + (1,) * (x.ndim - 2) + (-1,)))
        y = super().__call__(x)
        y, s = nn.dtypes.promote_dtype(y, s, dtype=self.dtype)
        y = jnp.multiply(y, jnp.reshape(s + self.s_base, (self.ensemble_size,) + (1,) * (y.ndim - 2) + (-1,)))
        if b is not None:
            y += jnp.reshape(b, (1,) * (y.ndim - 1) + (-1,))
        return y.reshape((-1,) + y.shape[2:])


class ConvNormalRankOneBNN(nn.Conv):
    use_bias: bool = False
    ensemble_size: int = 1
    use_ensemble_bias: bool = True
    deterministic: Optional[bool] = None
    rng_collection: str = 'rank_one_bnn'
    r_base: float = 1.0
    s_base: float = 1.0
    r_mean_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    s_mean_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    r_rawstd_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.constant(math.log(math.exp(0.01) - 1.0))
    s_rawstd_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.constant(math.log(math.exp(0.01) - 1.0))

    @nn.compact
    def __call__(self, inputs: Array, deterministic: Optional[bool] = None) -> Array:
        r_mean   = self.param('rank_one_bnn_r_mean',   self.r_mean_init,   (self.ensemble_size, inputs.shape[-1]), self.param_dtype)
        s_mean   = self.param('rank_one_bnn_s_mean',   self.s_mean_init,   (self.ensemble_size,    self.features), self.param_dtype)
        r_rawstd = self.param('rank_one_bnn_r_rawstd', self.r_rawstd_init, (self.ensemble_size, inputs.shape[-1]), self.param_dtype)
        s_rawstd = self.param('rank_one_bnn_s_rawstd', self.s_rawstd_init, (self.ensemble_size,    self.features), self.param_dtype)
        b = self.param('ensemble_bias', self.bias_init, (self.features,), self.param_dtype) if self.use_ensemble_bias else None
        
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        if deterministic:
            r = r_mean
            s = s_mean
        else:
            rngs = jax.random.split(self.make_rng(self.rng_collection))
            r = r_mean + jax.nn.softplus(r_rawstd) * jax.random.normal(rngs[0], r_mean.shape)
            s = s_mean + jax.nn.softplus(s_rawstd) * jax.random.normal(rngs[1], s_mean.shape)

        x = jnp.reshape(inputs, (self.ensemble_size, -1) + inputs.shape[1:])
        x, r = nn.dtypes.promote_dtype(x, r, dtype=self.dtype)
        x = jnp.multiply(x, jnp.reshape(r + self.r_base, (self.ensemble_size,) + (1,) * (x.ndim - 2) + (-1,)))
        y = super().__call__(x)
        y, s = nn.dtypes.promote_dtype(y, s, dtype=self.dtype)
        y = jnp.multiply(y, jnp.reshape(s + self.s_base, (self.ensemble_size,) + (1,) * (y.ndim - 2) + (-1,)))
        if b is not None:
            y += jnp.reshape(b, (1,) * (y.ndim - 1) + (-1,))
        return y.reshape((-1,) + y.shape[2:])


class ConvDropFilter(nn.Conv):
    drop_rate: float = 0.0
    deterministic: Optional[bool] = None
    rng_collection: str = 'dropout'

    @nn.compact
    def __call__(self, inputs: Array, deterministic: Optional[bool] = None) -> Array:
        deterministic = nn.module.merge_param('deterministic', self.deterministic, deterministic)
        if self.drop_rate == 0.0 or deterministic:
            x = inputs
            return super().__call__(x)
        if self.drop_rate == 1.0:
            x = jnp.zeros_like(inputs)
            return super().__call__(x)

        kernel_size = tuple(self.kernel_size)
        num_batch_dimensions = inputs.ndim - (len(kernel_size) + 1)
        broadcast_shape = list(inputs.shape)
        for i in range(len(kernel_size)):
            broadcast_shape[num_batch_dimensions + i] = 1

        keep_prob = 1.0 - self.drop_rate
        rng = self.make_rng(self.rng_collection)
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)

        x = jax.lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))
        return super().__call__(x)
 