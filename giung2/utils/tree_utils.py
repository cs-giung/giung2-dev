import jax
import jax.numpy as jnp


def rand_like_tree(rng_key, target, minval=0.0, maxval=1.0):
    treedef = jax.tree_util.tree_structure(target)
    keys = jax.tree_util.tree_unflatten(
        treedef, jax.random.split(rng_key, treedef.num_leaves))
    return jax.tree_util.tree_map(
        lambda e, k: jax.random.uniform(
            k, e.shape, e.dtype, minval=minval, maxval=maxval), target, keys)


def randn_like_tree(rng_key, target, mean=0.0, std=1.0):
    treedef = jax.tree_util.tree_structure(target)
    keys = jax.tree_util.tree_unflatten(
        treedef, jax.random.split(rng_key, treedef.num_leaves))
    return jax.tree_util.tree_map(
        lambda e, k: mean + std * jax.random.normal(k, e.shape, e.dtype),
        target, keys)
