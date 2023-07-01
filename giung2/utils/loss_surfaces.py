import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple


def get_2d_loss_surfaces(
        w1, w2, w3, density: int = 5, margin: float = 0.1):

    loss_surfaces = namedtuple(
        'loss_surfaces', ['plane_fn', 'coordinates', 'grid_xs', 'grid_ys'])
    
    unravel_pytree = None
    if not isinstance(w1, jnp.ndarray):
        w1, unravel_pytree = jax.flatten_util.ravel_pytree(w1)
        w2, unravel_pytree = jax.flatten_util.ravel_pytree(w2)
        w3, unravel_pytree = jax.flatten_util.ravel_pytree(w3)

    u = w2 - w1
    v = w3 - w1
    v = v - jnp.dot(v, u) / jnp.square(jnp.linalg.norm(u)) * u
    u_norm = jnp.linalg.norm(u)
    v_norm = jnp.linalg.norm(v)

    if unravel_pytree:
        plane_fn = lambda x, y: unravel_pytree(
            w1 + x * (u / u_norm) + y * (v / v_norm))
    else:
        plane_fn = lambda x, y: w1 + x * (u / u_norm) + y * (v / v_norm)
        
    coordinates = jnp.array(
        [[0, 0], [u_norm, 0], [jnp.dot(w3 - w1, u) / u_norm, v_norm]])
    
    x_min = float(coordinates[:, 0].min())
    x_max = float(coordinates[:, 0].max())
    x_med = float(jnp.median(coordinates[:, 0]))
    grid_xs = [x_min, x_med, x_max]
    grid_xs = np.concatenate((
        np.array([x_min - margin * (x_max - x_min),]),
        np.linspace(grid_xs[0], grid_xs[1], density)[:-1],
        np.linspace(grid_xs[1], grid_xs[2], density),
        np.array([x_max + margin * (x_max - x_min),])))
    
    y_min = float(coordinates[:, 0].min())
    y_max = float(coordinates[:, 0].max())
    y_med = (y_max + y_min) / 2.0
    grid_ys = [y_min, y_med, y_max]
    grid_ys = np.concatenate((
        np.array([y_min - margin * (y_max - y_min),]),
        np.linspace(grid_ys[0], grid_ys[1], density)[:-1],
        np.linspace(grid_ys[1], grid_ys[2], density),
        np.array([y_max + margin * (y_max - y_min),])))

    return loss_surfaces(plane_fn, coordinates, grid_xs, grid_ys)
