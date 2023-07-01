import jax
import jax.numpy as jnp
from collections import namedtuple


def get_2d_loss_surfaces(
        w1, w2, w3,
        x_points: int = 10,
        y_points: int = 10,
        x_margin: float = 0.1,
        y_margin: float = 0.1,
    ):

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
    x_len = x_max - x_min
    x_min = x_min - x_margin * x_len
    x_max = x_max + x_margin * x_len
    grid_xs = jnp.linspace(x_min, x_max, x_points)
    
    y_min = float(coordinates[:, 0].min())
    y_max = float(coordinates[:, 0].max())
    y_len = y_max - y_min
    y_min = y_min - y_margin * y_len
    y_max = y_max + y_margin * y_len
    grid_ys = jnp.linspace(y_min, y_max, y_points)

    return loss_surfaces(plane_fn, coordinates, grid_xs, grid_ys)
