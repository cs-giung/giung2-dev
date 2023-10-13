import jax
import jax.numpy as jnp


__all__ = [
    'compute_div',
    'compute_agr',
]


def compute_div(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences: An array with shape [M, N, K,].
        log_input: Specifies whether confidences are given as log values.
        eps: Small value to avoid evaluation of log(0) when log_input is False.
        reduction: Specifies the reduction to apply to the output.
    Returns:
        An array of diversity values with shape [1,] when reduction in
        ["mean",], or raw diversity values with shape [M, N,] when reduction in
        ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    mean_confidences = jnp.mean(jnp.exp(log_confidences), axis=0)
    raw_results = jnp.array([
        jnp.sum(jnp.multiply(
            mean_confidences,
            jnp.log(mean_confidences + eps) - log_confidences[idx]
        ), axis=-1) for idx in range(log_confidences.shape[0])]) # [M, N]
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def compute_agr(confidences, log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences: An array with shape [M, N, K,].
        log_input: Specifies whether confidences are given as log values.
        eps: Small value to avoid evaluation of log(0) when log_input is False.
        reduction: Specifies the reduction to apply to the output.
    Returns:
        An array of agreement values with shape [1,] when reduction in
        ["mean",], or raw agreement values with shape [M, N,] when reduction in
        ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    mean_confidences = jnp.mean(jnp.exp(log_confidences), axis=0)
    raw_results = jnp.array([
        jnp.sum(jnp.multiply(
            jax.nn.one_hot(
                jnp.argmax(mean_confidences, axis=-1),
                log_confidences.shape[-1]),
            jax.nn.one_hot(
                jnp.argmax(log_confidences[idx], axis=-1),
                log_confidences.shape[-1]),
        ), axis=-1) for idx in range(log_confidences.shape[0])]) # [M, N]
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
