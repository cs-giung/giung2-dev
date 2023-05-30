import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize


__all__ = [
    'temperature_scaling',
    'get_optimal_temperature',
    'evaluate_acc',
    'evaluate_nll',
]


def temperature_scaling(confidences, temperature, log_input=True, eps=1e-8):
    """
    Args:
        confidences: An array with shape [N, K,].
        temperature: Specifies temperature value for smoothing.
        log_input: Specifies whether confidences are given as log values.
        eps: Small value to avoid evaluation of log(0) when log_input is False.
    Returns:
        An array of temperature-scaled confidences or log-confidences with
        shape [N, K,].
    """
    if log_input:
        # it returns temperature-scaled log_confidences when log_input is True.
        return jax.nn.log_softmax(confidences / temperature, axis=-1)
    else:
        # it returns temperature-scaled confidences when log_input is False.
        return jax.nn.softmax(jnp.log(confidences + eps) / temperature, axis=-1)


def get_optimal_temperature(confidences, true_labels, log_input=True, eps=1e-8):
    """
    Args:
        confidences: An array with shape [N, K,].
        temperature: Specifies temperature value for smoothing.
        log_input: Specifies whether confidences are given as log values.
        eps: Small value to avoid evaluation of log(0) when log_input is False.
    Returns:
        An array of temprature with shape [1,] which minimizes negative
        log-likelihood for given temperature-scaled confidences and true_labels.
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    def obj(t):
        return evaluate_nll(
            temperature_scaling(
                log_confidences, t, log_input=True
            ), true_labels, log_input=True
        )
    optimal_temperature = minimize(
        obj, jnp.asarray([1.0,]), method='BFGS', tol=1e-3).x[0]
    return optimal_temperature


def evaluate_acc(confidences, true_labels,
                 log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences: An array with shape [N, K,].
        temperature: Specifies temperature value for smoothing.
        log_input: Specifies whether confidences are given as log values.
        eps: Small value to avoid evaluation of log(0) when log_input is False.
        reduction: Specifies the reduction to apply to the output.
    Returns:
        An array of accuracy with shape [1,] when reduction in ["mean", "sum",],
        or raw accuracy values with shape [N,] when reduction in ["none",].
    """
    pred_labels = jnp.argmax(confidences, axis=1)
    raw_results = jnp.equal(pred_labels, true_labels)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')


def evaluate_nll(confidences, true_labels,
                 log_input=True, eps=1e-8, reduction="mean"):
    """
    Args:
        confidences: An array with shape [N, K,].
        temperature: Specifies temperature value for smoothing.
        log_input: Specifies whether confidences are given as log values.
        eps: Small value to avoid evaluation of log(0) when log_input is False.
        reduction: Specifies the reduction to apply to the output.
    Returns:
        An array of negative log-likelihood with shape [1,] when reduction in
        ["mean", "sum",], or raw negative log-likelihood values with shape [N,]
        when reduction in ["none",].
    """
    log_confidences = confidences if log_input else jnp.log(confidences + eps)
    true_target = jax.nn.one_hot(
        true_labels, num_classes=log_confidences.shape[1])
    raw_results = -jnp.sum(
        jnp.where(true_target, true_target * log_confidences, 0.0), axis=-1)
    if reduction == "none":
        return raw_results
    elif reduction == "mean":
        return jnp.mean(raw_results)
    elif reduction == "sum":
        return jnp.sum(raw_results)
    else:
        raise NotImplementedError(f'Unknown reduction=\"{reduction}\"')
