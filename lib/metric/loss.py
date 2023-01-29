import jax
import jax.numpy as jnp
from jax.nn import softplus, sigmoid
from jax.tree_util import tree_flatten


@jax.jit
def l2_squared(pytree):
    """L2-norm of the parameters in `pytree`."""
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.vdot(x, x) for x in leaves)


@jax.jit
def l1_absolute(pytree):
    """L1-norm of the parameters in `pytree`."""
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.sum(jnp.fabs(x)) for x in leaves)


@jax.jit
def bce(y: jnp.ndarray, logits: jnp.ndarray, mask: jnp.ndarray):
    """Binary cross-entropy loss, averaged,
    vectorized (multi-label classification).
    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Sigmoid function, where
      each element is a float in :math:`(-\infty, \infty)`.
    """
    terms = y * softplus(-logits) + (1 - y) * softplus(logits)
    return jnp.nanmean(terms, where=mask)


@jax.jit
def softmax_logits_bce(y: jnp.ndarray, logits: jnp.ndarray, mask: jnp.ndarray):
    """Categorical cross-entropy, averaged.

    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
      element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Softmax function,
      where each element is a float in :math:`(-\infty, \infty)`.
    """
    terms = y * jax.nn.log_softmax(logits) + (
        1 - y) * jnp.log(1 - jax.nn.softmax(logits))
    return -jnp.nanmean(terms, where=mask)


@jax.jit
def weighted_bce(y: jnp.ndarray, logits: jnp.ndarray, mask: jnp.ndarray,
                 weights: jnp.ndarray):
    """Weighted version of ``bce``."""
    terms = weights * (y * softplus(-logits) + (1 - y) * softplus(logits))
    return jnp.nanmean(terms, where=mask)


@jax.jit
def softmax_logits_weighted_bce(y: jnp.ndarray, logits: jnp.ndarray,
                                mask: jnp.ndarray):
    """Weighted version of ``softmax_logits_bce``."""
    terms = (y * jax.nn.log_softmax(logits) +
             (1 - y) * jnp.log(1 - jax.nn.softmax(logits)))
    weights = y.shape[0] / (y.sum(axis=0) + 1)
    return -jnp.nanmean(weights * terms, where=mask)


@jax.jit
def balanced_focal_bce(y: jnp.ndarray,
                       logits: jnp.ndarray,
                       mask: jnp.ndarray,
                       gamma=2,
                       beta=0.999):
    """
    This loss function employs two concepts:
      - Effective number of sample, to mitigate class imbalance [1].
      - Focal loss, to underweight the easy to classify samples [2].

    The function takes four inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Sigmoid function, where
      each element is a float in :math:`(-\infty, \infty)`.
      - `gamma` is the :math:`\gamma` parameter in [1].
      - `beta` is the :math:`\beta` parameter in [2].

    References:
      [1] _Cui et al._, Class-Balanced Loss Based on Effective Number of Samples.
      [2] _Lin et al., Focal Loss for Dense Object Detection.
    """

    n1 = jnp.sum(y)
    n0 = jnp.size(y) - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-1
    e0 = (1 - beta**n0) / (1 - beta) + 1e-1

    # Focal weighting
    p = sigmoid(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    # Note: softplus(-logits) = -log(sigmoid(logits)) = -log(p)
    # Note: softplut(logits) = -log(1 - sigmoid(logits)) = -log(1-p)
    terms = y * (w1 / e1) * softplus(-logits) + (1 - y) * (
        w0 / e0) * softplus(logits)
    return jnp.nanmean(terms, where=mask)


@jax.jit
def softmax_logits_balanced_focal_bce(y: jnp.ndarray,
                                      logits: jnp.ndarray,
                                      mask: jnp.ndarray,
                                      gamma=2,
                                      beta=0.999):
    """Same as ``balanced_focal_bce``, but with
    applying Softmax activation on the `logits`."""
    #TODO:FIX this for softmax (multinomial case), n1 = np.sum(y, axis=0)
    n1 = jnp.sum(y, axis=0)
    n0 = y.shape[0] - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-5
    e0 = (1 - beta**n0) / (1 - beta) + 1e-5

    # Focal weighting
    p = jax.nn.softmax(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    terms = y * (w1 / e1) * jax.nn.log_softmax(logits) + (1 - y) * (
        w0 / e0) * jnp.log(1 - p)
    return -jnp.nanmean(terms, where=mask)


@jax.jit
def numeric_error(mean_true: jnp.ndarray, mean_predicted: jnp.ndarray,
                  logvar: jnp.ndarray) -> jnp.ndarray:
    """Return the exponent of the normal-distribution liklihood function."""
    sigma = jnp.exp(0.5 * logvar)
    return (mean_true - mean_predicted) / sigma


@jax.jit
def lognormal_loss(mask: jnp.ndarray, error: jnp.ndarray,
                   logvar: jnp.ndarray) -> float:
    """Return the negative log-liklihood, masked and vectorized."""
    log_lik_c = jnp.log(jnp.sqrt(2 * jnp.pi))
    return 0.5 * ((jnp.power(error, 2) + logvar + 2 * log_lik_c) *
                  mask).sum() / (mask.sum() + 1e-10)


def gaussian_KL(mu_1: jnp.ndarray, mu_2: jnp.ndarray, sigma_1: jnp.ndarray,
                sigma_2: float) -> jnp.ndarray:
    """Return the Guassian Kullback-Leibler."""
    return (jnp.log(sigma_2) - jnp.log(sigma_1) +
            (jnp.power(sigma_1, 2) + jnp.power(
                (mu_1 - mu_2), 2)) / (2 * sigma_2**2) - 0.5)


@jax.jit
def compute_KL_loss(mean_true: jnp.ndarray,
                    mask: jnp.ndarray,
                    mean_predicted: jnp.ndarray,
                    logvar_predicted: jnp.ndarray,
                    obs_noise_std: float = 1e-1) -> float:
    """Return the Gaussian Kullback-Leibler divergence, masked."""
    std = jnp.exp(0.5 * logvar_predicted)
    return (gaussian_KL(mu_1=mean_predicted,
                        mu_2=mean_true,
                        sigma_1=std,
                        sigma_2=obs_noise_std) * mask).sum() / (jnp.sum(mask) +
                                                                1e-10)
