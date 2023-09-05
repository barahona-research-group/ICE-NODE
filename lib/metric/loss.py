import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.tree_util import tree_flatten
import equinox as eqx
import re
from .dtw import SoftDTW


@eqx.filter_jit
def l2_squared(pytree):
    """L2-norm of the parameters in `pytree`."""
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.vdot(x, x) for x in leaves)


@eqx.filter_jit
def l1_absolute(pytree):
    """L1-norm of the parameters in `pytree`."""
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.sum(jnp.fabs(x)) for x in leaves)


@eqx.filter_jit
def bce(y: jnp.ndarray, logits: jnp.ndarray, mask: jnp.ndarray, axis=None):
    """Binary cross-entropy loss, averaged,
    vectorized (multi-label classification).
    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Sigmoid function, where
      each element is a float in :math:`(-\infty, \infty)`.
    """
    terms = y * jnn.softplus(-logits) + (1 - y) * jnn.softplus(logits)
    return jnp.nanmean(terms, where=mask, axis=axis)


@eqx.filter_jit
def softmax_bce(y: jnp.ndarray,
                logits: jnp.ndarray,
                mask: jnp.ndarray,
                axis=None):
    """Categorical cross-entropy, averaged.

    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
      element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Softmax function,
      where each element is a float in :math:`(-\infty, \infty)`.
    """
    terms = y * jnn.log_softmax(logits) + (
        1 - y) * jnp.log(1 - jnn.softmax(logits))
    return -jnp.nanmean(terms, where=mask, axis=axis)


@eqx.filter_jit
def balanced_focal_bce(y: jnp.ndarray,
                       logits: jnp.ndarray,
                       mask: jnp.ndarray,
                       gamma=2,
                       beta=0.999,
                       axis=None):
    """
    This loss function employs two concepts:
      - Effective number of sample, to mitigate class imbalance [1].
      - Focal loss, to underweight the easy to classify samples [2].

    The function takes four inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Sigmoid function, where
      each element is a float in :math:`(-\infty, \infty)`.
      - `gamma` is the :math:`\gamma` parameter in [2].
      - `beta` is the :math:`\beta` parameter in [1].

    References:
      [1] _Cui et al._, Class-Balanced Loss Based on Effective Number \
          of Samples.
      [2] _Lin et al., Focal Loss for Dense Object Detection.
    """

    n1 = jnp.sum(y, axis=0)
    n0 = len(y) - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-1
    e0 = (1 - beta**n0) / (1 - beta) + 1e-1

    # Focal weighting
    p = jnn.sigmoid(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    # Note: softplus(-logits) = -log(sigmoid(logits)) = -log(p)
    # Note: softplut(logits) = -log(1 - sigmoid(logits)) = -log(1-p)
    terms = y * (w1 / e1) * jnn.softplus(-logits) + (1 - y) * (
        w0 / e0) * jnn.softplus(logits)
    return jnp.nanmean(terms, where=mask, axis=axis)


@eqx.filter_jit
def softmax_balanced_focal_bce(y: jnp.ndarray,
                               logits: jnp.ndarray,
                               mask: jnp.ndarray,
                               gamma=2,
                               beta=0.999,
                               axis=None):
    """Same as ``balanced_focal_bce``, but with
    applying Softmax activation on the `logits`."""
    #TODO:FIX this for softmax (multinomial case), n1 = np.sum(y, axis=0)
    n1 = jnp.sum(y, axis=0)
    n0 = y.shape[0] - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-5
    e0 = (1 - beta**n0) / (1 - beta) + 1e-5

    # Focal weighting
    p = jnn.softmax(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    terms = y * (w1 / e1) * jnn.log_softmax(logits) + (1 - y) * (
        w0 / e0) * jnp.log(1 - p)
    return -jnp.nanmean(terms, where=mask, axis=axis)


@eqx.filter_jit
def allpairs_hard_rank(y: jnp.ndarray,
                       logits: jnp.ndarray,
                       mask: jnp.ndarray,
                       axis=None):
    """All-pairs loss for ranking.
    The function takes two inputs:
        - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
        - The predictions `logits`, before applying the Sigmoid function, where
        each element is a float in :math:`(-\infty, \infty)`.
    """
    p = jnn.sigmoid(logits)
    p_diff = jnp.expand_dims(p, axis=1) - jnp.expand_dims(p, axis=0)
    p_diff = jnp.where(p_diff < 0, -p_diff, 0.0)

    y_higher = (jnp.expand_dims(y, axis=1).astype(int) -
                jnp.expand_dims(y, axis=0).astype(int)) > 0

    mask = jnp.expand_dims(mask, axis=1) * jnp.expand_dims(mask, axis=0)
    return jnp.nanmean(p_diff, where=(mask & y_higher), axis=axis)


@eqx.filter_jit
def allpairs_exp_rank(y: jnp.ndarray,
                      logits: jnp.ndarray,
                      mask: jnp.ndarray,
                      axis=None):
    """All-pairs loss for ranking, using sigmoid activation.
    The function takes two inputs:
        - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
        - The predictions `logits`, before applying the Sigmoid function, where
        each element is a float in :math:`(-\infty, \infty)`.
    """
    p = jnn.sigmoid(logits)
    p_diff = (jnp.expand_dims(p, axis=1) - jnp.expand_dims(p, axis=0))

    y_diff = (jnp.expand_dims(y, axis=1).astype(int) -
              jnp.expand_dims(y, axis=0).astype(int))
    mask = jnp.expand_dims(mask, axis=1) * jnp.expand_dims(mask, axis=0)

    loss_array = jnp.exp(-8 * y_diff * p_diff)
    return jnp.nanmean(loss_array,
                       where=jnp.triu(mask & (y_diff != 0)),
                       axis=axis)


@eqx.filter_jit
def allpairs_sigmoid_rank(y: jnp.ndarray,
                          logits: jnp.ndarray,
                          mask: jnp.ndarray,
                          axis=None):
    """All-pairs loss for ranking, using sigmoid activation.
    The function takes two inputs:
        - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
        - The predictions `logits`, before applying the Sigmoid function, where
        each element is a float in :math:`(-\infty, \infty)`.
    """
    p = jnn.sigmoid(logits)
    p_diff = (jnp.expand_dims(p, axis=1) - jnp.expand_dims(p, axis=0))

    y_diff = (jnp.expand_dims(y, axis=1).astype(int) -
              jnp.expand_dims(y, axis=0).astype(int))
    mask = jnp.expand_dims(mask, axis=1) * jnp.expand_dims(mask, axis=0)

    loss_array = jnn.sigmoid(-8 * y_diff * p_diff)
    return jnp.nanmean(loss_array, where=(mask & (y_diff != 0)), axis=axis)


@eqx.filter_jit
def masked_mse(y: jnp.ndarray,
               y_hat: jnp.ndarray,
               mask: jnp.ndarray,
               axis=None):
    """Masked L2 loss."""
    return jnp.nanmean(jnp.power(y - y_hat, 2), where=mask, axis=axis)


@eqx.filter_jit
def masked_mae(y: jnp.ndarray,
               y_hat: jnp.ndarray,
               mask: jnp.ndarray,
               axis=None):
    """Masked L1 loss."""
    return jnp.nanmean(jnp.abs(y - y_hat), where=mask, axis=axis)


@eqx.filter_jit
def masked_rms(y: jnp.ndarray,
               y_hat: jnp.ndarray,
               mask: jnp.ndarray,
               axis=None):
    """Masked root mean squared error."""
    return jnp.sqrt(masked_mse(y, y_hat, mask, axis=axis))


# @eqx.filter_jit
# def numeric_error(mean_true: jnp.ndarray, mean_predicted: jnp.ndarray,
#                   logvar: jnp.ndarray) -> jnp.ndarray:
#     """Return the exponent of the normal-distribution liklihood function."""
#     sigma = jnp.exp(0.5 * logvar)
#     return (mean_true - mean_predicted) / sigma

# @eqx.filter_jit
# def lognormal_loss(mask: jnp.ndarray, error: jnp.ndarray,
#                    logvar: jnp.ndarray) -> float:
#     """Return the negative log-liklihood, masked and vectorized."""
#     log_lik_c = jnp.log(jnp.sqrt(2 * jnp.pi))
#     return 0.5 * ((jnp.power(error, 2) + logvar + 2 * log_lik_c) *
#                   mask).sum() / (mask.sum() + 1e-10)

# def gaussian_KL(mu_1: jnp.ndarray, mu_2: jnp.ndarray, sigma_1: jnp.ndarray,
#                 sigma_2: float) -> jnp.ndarray:
#     """Return the Guassian Kullback-Leibler."""
#     return (jnp.log(sigma_2) - jnp.log(sigma_1) +
#             (jnp.power(sigma_1, 2) + jnp.power(
#                 (mu_1 - mu_2), 2)) / (2 * sigma_2**2) - 0.5)

# @eqx.filter_jit
# def compute_KL_loss(mean_true: jnp.ndarray,
#                     mask: jnp.ndarray,
#                     mean_predicted: jnp.ndarray,
#                     logvar_predicted: jnp.ndarray,
#                     obs_noise_std: float = 1e-1) -> float:
#     """Return the Gaussian Kullback-Leibler divergence, masked."""
#     std = jnp.exp(0.5 * logvar_predicted)
#     return (gaussian_KL(mu_1=mean_predicted,
#                         mu_2=mean_true,
#                         sigma_1=std,
#                         sigma_2=obs_noise_std) * mask).sum() / (jnp.sum(mask) +
#                                                                 1e-10)


class SoftDTWDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = set()

    def __setitem__(self, key, value):
        self._keys.add(key)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key not in self._keys and key.startswith('softdtw'):
            gamma = float(key.split('(')[1].split(')')[0])
            self.__setitem__(key, SoftDTW(gamma=gamma))
        return super().__getitem__(key)


binary_loss = {
    'softmax_bce': softmax_bce,
    'balanced_focal_softmax_bce': softmax_balanced_focal_bce,
    'balanced_focal_bce': balanced_focal_bce,
    'allpairs_hard_rank': allpairs_hard_rank,
    'allpairs_exp_rank': allpairs_exp_rank,
    'allpairs_sigmoid_rank': allpairs_sigmoid_rank
}

numeric_loss = SoftDTWDict({
    'mse': masked_mse,
    'mae': masked_mae,
    'rms': masked_rms,
    'softdtw(0.1)': SoftDTW(gamma=0.1),  # And so any other gamma!
})

colwise_binary_loss = {
    'softmax_bce':
    lambda y, y_hat, mask: softmax_bce(y, y_hat, mask, axis=0),
    'balanced_focal_softmax_bce':
    lambda y, y_hat, mask: softmax_balanced_focal_bce(y, y_hat, mask, axis=0),
    'balanced_focal_bce':
    lambda y, y_hat, mask: balanced_focal_bce(y, y_hat, mask, axis=0)
}

colwise_numeric_loss = {
    'mse': lambda y, y_hat, mask: masked_mse(y, y_hat, mask, axis=0),
    'mae': lambda y, y_hat, mask: masked_mae(y, y_hat, mask, axis=0),
    'rms': lambda y, y_hat, mask: masked_rms(y, y_hat, mask, axis=0)
}
