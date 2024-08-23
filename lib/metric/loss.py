from typing import Optional, Literal, Callable, Final, Dict, Tuple

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jax.tree_util import tree_flatten

from .dtw import SoftDTW
from ..base import Array


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
def bce(y: jnp.ndarray, logits: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
        axis: Optional[int] = None) -> jnp.ndarray:
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
def prob_bce(y: jnp.ndarray, p: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
             axis: Optional[int] = None) -> jnp.ndarray:
    """Binary cross-entropy loss, averaged,
    vectorized (multi-label classification).
    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `p`, after applying the Sigmoid function, where
        each element is a float in :math:`(0, 1)`.
    """
    # For numerical stability.
    p = jnp.where(p < 1e-7, p + 1e-7, p)
    p = jnp.where(p > 1 - 1e-7, p - 1e-7, p)
    terms = y * jnp.log(p) + (1 - y) * jnp.log(1 - p)
    return -jnp.nanmean(terms, where=mask, axis=axis)


@eqx.filter_jit
def softmax_bce(y: jnp.ndarray,
                logits: jnp.ndarray,
                mask: Optional[jnp.ndarray] = None,
                axis: Optional[int] = None):
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
                       mask: Optional[jnp.ndarray] = None,
                       gamma: float = 2,
                       beta: float = 0.999,
                       axis: Optional[int] = None):
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
    e1 = (1 - beta ** n1) / (1 - beta) + 1e-1
    e0 = (1 - beta ** n0) / (1 - beta) + 1e-1

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
                               mask: Optional[jnp.ndarray] = None,
                               gamma: float = 2,
                               beta: float = 0.999,
                               axis: Optional[int] = None):
    """Same as ``balanced_focal_bce``, but with
    applying Softmax activation on the `logits`."""
    # TODO:FIX this for softmax (multinomial case), n1 = np.sum(y, axis=0)
    n1 = jnp.sum(y, axis=0)
    n0 = y.shape[0] - n1
    # Effective number of samples.
    e1 = (1 - beta ** n1) / (1 - beta) + 1e-5
    e0 = (1 - beta ** n0) / (1 - beta) + 1e-5

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


def mse_terms(y: jnp.ndarray, y_hat: jnp.ndarray):
    """L2 loss terms."""
    return jnp.power(y - y_hat, 2)


@eqx.filter_jit
def mse(y: jnp.ndarray, y_hat: jnp.ndarray, mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None):
    """L2 loss."""
    return jnp.mean(mse_terms(y, y_hat), axis=axis, where=mask)


def mae_terms(y: jnp.ndarray, y_hat: jnp.ndarray):
    """L1 loss terms."""
    return jnp.abs(y - y_hat)


@eqx.filter_jit
def mae(y: jnp.ndarray, y_hat: jnp.ndarray, mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None):
    """L1 loss."""
    return jnp.mean(mae_terms(y, y_hat), axis=axis, where=mask)


@eqx.filter_jit
def rms(y: jnp.ndarray, y_hat: jnp.ndarray, mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None):
    """Root mean squared error."""
    return jnp.sqrt(mse(y, y_hat, axis=axis, mask=mask))


_soft_dtw_0_1 = SoftDTW(gamma=0.1)


@eqx.filter_jit
def softdtw(y: jnp.ndarray, y_hat: jnp.ndarray, mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None):
    """Soft-DTW loss."""
    return _soft_dtw_0_1(y, y_hat)


def r2(y: jnp.ndarray, y_hat: jnp.ndarray, mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None):
    """R2 score."""
    y_mean = jnp.mean(y, axis=axis, where=mask)
    ss_tot = jnp.sum(jnp.power(y - y_mean, 2), axis=axis, where=mask)
    ss_res = jnp.sum(jnp.power(y - y_hat, 2), axis=axis, where=mask)
    return 1 - ss_res / ss_tot


@eqx.filter_jit
def gaussian_kl(y: Tuple[jnp.ndarray, jnp.ndarray], y_hat: Tuple[jnp.ndarray, jnp.ndarray],
                mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None) -> jnp.ndarray:
    """KL divergence between two Gaussian distributions."""
    mean, std = y
    mean_hat, std_hat = y_hat
    kl = jnp.log(std) - jnp.log(std_hat) + (std_hat ** 2 + (mean - mean_hat) ** 2) / (2 * (std ** 2)) - 0.5
    return jnp.mean(kl, axis=axis, where=mask)


@eqx.filter_jit
def gaussian_jsd(y: Tuple[jnp.ndarray, jnp.ndarray], y_hat: Tuple[jnp.ndarray, jnp.ndarray],
                 mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None) -> jnp.ndarray:
    """Jenson-Shannon divergence between two Gaussian distributions."""
    term1 = gaussian_kl(y, y_hat, mask=mask, axis=axis)
    term2 = gaussian_kl(y_hat, y, mask=mask, axis=axis)
    return 0.5 * (term1 + term2)


@eqx.filter_jit
def log_normal(y: Tuple[jnp.ndarray, jnp.ndarray], y_hat: Tuple[jnp.ndarray, jnp.ndarray],
               mask: Optional[jnp.ndarray] = None, axis: Optional[int] = None) -> jnp.ndarray:
    """Log-normal loss."""
    mean, _ = y
    mean_hat, std_hat = y_hat
    error = (mean - mean_hat) / (std_hat + 1e-6)
    log_normal_loss = 0.5 * (error ** 2 + 2 * jnp.log(std_hat + 1e-6))
    return jnp.mean(log_normal_loss, axis=axis, where=mask)


BinaryLossLiteral = Literal[
    'softmax_bce', 'balanced_focal_softmax_bce', 'balanced_focal_bce', 'allpairs_hard_rank',
    'allpairs_exp_rank', 'allpairs_sigmoid_rank', 'bce']

NumericLossLiteral = Literal['mse', 'mae', 'rms', 'soft_dtw_0_1', 'r2']

ProbNumericLossLiteral = Literal['kl_gaussian', 'log_normal', 'jsd_gaussian']

LossSignature = Callable[[Array, Array, Optional[Array], Optional[int]], Array | float]
ProbLossSignature = Callable[[Tuple[Array, Array], Tuple[Array, Array], Optional[Array], Optional[int]], Array | float]

LOGITS_BINARY_LOSS: Final[Dict[BinaryLossLiteral, LossSignature]] = {
    'bce': bce,
    'softmax_bce': softmax_bce,
    'balanced_focal_softmax_bce': softmax_balanced_focal_bce,
    'balanced_focal_bce': balanced_focal_bce,
    'allpairs_hard_rank': allpairs_hard_rank,
    'allpairs_exp_rank': allpairs_exp_rank,
    'allpairs_sigmoid_rank': allpairs_sigmoid_rank
}

PROB_BINARY_LOSS: Final[Dict[BinaryLossLiteral, LossSignature]] = {
    'bce': prob_bce}

NUMERIC_LOSS: Final[Dict[NumericLossLiteral, LossSignature]] = {
    'mse': mse,
    'mae': mae,
    'rms': rms,
    'soft_dtw_0_1': softdtw,
    'r2': r2
}

PROB_NUMERIC_LOSS: Final[Dict[ProbNumericLossLiteral, ProbLossSignature]] = {
    'kl_gaussian': gaussian_kl,
    'jsd_gaussian': gaussian_jsd,
    'log_normal': log_normal}
