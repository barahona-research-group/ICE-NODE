from functools import partial
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple, Union)

import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from jax.profiler import annotate_function
from jax.nn import softplus, sigmoid, leaky_relu
from jax.tree_util import tree_flatten, tree_map


@jax.jit
def jit_sigmoid(x):
    return sigmoid(x)


@partial(annotate_function, name="bce")
@jax.jit
def bce(y: jnp.ndarray, logits: jnp.ndarray):
    return jnp.mean(y * softplus(-logits) + (1 - y) * softplus(logits))


# The following loss function employs two concepts:
# A) Effective number of sample, to mitigate class imbalance:
# Paper: Class-Balanced Loss Based on Effective Number of Samples (Cui et al)
# B) Focal loss, to underweight the easy to classify samples:
# Paper: Focal Loss for Dense Object Detection (Lin et al)
@partial(annotate_function, name="balanced_focal_bce")
@jax.jit
def balanced_focal_bce(y: jnp.ndarray,
                       logits: jnp.ndarray,
                       gamma=2,
                       beta=0.999):
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
    return jnp.mean(y * (w1 / e1) * softplus(-logits) + (1 - y) *
                    (w0 / e0) * softplus(logits))


@partial(annotate_function, name="l2_loss")
@jax.jit
def l2_squared(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.vdot(x, x) for x in leaves)


@partial(annotate_function, name="l1_loss")
@jax.jit
def l1_absolute(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.sum(jnp.fabs(x)) for x in leaves)


def parameters_size(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.size(x) for x in leaves)

@partial(annotate_function, name="numeric_error")
@jax.jit
def numeric_error(mean_true: jnp.ndarray, mean_predicted: jnp.ndarray,
                  logvar: jnp.ndarray) -> jnp.ndarray:
    sigma = jnp.exp(0.5 * logvar)
    return (mean_true - mean_predicted) / sigma


@partial(annotate_function, name="lognormal_loss")
@jax.jit
def lognormal_loss(mask: jnp.ndarray, error: jnp.ndarray,
                   logvar: jnp.ndarray) -> float:
    log_lik_c = jnp.log(jnp.sqrt(2 * jnp.pi))
    return 0.5 * ((jnp.power(error, 2) + logvar + 2 * log_lik_c) *
                  mask).sum() / (mask.sum() + 1e-10)


def gaussian_KL(mu_1: jnp.ndarray, mu_2: jnp.ndarray, sigma_1: jnp.ndarray,
                sigma_2: float) -> jnp.ndarray:
    return (jnp.log(sigma_2) - jnp.log(sigma_1) +
            (jnp.power(sigma_1, 2) + jnp.power(
                (mu_1 - mu_2), 2)) / (2 * sigma_2**2) - 0.5)


@partial(annotate_function, name="kl_loss")
@jax.jit
def compute_KL_loss(mean_true: jnp.ndarray,
                    mask: jnp.ndarray,
                    mean_predicted: jnp.ndarray,
                    logvar_predicted: jnp.ndarray,
                    obs_noise_std: float = 1e-1) -> float:
    std = jnp.exp(0.5 * logvar_predicted)
    return (gaussian_KL(mu_1=mean_predicted,
                        mu_2=mean_true,
                        sigma_1=std,
                        sigma_2=obs_noise_std) * mask).sum() / (jnp.sum(mask) +
                                                                1e-10)


@jax.jit
def confusion_matrix(y_true: jnp.ndarray, y_hat: jnp.ndarray):
    y_hat = (jnp.round(y_hat) == 1)
    y_true = (y_true == 1)

    tp = jnp.sum(y_true & y_hat)
    tn = jnp.sum((~y_true) & (~y_hat))
    fp = jnp.sum((~y_true) & y_hat)
    fn = jnp.sum(y_true & (~y_hat))

    return jnp.array([[tp, fn], [fp, tn]], dtype=int)


def confusion_matrix_scores(cm: jnp.ndarray):
    cm = cm / cm.sum()
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    p = tp + fn
    n = tn + fp
    return {
        'accuracy': (tp + tn) / (p + n),
        'recall': tp / p,
        'npv': tn / (tn + fn),
        'specificity': tn / n,
        'precision': tp / (tp + fp),
        'f1-score': 2 * tp / (2 * tp + fp + fn),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def code_detectability(top_k: int, true_diag: jnp.ndarray,
                       prejump_predicted_diag: jnp.ndarray,
                       postjump_predicted_diag: jnp.ndarray):
    ground_truth = jnp.argwhere(true_diag).squeeze()
    if ground_truth.ndim > 0:
        ground_truth = set(onp.array(ground_truth))
    else:
        ground_truth = {ground_truth.item()}

    prejump_predictions = set(onp.array(jnp.argsort(prejump_predicted_diag)[-top_k:]))
    postjump_predictions = set(onp.array(jnp.argsort(postjump_predicted_diag)[-top_k:]))
    detections = []
    for code_i in ground_truth:
        pre_detected, post_detected = 0, 0
        if code_i in prejump_predictions:
            pre_detected = 1
        if code_i in postjump_predictions:
            post_detected = 1
        detections.append((code_i, pre_detected, post_detected))

    return detections


def code_detectability_df(top_k: int, true_diag: Dict[int, jnp.ndarray],
                          prejump_predicted_diag: Dict[int, jnp.ndarray],
                          postjump_predicted_diag: Dict[int, jnp.ndarray],
                          point_n: int):
    detections = {
        i: code_detectability(top_k, true_diag[i], prejump_predicted_diag[i],
                              postjump_predicted_diag[i])
        for i in true_diag.keys()
    }
    df_list = []

    for subject_id, _detections in detections.items():
        for code_i, pre_detected, post_detected in _detections:
            df_list.append((subject_id, point_n, code_i, pre_detected,
                            post_detected, top_k))

    if df_list:
        return pd.DataFrame(df_list,
                            columns=[
                                'subject_id', 'point_n', 'code',
                                'pre_detected', 'post_detected', 'top_k'
                            ])
    else:
        return None


def code_detectability_by_percentiles(codes_by_percentiles, detections_df):
    rate = {'pre': {}, 'post': {}}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        detection_rate_pre = codes_detections_df.pre_detected.mean()
        detection_rate_post = codes_detections_df.post_detected.mean()
        C = len(codes)
        N = len(codes_detections_df)
        rate['pre'][f'P{i}(N={N} C={len(codes)})'] = detection_rate_pre
        rate['post'][f'P{i}(N={N} C={len(codes)})'] = detection_rate_post
    return rate


