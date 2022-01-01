from functools import partial
from collections import defaultdict
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional, Tuple)
from enum import Flag, auto

from absl import logging

import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from jax.nn import softplus, sigmoid
from jax.tree_util import tree_flatten
from sklearn import metrics


@jax.jit
def jit_sigmoid(x):
    return sigmoid(x)


@jax.jit
def bce(y: jnp.ndarray, logits: jnp.ndarray):
    return jnp.mean(y * softplus(-logits) + (1 - y) * softplus(logits))


# The following loss function employs two concepts:
# A) Effective number of sample, to mitigate class imbalance:
# Paper: Class-Balanced Loss Based on Effective Number of Samples (Cui et al)
# B) Focal loss, to underweight the easy to classify samples:
# Paper: Focal Loss for Dense Object Detection (Lin et al)
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


@jax.jit
def l2_squared(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.vdot(x, x) for x in leaves)


@jax.jit
def l1_absolute(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.sum(jnp.fabs(x)) for x in leaves)


def parameters_size(pytree):
    leaves, _ = tree_flatten(pytree)
    return sum(jnp.size(x) for x in leaves)


@jax.jit
def numeric_error(mean_true: jnp.ndarray, mean_predicted: jnp.ndarray,
                  logvar: jnp.ndarray) -> jnp.ndarray:
    sigma = jnp.exp(0.5 * logvar)
    return (mean_true - mean_predicted) / sigma


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
    cm = cm / (cm.sum() + 1e-10)
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


def auc_scores(detectability, label_prefix):
    def compute_auc(v_truth, v_preds):
        fpr, tpr, _ = metrics.roc_curve(v_truth, v_preds, pos_label=1)
        return metrics.auc(fpr, tpr)

    gtruth = []
    preds = []
    for points in detectability.values():
        for inference in points.values():
            # In some cases the ground truth is all negative, avoid them.
            # Note: in Python {0.0, 1.0} == {0, 1} => True
            if set(onp.unique(inference['diag_true'])) == {0, 1}:
                gtruth.append(inference['diag_true'])
                preds.append(jit_sigmoid(inference[f'{label_prefix}_logits']))

    if len(preds) == 0 or any(onp.isnan(p).any() for p in preds):
        logging.warning('no detections or nan probs')
        # nan is returned indicator of undetermined AUC.
        return float('nan')

    macro_auc = compute_auc(onp.hstack(gtruth), onp.hstack(preds))
    micro_auc = sum(compute_auc(t, p) for t, p in zip(gtruth, preds))

    return {'MACRO-AUC': macro_auc, 'MICRO-AUC': micro_auc / len(preds)}


def top_k_detectability_scores(codes_by_percentiles, detections_df,
                               label_prefix):
    rate = {}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        detection_rate = codes_detections_df[f'{label_prefix}_detected'].mean()
        rate[f'{label_prefix}_ACC-P{i}'] = detection_rate

    percentiles = {}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        C = len(codes)
        N = len(codes_detections_df)
        percentiles[f'P{i}: N ~ C'] = f'{N} ~ {C}'

    return rate, percentiles


def compute_confusion_matrix(detectability, label_prefix):
    cm = []
    label = f'{label_prefix}_logits'
    for points in detectability.values():
        for inference in points.values():
            ground_truth = inference['diag_true']
            logits = inference[label]
            cm.append(confusion_matrix(ground_truth, jit_sigmoid(logits)))
    if cm:
        return sum(cm)
    else:
        return onp.zeros((2, 2)) + onp.nan


def top_k_detectability_df(top_k: int, res, prefixes):
    cols = ['subject_id', 'point_n', 'code']
    for prefix in prefixes:
        cols.append(f'{prefix}_detected')

    def top_k_detectability(point):
        ground_truth = jnp.argwhere(point['diag_true']).squeeze()
        if ground_truth.ndim > 0:
            ground_truth = set(onp.array(ground_truth))
        else:
            ground_truth = {ground_truth.item()}

        detections = defaultdict(list)

        for prefix in prefixes:
            predictions = set(
                onp.array(onp.argsort(point[f'{prefix}_logits'])[-top_k:]))
            for code_i in ground_truth:
                detections[code_i].append(code_i in predictions)
        return detections

    df_list = []

    for subject_id, points in res.items():
        for point_n, point in points.items():
            for code_i, detected in top_k_detectability(point).items():
                df_list.append((subject_id, point_n, code_i, *detected))

    return pd.DataFrame(df_list, columns=cols)


class EvalFlag(Flag):
    CM = auto()
    POST = auto()

    @staticmethod
    def has(flag, attr):
        return (flag & attr).value != 0


def evaluation_table(raw_results, codes_by_percentiles):
    evals = {colname: {} for colname in raw_results}
    flat_evals = {}
    for colname, res in raw_results.items():
        for rowname, val in res['loss'].items():
            evals[colname][rowname] = val

        det = res['diag_detectability']
        cm = compute_confusion_matrix(det, 'pre')
        cm_scores = confusion_matrix_scores(cm)
        for rowname, val in cm_scores.items():
            evals[colname][rowname] = val

        for rowname, val in auc_scores(det, 'pre').items():
            evals[colname][rowname] = val

        det20 = top_k_detectability_df(20, det, ['pre'])
        det20_scores = top_k_detectability_scores(codes_by_percentiles, det20,
                                                  'pre')[0]
        for rowname, val in det20_scores.items():
            evals[colname][rowname] = val

        evals[colname] = {
            rowname: float(val)
            for rowname, val in evals[colname].items()
        }
        flat_evals.update({
            f'{colname}_{rowname}': val
            for rowname, val in evals[colname].items()
        })
    return pd.DataFrame(evals), flat_evals
