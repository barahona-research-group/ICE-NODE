from functools import partial
import sys
from collections import defaultdict
from typing import (AbstractSet, Any, Callable, Dict, Iterable, List, Mapping,
                    Optional)
from enum import Flag, auto

from absl import logging
from tqdm import tqdm

import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from jax.nn import softplus, sigmoid
from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten
from sklearn import metrics
import scipy.stats as st

from .delong import DeLongTest, FastDeLongTest


class OOPError(Exception):
    pass


@jax.jit
def jit_sigmoid(x):
    return sigmoid(x)


@jax.jit
def bce(y: jnp.ndarray, logits: jnp.ndarray):
    return jnp.mean(y * softplus(-logits) + (1 - y) * softplus(logits))


@jax.jit
def softmax_logits_bce(y: jnp.ndarray, logits: jnp.ndarray):
    return -jnp.mean(y * jax.nn.log_softmax(logits) +
                     (1 - y) * jnp.log(1 - jax.nn.softmax(logits)))


@jax.jit
def weighted_bce(y: jnp.ndarray, logits: jnp.ndarray, weights: jnp.ndarray):
    return jnp.mean(weights * (y * softplus(-logits) +
                               (1 - y) * softplus(logits)))


@jax.jit
def softmax_logits_weighted_bce(y: jnp.ndarray, logits: jnp.ndarray,
                                weights: jnp.ndarray):
    return -jnp.mean(weights * (y * jax.nn.log_softmax(logits) +
                                (1 - y) * jnp.log(1 - jax.nn.softmax(logits))))


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
def softmax_logits_balanced_focal_bce(y: jnp.ndarray,
                                      logits: jnp.ndarray,
                                      gamma=2,
                                      beta=0.999):
    n1 = jnp.sum(y)
    n0 = jnp.size(y) - n1
    # Effective number of samples.
    e1 = (1 - beta**n1) / (1 - beta) + 1e-1
    e0 = (1 - beta**n0) / (1 - beta) + 1e-1

    # Focal weighting
    p = jax.nn.softmax(logits)
    w1 = jnp.power(1 - p, gamma)
    w0 = jnp.power(p, gamma)
    return -jnp.mean(y * (w1 / e1) * jax.nn.log_softmax(logits) + (1 - y) *
                     (w0 / e0) * jnp.log(1 - p))


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


def compute_auc(v_truth, v_preds):
    fpr, tpr, _ = metrics.roc_curve(v_truth, v_preds, pos_label=1)
    return metrics.auc(fpr, tpr)


def auc_scores(detectability):
    gtruth = []
    preds = []
    for points in detectability.values():
        for inference in points.values():
            # In some cases the ground truth is all negative, avoid them.
            # Note: in Python {0.0, 1.0} == {0, 1} => True
            if set(onp.unique(inference['true_diag'])) == {0, 1}:
                gtruth.append(inference['true_diag'])
                preds.append(jit_sigmoid(inference['pred_logits']))

    if len(preds) == 0 or any(onp.isnan(p).any() for p in preds):
        logging.warning('no detections or nan probs')
        # nan is returned indicator of undetermined AUC.
        return {'MACRO-AUC': float('nan'), 'MICRO-AUC': float('nan')}

    macro_auc = compute_auc(onp.hstack(gtruth), onp.hstack(preds))
    micro_auc = sum(compute_auc(t, p) for t, p in zip(gtruth, preds))

    return {'MACRO-AUC': macro_auc, 'MICRO-AUC': micro_auc / len(preds)}


def codes_auc_scores(detectability):
    ground_truth = []
    predictions = []

    for admissions in detectability.values():
        for info in admissions.values():
            ground_truth.append(info['true_diag'])
            predictions.append(jit_sigmoid(info['pred_logits']))

    ground_truth_mat = onp.vstack(ground_truth)
    predictions_mat = onp.vstack(predictions)

    n_codes = []
    auc = []
    for code_index in range(ground_truth_mat.shape[1]):
        code_ground_truth = ground_truth_mat[:, code_index]
        code_predictions = predictions_mat[:, code_index]

        n_codes.append(code_ground_truth.sum())
        auc.append(compute_auc(code_ground_truth, code_predictions))

    data = {'CODE_INDEX': range(len(auc)), 'N_CODES': n_codes, 'AUC': auc}

    return pd.DataFrame(data)


def admissions_auc_scores(detectability):
    subject_ids = []
    # HADM_ID in MIMIC-III
    admission_ids = []
    # Admission order mostly
    admission_indexes = []
    # Time as days since first discharge
    time = []
    los = []
    n_codes = []
    adm_auc = []
    nfe = []
    intervals = []
    r = []

    for subject_id, admissions in detectability.items():
        for i, admission_index in enumerate(sorted(admissions.keys())):
            info = admissions[admission_index]
            # If ground truth has no clinical codes, skip.
            if set(onp.unique(info['true_diag'])) != {0, 1}:
                continue

            if 'time' in info:
                time.append(info['time'])

                # Record intervals between two discharges
                if i == 0:
                    intervals.append(time[-1])
                else:
                    intervals.append(info['time'] - time[-2])

            if 'nfe' in info:
                nfe.append(info['nfe'])

            if 'los' in info:
                los.append(info['los'])

            if 'R/T' in info:
                r.append(info['R/T'])

            admission_indexes.append(i)
            admission_ids.append(info['admission_id'])
            subject_ids.append(subject_id)

            true_diag = info['true_diag']
            diag_score = jit_sigmoid(info['pred_logits'])
            auc_score = compute_auc(true_diag, diag_score)
            adm_auc.append(auc_score)
            n_codes.append(true_diag.sum())

    data = {
        'SUBJECT_ID': subject_ids,
        'HADM_ID': admission_ids,
        'HADM_IDX': admission_indexes,
        'AUC': adm_auc,
        'N_CODES': n_codes
    }
    if (len(time) > 0 and len(intervals) > 0 and len(los) > 0 and len(r) > 0
            and len(nfe) > 0):
        data['TIME'] = time
        data['INTERVALS'] = intervals
        data['LOS'] = los
        data['R/T'] = r
        data['NFE'] = nfe
    return pd.DataFrame(data)


def top_k_detectability_scores(codes_by_percentiles, detections_df):
    rate = {}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        detection_rate = codes_detections_df['detected'].mean()
        rate[f'ACC-P{i}'] = detection_rate

    percentiles = {}
    for i, codes in enumerate(codes_by_percentiles):
        codes_detections_df = detections_df[detections_df.code.isin(codes)]
        C = len(codes)
        N = len(codes_detections_df)
        percentiles[f'P{i}: N ~ C'] = f'{N} ~ {C}'

    return rate, percentiles


def compute_confusion_matrix(detectability):
    cm = []
    for points in detectability.values():
        for inference in points.values():
            ground_truth = inference['true_diag']
            logits = inference['pred_logits']
            cm.append(confusion_matrix(ground_truth, jit_sigmoid(logits)))
    if cm:
        return sum(cm)
    else:
        return onp.zeros((2, 2)) + onp.nan


def top_k_detectability_df(top_k: int, res):
    cols = ['subject_id', 'point_n', 'code', 'detected']

    def top_k_detectability(point):
        ground_truth = jnp.argwhere(point['true_diag']).squeeze()
        if ground_truth.ndim > 0:
            ground_truth = set(onp.array(ground_truth))
        else:
            ground_truth = {ground_truth.item()}

        detections = defaultdict(list)
        predictions = set(onp.array(
            onp.argsort(point['pred_logits'])[-top_k:]))
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


def evaluation_table(raw_results, codes_by_percentiles, top_k=20):
    evals = {colname: {} for colname in raw_results}
    flat_evals = {}
    for colname, res in raw_results.items():
        for rowname, val in res['loss'].items():
            evals[colname][rowname] = val
        for rowname, val in res['stats'].items():
            evals[colname][rowname] = val

        det = res['diag_detectability']
        cm = compute_confusion_matrix(det)
        cm_scores = confusion_matrix_scores(cm)
        for rowname, val in cm_scores.items():
            evals[colname][rowname] = val

        for rowname, val in auc_scores(det).items():
            evals[colname][rowname] = val

        det_topk = top_k_detectability_df(top_k, det)
        det_topk_scores = top_k_detectability_scores(codes_by_percentiles,
                                                     det_topk)[0]
        for rowname, val in det_topk_scores.items():
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


# MIT License

# Copyright (c) 2021 Konrad Heidler

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class AbstractDTW:
    """https://github.com/khdlr/softdtw_jax/blob/main/softdtw_jax/softdtw_jax.py"""

    def __init__(self, pairwise_distance_f):
        self.pairwise_distance_f = pairwise_distance_f

    def minimum(self, *args):
        raise OOPError('This should be implemented')

    def __call__(self, target, prediction):
        """
        Compute the DTW distance.

        """
        D = self.pairwise_distance_f(target, prediction)

        # wlog: H >= W
        if D.shape[0] < D.shape[1]:
            D = D.T
        H, W = D.shape

        rows = []
        for row in range(H):
            rows.append(pad_inf(D[row], row, H - row - 1))

        model_matrix = jnp.stack(rows, axis=1)
        init = (pad_inf(model_matrix[0], 1,
                        0), pad_inf(model_matrix[1] + model_matrix[0, 0], 1,
                                    0))

        def scan_step(carry, current_antidiagonal):
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right = one_ago[:-1]
            down = one_ago[1:]
            best = self.minimum(jnp.stack([diagonal, right, down], axis=-1))
            next_row = best + current_antidiagonal
            next_row = pad_inf(next_row, 1, 0)

            return (one_ago, next_row), next_row

        # Manual unrolling:
        # carry = init
        # for i, row in enumerate(model_matrix[2:]):
        #     carry, y = scan_step(carry, row)

        carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
        return carry[1][-1]


class DTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Dynamic programming algorithm optimization for spoken word recognition"
    by Hiroaki Sakoe and Seibi Chiba (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'DTW'

    def minimum(self, args):
        return jnp.min(args, axis=-1)


class SoftDTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
    by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'SoftDTW'

    def __init__(self, pairwise_distance_f, gamma=1.0):
        super().__init__(pairwise_distance_f=pairwise_distance_f)

        assert gamma > 0, "Gamma needs to be positive."
        self.gamma = gamma
        self.__name__ = f'SoftDTW({self.gamma})'
        self.minimum_impl = self.make_softmin(gamma)

    def make_softmin(self, gamma):
        """
        We need to manually define the gradient of softmin
        to ensure (1) numerical stability and (2) prevent nans from
        propagating over valid values.
        """

        def softmin_raw(array):
            return -gamma * logsumexp(array / -gamma, axis=-1)

        softmin = jax.custom_vjp(softmin_raw)

        def softmin_fwd(array):
            return softmin(array), (array / -gamma, )

        def softmin_bwd(res, g):
            scaled_array, = res
            grad = jnp.where(
                jnp.isinf(scaled_array), jnp.zeros(scaled_array.shape),
                jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1))
            return grad,

        softmin.defvjp(softmin_fwd, softmin_bwd)
        return softmin

    def minimum(self, args):
        return self.minimum_impl(args)


@jax.jit
def distance_matrix_bce(a, b_logits):
    """
    Return pairwise crossentropy between two timeseries.
    Args:
        a: First time series (m, p).
        b: Second time series (n, p).
    Returns:
        An (m, n) distance matrix computed by a pairwise distance function.
            on the elements of a and b.
    """
    m, p = a.shape
    n, p = b_logits.shape
    assert a.shape[1] == b_logits.shape[1], "Dimensions mismatch."

    b_logits = jnp.broadcast_to(b_logits, (m, n, p))
    a = jnp.expand_dims(a, 1)
    a = jnp.broadcast_to(a, (m, n, p))

    D = a * jax.nn.softplus(-b_logits) + (1 - a) * jax.nn.softplus(b_logits)

    return jnp.mean(D, axis=-1)


# Utility functions
@jax.jit
def distance_matrix_euc(a, b_logits):
    b = sigmoid(b_logits)
    a = jnp.expand_dims(a, axis=1)
    b = jnp.expand_dims(b, axis=0)
    D = jnp.square(a - b)
    return jnp.mean(D, axis=-1)


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def codes_auc_pairwise_tests(results, fast=False):
    """
    Evaluate the AUC scores for each diagnosis code for each classifier. In addition,
    conduct a pairwise test on the difference of AUC scores between each
    pair of classifiers using DeLong test. Codes that have either less than two positive cases or
    have less than two negative cases are discarded (AUC computation and difference test requirements).
    """
    # Classifier labels
    clf_labels = list(sorted(results.keys()))

    def extract_subjects():
        _results = list(results.values())
        subjects = set(_results[0].keys())
        assert all(set(_res.keys()) == subjects for _res in
                   _results), "results should correspond to the same group"
        return list(sorted(subjects))

    subjects = extract_subjects()

    def extract_ground_truth_and_scores():
        ground_truth_mat = {}
        scores_mat = {}
        for clf_label in clf_labels:
            clf_ground_truth = []
            clf_scores = []
            clf_results = results[clf_label]
            for subject_id in subjects:
                subject_results = clf_results[subject_id]
                for adm_idx in sorted(subject_results.keys()):
                    adm_results = subject_results[adm_idx]
                    clf_ground_truth.append(adm_results['true_diag'])
                    clf_scores.append(adm_results['pred_logits'])
            ground_truth_mat[clf_label] = onp.vstack(clf_ground_truth)
            scores_mat[clf_label] = onp.vstack(clf_scores)

        g0 = ground_truth_mat[clf_labels[0]]
        assert all(
            (g0 == gi).all()
            for gi in ground_truth_mat.values()), "Mismatch in ground-truth!"

        return g0, scores_mat

    ground_truth_mat, scores = extract_ground_truth_and_scores()

    clf_pairs = []
    for i in range(len(clf_labels)):
        for j in range(i + 1, len(clf_labels)):
            clf_pairs.append((clf_labels[i], clf_labels[j]))

    n_positive_codes = []
    auc = {clf: [] for clf in clf_labels}
    auc_var = {clf: [] for clf in clf_labels}

    pairwise_tests = {pair: [] for pair in clf_pairs}

    for code_index in tqdm(range(ground_truth_mat.shape[1])):
        code_ground_truth = ground_truth_mat[:, code_index]
        n_pos = code_ground_truth.sum()
        n_neg = len(code_ground_truth) - n_pos

        # This is a requirement for pairwise testing.
        if n_pos < 2 or n_neg < 2:
            continue

        n_positive_codes.append(n_pos)
        # Since we iterate on pairs, some AUC are computed more than once.
        # Update this dictionary for each AUC computed, then append the results
        # to the big list of auc.
        _auc = {}
        _auc_var = {}
        for (clf1, clf2) in clf_pairs:
            scores1 = scores[clf1][:, code_index]
            scores2 = scores[clf2][:, code_index]
            if fast:
                auc1, auc2, auc1_v, auc2_v, p = FastDeLongTest.delong_roc_test(
                    code_ground_truth, scores1, scores2)
            else:
                auc1, auc2, auc1_v, auc2_v, p = DeLongTest.difference_test(
                    code_ground_truth, scores1, scores2)

            pairwise_tests[(clf1, clf2)].append(p)

            _auc[clf1] = auc1
            _auc[clf2] = auc2
            _auc_var[clf1] = auc1_v
            _auc_var[clf2] = auc2_v

        for clf, a in _auc.items():
            auc[clf].append(a)
        for clf, v in _auc_var.items():
            auc_var[clf].append(v)
    data = {
        'CODE_INDEX': range(len(n_positive_codes)),
        'N_POSITIVE_CODES': n_positive_codes,
        **{f'AUC({clf})': auc_vals
           for clf, auc_vals in sorted(auc.items())},
        **{
            f'VAR[AUC({clf})]': auc_v
            for clf, auc_v in sorted(auc_var.items())
        },
        **{
            f'P0(AUC_{clf1}==AUC_{clf2})': p_vals
            for (clf1, clf2), p_vals in sorted(pairwise_tests.items())
        }
    }

    return pd.DataFrame(data)
