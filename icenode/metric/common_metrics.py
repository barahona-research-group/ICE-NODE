"""Performance metrics and loss functions."""

from collections import defaultdict
from enum import Flag, auto
from typing import Dict, Iterable

from absl import logging
from tqdm import tqdm

import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from jax.nn import softplus, sigmoid
from jax.tree_util import tree_flatten
from sklearn import metrics

from .delong import DeLongTest, FastDeLongTest
from ..ehr_predictive.risk import BatchPredictedRisks, SubjectPredictedRisk


@jax.jit
def jit_sigmoid(x):
    """JAX-compiled Sigmoid activation."""
    return sigmoid(x)


@jax.jit
def bce(y: jnp.ndarray, logits: jnp.ndarray):
    """Binary cross-entropy loss, averaged,
    vectorized (multi-label classification).
    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
        element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Sigmoid function, where
      each element is a float in :math:`(-\infty, \infty)`.
    """
    return jnp.mean(y * softplus(-logits) + (1 - y) * softplus(logits))


@jax.jit
def softmax_logits_bce(y: jnp.ndarray, logits: jnp.ndarray):
    """Categorical cross-entropy, averaged.

    The function takes two inputs:
      - The ground-truth `y` is a vector, where each
      element is an integer in :math:`\{0, 1\}`.
      - The predictions `logits`, before applying the Softmax function,
      where each element is a float in :math:`(-\infty, \infty)`.
    """
    return -jnp.mean(y * jax.nn.log_softmax(logits) +
                     (1 - y) * jnp.log(1 - jax.nn.softmax(logits)))


@jax.jit
def weighted_bce(y: jnp.ndarray, logits: jnp.ndarray, weights: jnp.ndarray):
    """Weighted version of ``bce``."""
    return jnp.mean(weights * (y * softplus(-logits) +
                               (1 - y) * softplus(logits)))


@jax.jit
def softmax_logits_weighted_bce(y: jnp.ndarray, logits: jnp.ndarray,
                                weights: jnp.ndarray):
    """Weighted version of ``softmax_logits_bce``."""
    return -jnp.mean(weights * (y * jax.nn.log_softmax(logits) +
                                (1 - y) * jnp.log(1 - jax.nn.softmax(logits))))


@jax.jit
def balanced_focal_bce(y: jnp.ndarray,
                       logits: jnp.ndarray,
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
    return jnp.mean(y * (w1 / e1) * softplus(-logits) + (1 - y) *
                    (w0 / e0) * softplus(logits))


@jax.jit
def softmax_logits_balanced_focal_bce(y: jnp.ndarray,
                                      logits: jnp.ndarray,
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
    return -jnp.mean(y * (w1 / e1) * jax.nn.log_softmax(logits) + (1 - y) *
                     (w0 / e0) * jnp.log(1 - p))


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


@jax.jit
def confusion_matrix(y_true: jnp.ndarray, y_hat: jnp.ndarray):
    """Return the confusion matrix given the ground-truth `y_true`
    and the predictions `y_hat (rounded to :math:`\{0, 1\})`."""
    y_hat = (jnp.round(y_hat) == 1)
    y_true = (y_true == 1)

    tp = jnp.sum(y_true & y_hat)
    tn = jnp.sum((~y_true) & (~y_hat))
    fp = jnp.sum((~y_true) & y_hat)
    fn = jnp.sum(y_true & (~y_hat))

    return jnp.array([[tp, fn], [fp, tn]], dtype=int)


def confusion_matrix_scores(cm: jnp.ndarray):
    """From the confusion matrix, compute: accuracy, recall, NPV, specificity,
    precision, F1-Score, TP, TN, FP, and FN."""
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
    """Compute the area under the ROC from the ground-truth `v_truth` and
     the predictions `v_preds`."""
    fpr, tpr, _ = metrics.roc_curve(v_truth, v_preds, pos_label=1)
    return metrics.auc(fpr, tpr)


def auc_scores(risk_prediction: BatchPredictedRisks):
    gtruth = []
    preds = []
    for subject_risks in risk_prediction.subject_risks.values():
        for risk in subject_risks.values():
            # In some cases the ground truth is all negative, avoid them.
            # Note: in Python {0.0, 1.0} == {0, 1} => True
            if set(onp.unique(risk.ground_truth)) == {0, 1}:
                gtruth.append(risk.ground_truth)
                preds.append(jit_sigmoid(risk.prediction))

    if len(preds) == 0 or any(onp.isnan(p).any() for p in preds):
        logging.warning('no detections or nan probs')
        # nan is returned indicator of undetermined AUC.
        return {'MACRO-AUC': float('nan'), 'MICRO-AUC': float('nan')}

    macro_auc = compute_auc(onp.hstack(gtruth), onp.hstack(preds))
    micro_auc = sum(compute_auc(t, p) for t, p in zip(gtruth, preds))

    return {'MACRO-AUC': macro_auc, 'MICRO-AUC': micro_auc / len(preds)}


def codes_auc_scores(risk_prediction: BatchPredictedRisks):
    ground_truth = []
    predictions = []

    for subject_risks in risk_prediction.subject_risks.values():
        for risk in subject_risks.values():
            ground_truth.append(risk.ground_truth)
            predictions.append(jit_sigmoid(risk.prediction))

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


def admissions_auc_scores(risk_prediction: BatchPredictedRisks):
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

    for subject_id, subject_risks in risk_prediction.subject_risks.items():
        for i, admission_index in enumerate(sorted(subject_risks.keys())):
            risk = subject_risks[admission_index]
            # If ground truth has no clinical codes, skip.
            if set(onp.unique(risk.ground_truth)) != {0, 1}:
                continue

            if 'time' in risk.other_attrs:
                time.append(risk.other_attrs['time'])

                # Record intervals between two discharges
                if i == 0:
                    intervals.append(time[-1])
                else:
                    intervals.append(risk.other_attrs['time'] - time[-2])

            if 'nfe' in risk.other_attrs:
                nfe.append(risk.other_attrs['nfe'])

            if 'los' in risk.other_attrs:
                los.append(risk.other_attrs['los'])

            admission_indexes.append(i)
            admission_ids.append(risk.admission_id)
            subject_ids.append(subject_id)

            true_diag = risk.ground_truth
            diag_score = jit_sigmoid(risk.prediction)
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
    if all(map(len, (time, intervals, los, nfe))):
        data['TIME'] = time
        data['INTERVALS'] = intervals
        data['LOS'] = los
        data['NFE'] = nfe
    return pd.DataFrame(data)


def compute_confusion_matrix(risk_prediction: BatchPredictedRisks):
    cm = []
    for subject_risks in risk_prediction.subject_risks.values():
        for risk in subject_risks.values():
            logits = risk.prediction
            cm.append(confusion_matrix(risk.ground_truth, jit_sigmoid(logits)))
    if cm:
        return sum(cm)
    else:
        return onp.zeros((2, 2)) + onp.nan


def top_k_detectability_scores(code_groups, risk_predictions, top_k_list):
    ground_truth = []
    risks = []

    for subject_risks in risk_predictions.subject_risks.values():
        for risk_pred in subject_risks.values():
            risks.append(risk_pred.prediction)
            ground_truth.append(risk_pred.ground_truth)

    risks = onp.vstack(risks)
    ground_truth = onp.vstack(ground_truth).astype(bool)
    topk_risks = onp.argpartition(-risks, top_k_list, axis=1)

    true_positive = {}
    for k in top_k_list:
        topk_risks_i = topk_risks[:, :k]
        topk_risks_k = onp.zeros_like(risks, dtype=bool)
        onp.put_along_axis(topk_risks_k, topk_risks_i, True, 1)
        true_positive[k] = (topk_risks_k & ground_truth)

    rate = {k: {} for k in top_k_list}
    for i, code_indices in enumerate(code_groups):
        group_ground_truth = ground_truth[:, tuple(code_indices)]
        for k in top_k_list:
            group_true_positive = true_positive[k][:, tuple(code_indices)]
            rate[k][f'ACC-P{i}-k{k}'] = group_true_positive.sum(
            ) / group_ground_truth.sum()
    return rate


class EvalFlag(Flag):
    CM = auto()
    POST = auto()

    @staticmethod
    def has(flag, attr):
        return (flag & attr).value != 0


def evaluation_table(raw_results, code_frequency_groups=None, top_k_list=[20]):
    evals = {colname: {} for colname in raw_results}
    flat_evals = {}
    for colname, res in raw_results.items():
        for rowname, val in res['loss'].items():
            evals[colname][rowname] = val
        for rowname, val in res['stats'].items():
            evals[colname][rowname] = val

        cm = compute_confusion_matrix(res['risk_prediction'])
        cm_scores = confusion_matrix_scores(cm)
        for rowname, val in cm_scores.items():
            evals[colname][rowname] = val

        for rowname, val in auc_scores(res['risk_prediction']).items():
            evals[colname][rowname] = val

        if code_frequency_groups is not None:
            det_topk_scores = top_k_detectability_scores(
                code_frequency_groups, res['risk_prediction'], top_k_list)
            for k in top_k_list:
                for rowname, val in det_topk_scores[k].items():
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


def codes_auc_pairwise_tests(results: Dict[str, BatchPredictedRisks],
                             fast=False):
    """
    Evaluate the AUC scores for each diagnosis code for each classifier. In addition,
    conduct a pairwise test on the difference of AUC scores between each
    pair of classifiers using DeLong test. Codes that have either less than two positive cases or
    have less than two negative cases are discarded (AUC computation and difference test requirements).
    """
    # Classifier labels
    clf_labels = list(sorted(results.keys()))

    def extract_subjects():
        example_risk_predictions = results[clf_labels[0]]
        subjects = set(example_risk_predictions.subject_risks.keys())
        assert all(
            set(other_risk_prediction.subject_risks.keys()) == subjects
            for other_risk_prediction in
            results.values()), "results should correspond to the same group"
        return list(sorted(subjects))

    subjects = extract_subjects()

    def extract_ground_truth_and_scores():
        ground_truth_mat = {}
        scores_mat = {}
        for clf_label in clf_labels:
            clf_ground_truth = []
            clf_scores = []
            clf_risk_prediction = results[clf_label]
            for subject_id in subjects:
                subj_pred_risks = clf_risk_prediction.subject_risks[subject_id]
                for index in sorted(subj_pred_risks):
                    risk_prediction = subj_pred_risks[index]
                    clf_ground_truth.append(risk_prediction.ground_truth)
                    clf_scores.append(risk_prediction.prediction)
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
