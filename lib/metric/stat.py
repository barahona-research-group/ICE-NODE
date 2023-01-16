"""Performance metrics and loss functions."""

from enum import Flag, auto
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from collections import namedtuple
from absl import logging
from tqdm import tqdm
from abc import ABC, abstractmethod, ABCMeta

import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from sklearn import metrics

from ..ehr import AbstractScheme, OutcomeExtractor
from ..ehr import Subject_JAX
from .delong import DeLongTest, FastDeLongTest
from .risk import BatchPredictedRisks


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

    def _reject_auc(ground_truth):
        # Reject if all positives (1) or all negatives (0).
        n_pos = ground_truth.sum()
        return (n_pos == 0 or n_pos == onp.size(ground_truth))

    if _reject_auc(v_truth):
        return float('nan')
    """Compute the area under the ROC from the ground-truth `v_truth` and
     the predictions `v_preds`."""
    fpr, tpr, _ = metrics.roc_curve(v_truth.flatten(),
                                    v_preds.flatten(),
                                    pos_label=1)
    return metrics.auc(fpr, tpr)


@dataclass
class Metric(metaclass=ABCMeta):
    subject_interface: Subject_JAX

    @classmethod
    def fields(cls):
        return tuple()

    @classmethod
    def classname(cls):
        return cls.__name__

    @classmethod
    def column(cls, field):
        return f'{cls.classname()}.{field}'

    @classmethod
    def columns(cls):
        return tuple(map(cls.column, cls.fields()))

    @abstractmethod
    def row(self, predictions: BatchPredictedRisks) -> Tuple[float]:
        pass


class VisitsAUC(Metric):

    @classmethod
    def fields(cls):
        return ('macro_auc', 'micro_auc')

    def row(self, predictions: BatchPredictedRisks) -> Tuple[float]:
        gtruth = []
        preds = []
        for subject_risks in predictions.values():
            for risk in subject_risks.values():
                gtruth.append(risk.ground_truth)
                preds.append(risk.prediction)

        gtruth_mat = onp.hstack(gtruth)
        preds_mat = onp.hstack(preds)
        macro_auc = compute_auc(gtruth_mat, preds_mat)
        micro_auc = onp.array(list(map(compute_auc, gtruth, preds)))

        return (macro_auc, onp.nanmean(micro_auc))


@dataclass
class CodeLevelMetric(Metric):

    index2code: Dict[int, str]
    code2index: Dict[str, int]

    def __init__(self, subject_interface: Subject_JAX):
        super().__init__(subject_interface)
        self.code2index = subject_interface.dx_outcome_extractor.index
        self.index2code = {i: c for c, i in self.code2index.items()}

    @classmethod
    def fields(cls):
        return tuple()

    def code_qualifier(self, code_index):
        return f'I{code_index}C{self.index2code[code_index]}'

    def column(self, code_index, field):
        clsname = self.classname()
        return f'{clsname}.{self.code_qualifier(code_index)}.{field}'

    def order(self):
        for index in sorted(self.index2code):
            for field in self.fields():
                yield (index, field)

    def columns(self):
        return tuple(self.column(idx, field) for idx, field in self.order())


class CodeAUC(CodeLevelMetric):

    @classmethod
    def fields(cls):
        return ('auc', 'n')

    def row(self, predictions: BatchPredictedRisks):
        ground_truth = []
        preds = []

        for subj_id, admission_risks in predictions.items():
            for admission_idx in sorted(admission_risks):
                risk = admission_risks[admission_idx]
                ground_truth.append(risk.ground_truth)
                preds.append(risk.prediction)

        ground_truth_mat = onp.vstack(ground_truth)
        predictions_mat = onp.vstack(preds)

        vals = {'auc': [], 'n': []}
        for code_index in range(ground_truth_mat.shape[1]):
            code_ground_truth = ground_truth_mat[:, code_index]
            code_predictions = predictions_mat[:, code_index]
            vals['n'].append(code_ground_truth.sum())
            vals['auc'].append(compute_auc(code_ground_truth,
                                           code_predictions))

        return tuple(vals[field][index] for index, field in self.order())


class UntilFirstCodeAUC(CodeAUC):

    def row(self, predictions: BatchPredictedRisks):
        ground_truth = []
        preds = []
        masks = []

        for subj_id, admission_risks in predictions.items():
            # first_occ is a vector of admission indices where the code at the corresponding
            # coordinate has first appeard for the patient. The index value of
            # (-1) means that the code will not appear at all for this patient.
            first_occ = self.subject_interface.code_first_occurrence(subj_id)

            for admission_id in sorted(admission_risks):
                mask = (admission_id <= first_occ)
                risk = admission_risks[admission_id]
                ground_truth.append(risk.ground_truth)
                preds.append(risk.prediction)
                masks.append(mask)

        ground_truth_mat = onp.vstack(ground_truth)
        predictions_mat = onp.vstack(preds)
        mask_mat = onp.vstack(masks).astype(bool)

        vals = {'n': [], 'auc': []}
        for code_index in range(ground_truth_mat.shape[1]):
            code_mask = mask_mat[:, code_index]
            code_ground_truth = ground_truth_mat[code_mask, code_index]
            code_predictions = predictions_mat[code_mask, code_index]
            vals['n'].append(code_mask.sum())
            vals['auc'].append(compute_auc(code_ground_truth,
                                           code_predictions))

        return tuple(vals[field][index] for index, field in self.order())


class AdmissionLevelAUC(Metric):

    @classmethod
    def fields(cls):
        return ('auc', )

    def admission_qualifier(self, subject_id, admission_id):
        return f'S{subject_id}A{admission_id}'

    def column(self, subject_id, admission_id, field):
        clsname = self.classname()
        return f'{clsname}.{self.admission_qualifier(subject_id, admission_id)}.{field}'

    @classmethod
    def order(cls, predictions):
        for subject_id in sorted(predictions):
            subject_predictions = predictions[subject_id]
            for admission_id in sorted(subject_predictions):
                for field in cls.fields():
                    yield (subject_id, admission_id, field)

    def columns(self, predictions):
        return tuple(self.column(*o) for o in self.order(predictions))

    def row(self, predictions: BatchPredictedRisks):
        auc = {}
        for subject_id in sorted(predictions):
            subject_predictions = predictions[subject_id]
            subject_auc = {}
            for admission_id in enumerate(sorted(subject_predictions)):
                prediction = subject_predictions[admission_id]
                auc_score = compute_auc(prediction.ground_truth,
                                        prediction.prediction)
                subject_auc[admission_id] = auc_score
            auc[subject_id] = subject_auc

        return tuple(auc[s_i][a_i] for s_i, a_i, _ in self.order(predictions))


@dataclass
class CodeGroupTopAlarmAccuracy(Metric):

    code_groups: List[List[int]]
    top_k_list: List[int]

    @classmethod
    def fields(cls):
        return ('acc', )

    def column(self, group_index, k, field):
        f'G{group_index}k{k}.{field}'

    def order(self):
        for k in self.top_k_list:
            for gi in range(len(self.code_groups)):
                for field in self.fields():
                    yield gi, k, field

    def columns(self):
        tuple(self.column(gi, k, f) for gi, k, f in self.order())

    def row(self, predictions: BatchPredictedRisks):
        top_k_list = sorted(self.top_k_list)

        ground_truth = []
        preds = []

        for subject_risks in predictions.values():
            for pred in subject_risks.values():
                preds.append(pred.prediction)
                ground_truth.append(pred.ground_truth)

        preds = onp.vstack(preds)
        ground_truth = onp.vstack(ground_truth).astype(bool)
        topk_risks = onp.argpartition(preds * -1, top_k_list, axis=1)

        true_positive = {}
        for k in top_k_list:
            topk_risks_i = topk_risks[:, :k]
            topk_risks_k = onp.zeros_like(preds, dtype=bool)
            onp.put_along_axis(topk_risks_k, topk_risks_i, True, 1)
            true_positive[k] = (topk_risks_k & ground_truth)

        alarm_acc = {}
        for gi, code_indices in enumerate(self.code_groups):
            group_true = ground_truth[:, tuple(code_indices)]
            group_alarm_acc = {}
            for k in top_k_list:
                group_tp = true_positive[k][:, tuple(code_indices)]
                group_alarm_acc[k] = group_tp.sum() / group_true.sum()
            alarm_acc[gi] = group_alarm_acc

        return tuple(alarm_acc[gi][k] for gi, k, _ in self.order())


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
        subjects = set(example_risk_predictions.keys())
        assert all(
            set(other_risk_prediction.keys()) == subjects
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
                subj_pred_risks = clf_risk_prediction[subject_id]
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