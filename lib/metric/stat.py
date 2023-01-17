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
from sklearn import metrics

from ..ehr import AbstractScheme
from ..ehr import Subject_JAX
from .delong import FastDeLongTest
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

    def column(self, field):
        return f'{self.classname()}.{field}'

    def columns(self):
        return tuple(map(self.column, self.fields()))

    @abstractmethod
    def __call__(self, predictions: BatchPredictedRisks):
        pass

    def row(self, result: Dict[str, Any]) -> Tuple[float]:
        return tuple(map(result.get, self.fields()))

    def to_dict(self, predictions: BatchPredictedRisks):
        result = self(predictions)
        return dict(zip(self.columns(), self.row(result)))

    def to_df(self, index: int, predictions: BatchPredictedRisks):
        return pd.DataFrame(self.to_dict(predictions), index=[index])


class VisitsAUC(Metric):

    @classmethod
    def fields(cls):
        return ('macro_auc', 'micro_auc')

    def __call__(self, predictions: BatchPredictedRisks) -> Tuple[float]:
        gtruth = []
        preds = []
        for subject_risks in predictions.values():
            for risk in subject_risks.values():
                gtruth.append(risk.ground_truth)
                preds.append(risk.prediction)

        gtruth_mat = onp.hstack(gtruth)
        preds_mat = onp.hstack(preds)
        return {
            'macro_auc': compute_auc(gtruth_mat, preds_mat),
            'micro_auc': onp.array(list(map(compute_auc, gtruth, preds)))
        }


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

    def row(self, result: Dict[str, Dict[int, float]]):
        return tuple(result[field][index] for index, field in self.order())


class CodeAUC(CodeLevelMetric):

    @classmethod
    def fields(cls):
        return ('auc', 'n')

    def __call__(self, predictions: BatchPredictedRisks):
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
        return vals


class UntilFirstCodeAUC(CodeAUC):

    def __call__(self, predictions: BatchPredictedRisks):
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
        return vals


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

    def __call__(self, predictions: BatchPredictedRisks):
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
        return {'auc': auc}

    def row(self, predictions: BatchPredictedRisks):
        res = self(predictions)
        return tuple(res[f][s][a] for s, a, f in self.order(predictions))

    def to_dict(self, predictions: BatchPredictedRisks):
        return dict(zip(self.columns(predictions), self.row(predictions)))


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

    def __call__(self, predictions: BatchPredictedRisks):
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
        return {'acc': alarm_acc}

    def row(self, result: Dict[str, Dict[int, Dict[int, float]]]):
        return tuple(result[f][gi][k] for gi, k, f in self.order())


class DeLongTest(CodeLevelMetric):

    @classmethod
    def fields(cls):
        return {
            'code': ('n_pos', ),
            'model': ('auc', 'auc_var'),
            'pair': ('p_val', )
        }

    @classmethod
    def fields_category(cls):
        return ('code', 'model', 'pair')

    def code_qualifier(self, code_index):
        return f'I{code_index}C{self.index2code[code_index]}'

    def column(self):

        def code_col(self, code_index, field):
            clsname = self.classname()
            return f'{clsname}.{self.code_qualifier(code_index)}.{field}'

        def model_col(self, code_index, model, field):
            clsname = self.classname()
            return f'{clsname}.{self.code_qualifier(code_index)}.{model}.{field}'

        def pair_col(self, code_index, model_pair, field):
            clsname = self.classname()
            m1, m2 = model_pair
            return f'{clsname}.{self.code_qualifier(code_index)}.{m1}={m2}.{field}'

        return {'code': code_col, 'model': model_col, 'pair': pair_col}

    def columns(self, clfs: List[str]):
        cols = []
        order_gen = self.order(clfs)
        for cat in self.fields_category():
            col_f = self.column()[cat]
            cols.append(tuple(col_f(*args) for args in order_gen[cat]()))
        return sum(cols, tuple())

    def _row(self, data: Dict[str, Any], clfs: List[str]):
        order_gen = self.order(clfs)

        def code_row():
            d = data['code']
            return tuple(d[f][idx] for idx, f in order_gen['code']())

        def model_row():
            d = data['model']
            return tuple(d[f][idx][clf]
                         for idx, clf, f in order_gen['model']())

        def pair_row():
            d = data['pair']
            return tuple(d[f][idx][clfp]
                         for idx, clfp, f in order_gen['pair']())

        return {'code': code_row, 'model': model_row, 'pair': pair_row}

    def row(self, predictions: Dict[str, BatchPredictedRisks]):
        data = self(predictions)
        clfs = sorted(predictions)
        row_gen = self._row(data, clfs)
        return sum((row_gen[cat]() for cat in self.fields_category()), tuple())

    @classmethod
    def model_pairs(cls, clfs: List[str]):
        clf_pairs = []
        for i in range(len(clfs)):
            for j in range(i + 1, len(clfs)):
                clf_pairs.append((clfs[i], clfs[j]))
        return tuple(clf_pairs)

    def order(self, clfs: List[str]):

        def model_order():
            fields = self.fields()['model']
            for index in sorted(self.index2code):
                for model in clfs:
                    for field in fields:
                        yield (index, model, field)

        def pair_order():
            pairs = self.model_pairs(clfs)
            fields = self.fields()['pair']
            for index in sorted(self.index2code):
                for model_pair in pairs:
                    for field in fields:
                        yield (index, model_pair, field)

        def code_order():
            fields = self.fields()['code']
            for index in sorted(self.index2code):
                for field in fields:
                    yield index, field

        return {'pair': pair_order, 'code': code_order, 'model': model_order}

    @classmethod
    def extract_subjects(cls, predictions: Dict[str, BatchPredictedRisks]):
        subject_sets = [preds.get_subjects() for preds in predictions.values()]
        for s1, s2 in zip(subject_sets[:-1], subject_sets[1:]):
            assert set(s1) == set(s2), "Subjects mismatch across model outputs"
        return list(sorted(subject_sets[0]))

    @classmethod
    def extract_grountruth_vs_predictions(cls, predictions):
        subjects = cls.extract_subjects(predictions)
        true_mat = {}
        pred_mat = {}
        for clf_label, clf_preds in predictions.items():
            clf_true = []
            clf_pred = []
            for subject_id in subjects:
                subj_preds = clf_preds[subject_id]
                for adm_index in sorted(subj_preds):
                    pred = subj_preds[adm_index]
                    clf_true.append(pred.ground_truth)
                    clf_pred.append(pred.prediction)
            true_mat[clf_label] = onp.vstack(clf_true)
            pred_mat[clf_label] = onp.vstack(clf_pred)

        tm0 = list(true_mat.values())[0]
        for tm in true_mat.values():
            assert (tm0 == tm).all(), "Mismatch in ground-truth across models."

        return tm0, pred_mat

    def __call__(self, predictions: Dict[str, BatchPredictedRisks]):
        """
        Evaluate the AUC scores for each diagnosis code for each classifier. In addition,
        conduct a pairwise test on the difference of AUC scores between each
        pair of classifiers using DeLong test. Codes that have either less than two positive cases or
        have less than two negative cases are discarded (AUC computation and difference test requirements).
        """
        # Classifier labels
        clf_labels = list(sorted(predictions.keys()))
        true_mat, preds = self.extract_grountruth_vs_predictions(predictions)
        clf_pairs = self.model_pairs(clf_labels)

        n_pos = {}  # only codewise
        auc = {}  # modelwise
        auc_var = {}  # modelwise
        p_val = {}  # pairwise

        for code_index in tqdm(range(true_mat.shape[1])):
            code_truth = true_mat[:, code_index]
            n_pos = code_truth.sum()
            n_neg = len(code_truth) - n_pos
            n_pos[code_index] = n_pos
            # This is a requirement for pairwise testing.
            invalid_test = n_pos < 2 or n_neg < 2

            # Since we iterate on pairs, some AUC are computed more than once.
            # Update this dictionary for each AUC computed, then append the results
            # to the big list of auc.
            code_auc = {}
            code_auc_var = {}
            code_p_val = {}
            for (clf1, clf2) in clf_pairs:
                if invalid_test:
                    code_p_val[(clf1, clf2)] = float('nan')
                    for clf in (clf1, clf2):
                        code_auc[clf] = code_auc.get(clf, float('nan'))
                        code_auc_var[clf] = code_auc_var.get(clf, float('nan'))
                    continue

                preds1 = preds[clf1][:, code_index]
                preds2 = preds[clf2][:, code_index]
                auc1, auc2, auc1_v, auc2_v, p = FastDeLongTest.delong_roc_test(
                    code_truth, preds1, preds2)

                code_p_val[(clf1, clf2)] = p
                code_auc[clf1] = auc1
                code_auc[clf2] = auc2
                code_auc_var[clf1] = auc1_v
                code_auc_var[clf2] = auc2_v

            auc[code_index] = code_auc
            auc_var[code_index] = code_auc_var
            p_val[code_index] = code_p_val

        return {
            'pair': {
                'p_val': p_val
            },
            'code': {
                'n_pos': n_pos
            },
            'model': {
                'auc': auc,
                'auc_var': auc_var
            }
        }
