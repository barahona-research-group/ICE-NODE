"""Performance metrics and loss functions."""

from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod, ABCMeta

from tqdm import tqdm
from absl import logging
import pandas as pd
import numpy as onp
import jax
import jax.numpy as jnp
from sklearn import metrics

from ..ehr import Subject_JAX, BatchPredictedRisks
from .delong import FastDeLongTest
from . import loss as L


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

    @staticmethod
    def fields():
        return tuple()

    @staticmethod
    def dirs():
        return tuple()

    @classmethod
    def field_dir(cls, field):
        return dict(zip(cls.fields(), cls.dirs()))[field]

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

    def value_extractor(self, keys: Dict[str, str]):
        field_name = keys.get('field', self.fields()[0])
        colname = self.column(field_name)
        return self.from_df_functor(colname, self.field_dir(field_name))

    def from_df_functor(self, colname, direction):

        def from_df(df, index=-1):
            if index == 'best':
                if direction == 1:
                    return df[colname].argmax(), df[colname].max()
                else:
                    return df[colname].argmin(), df[colname].min()
            else:
                return df[colname].iloc[index]

        return from_df


class VisitsAUC(Metric):

    @staticmethod
    def fields():
        return ('macro_auc', 'micro_auc')

    @staticmethod
    def dirs():
        return (1, 1)

    @classmethod
    def field_dir(cls, field):
        return dict(zip(cls.fields(), cls.dirs()))[field]

    def __call__(self, predictions: BatchPredictedRisks) -> Tuple[float]:
        gtruth = []
        preds = []
        for subject_risks in predictions.values():
            for risk in subject_risks.values():
                gtruth.append(risk.get_outcome())
                preds.append(risk.prediction)

        gtruth_vec = onp.hstack(gtruth)
        preds_vec = onp.hstack(preds)
        return {
            'macro_auc':
            compute_auc(gtruth_vec, preds_vec),
            'micro_auc':
            onp.nanmean(onp.array(list(map(
                compute_auc,
                gtruth,
                preds,
            ))))
        }


@dataclass
class LossMetric(Metric):
    loss_functions: Dict[str, Callable] = field(init=False)

    def __post_init__(self):
        self.loss_functions = {
            'softmax': L.softmax_logits_bce,
            'weighted_softmax': L.softmax_logits_weighted_bce,
            'focal_softmax': L.softmax_logits_balanced_focal_bce,
            'focal_bce': L.balanced_focal_bce
        }

    def fields(self):
        return sorted(self.loss_functions.keys())

    def dirs(self):
        return (0, ) * len(self.loss_functions)

    def field_dir(self, field):
        return 0

    def __call__(self, predictions: BatchPredictedRisks):
        return {
            loss_key: float(predictions.prediction_loss(loss_f))
            for loss_key, loss_f in self.loss_functions.items()
        }


@dataclass
class CodeLevelMetric(Metric):
    index2code: Dict[int, str] = field(init=False)
    code2index: Dict[str, int] = field(init=False)
    # Show estimates per code
    code_level: bool = field(default=True)

    # Show estimates aggregated over all codes.
    aggregate_level: bool = field(default=True)

    def __post_init__(self):
        self.code2index = self.subject_interface.outcome_extractor.index
        self.index2code = {i: c for c, i in self.code2index.items()}

    @staticmethod
    def agg_fields():
        return (('mean', onp.nanmean), ('median', onp.nanmedian),
                ('max', onp.nanmax), ('min', onp.nanmin))

    def code_qualifier(self, code_index):
        return f'I{code_index}C{self.index2code[code_index]}'

    def column(self, code_index, field):
        clsname = self.classname()
        return f'{clsname}.{self.code_qualifier(code_index)}.{field}'

    @classmethod
    def agg_column(cls, agg_key, field):
        return f'{cls.classname()}.{agg_key}({field})'

    def agg_columns(self):
        clsname = self.classname()
        cols = []
        for field in self.fields():
            for agg_k, _ in self.agg_fields():
                cols.append(self.agg_column(agg_k, field))
        return tuple(cols)

    def order(self):
        for index in sorted(self.index2code):
            for field in self.fields():
                yield (index, field)

    def columns(self):
        cols = []
        if self.code_level:
            cols.append(
                tuple(self.column(idx, field) for idx, field in self.order()))
        if self.aggregate_level:
            cols.append(self.agg_columns())

        return sum(tuple(cols), tuple())

    def agg_row(self, result: Dict[str, Dict[int, float]]):
        row = []
        for field in self.fields():
            field_vals = onp.array(list(result[field].values()))
            for _, agg_f in self.agg_fields():
                row.append(agg_f(field_vals))
        return tuple(row)

    def row(self, result: Dict[str, Dict[int, float]]):
        row = []
        if self.code_level:
            row.append(
                tuple(result[field][index] for index, field in self.order()))
        if self.aggregate_level:
            row.append(self.agg_row(result))

        return sum(tuple(row), tuple())

    def value_extractor(self, keys):
        code_index = keys.get('code_index')
        code = keys.get('code')
        assert (code_index is None) != (
            code is
            None), "providing code and code_index are mutually exlusive"
        code_index = self.code2index[code] if code is not None else code_index
        column = self.column(code_index, keys['field'])
        return self.from_df_functor(column, self.field_dir(keys['field']))

    def aggregate_extractor(self, keys):
        agg = keys['aggregate']
        column = self.agg_column(agg, keys.get('field', self.fields()[0]))
        return self.from_df_functor(column, self.field_dir(keys['field']))


@dataclass
class CodeAUC(CodeLevelMetric):

    @staticmethod
    def fields():
        return ('auc', 'n')

    @staticmethod
    def dirs():
        return (1, 1)

    def __call__(self, predictions: BatchPredictedRisks):
        ground_truth = []
        preds = []

        for subj_id, admission_risks in predictions.items():
            for admission_idx in sorted(admission_risks):
                risk = admission_risks[admission_idx]
                ground_truth.append(risk.get_outcome())
                preds.append(risk.prediction)

        ground_truth_mat = onp.vstack(ground_truth)
        predictions_mat = onp.vstack(preds)

        vals = {'auc': {}, 'n': {}}
        for code_index in range(ground_truth_mat.shape[1]):
            code_ground_truth = ground_truth_mat[:, code_index]
            code_predictions = predictions_mat[:, code_index]
            vals['n'][code_index] = code_ground_truth.sum()
            vals['auc'][code_index] = compute_auc(code_ground_truth,
                                                  code_predictions)
        return vals


@dataclass
class UntilFirstCodeAUC(CodeAUC):

    def __call__(self, predictions: BatchPredictedRisks):
        ground_truth = []
        preds = []
        masks = []

        for subj_id, admission_risks in predictions.items():
            # first_occ is a vector of admission indices where the code at the corresponding
            # coordinate has first appeard for the patient. the index value of
            # (-1) means that the code will not appear at all for this patient.
            first_occ = self.subject_interface.code_first_occurrence(subj_id)

            for admission_id in sorted(admission_risks):
                mask = (admission_id <= first_occ)
                risk = admission_risks[admission_id]
                ground_truth.append(risk.get_outcome())
                preds.append(risk.prediction)
                masks.append(mask)

        ground_truth_mat = onp.vstack(ground_truth)
        predictions_mat = onp.vstack(preds)
        mask_mat = onp.vstack(masks).astype(bool)

        vals = {'n': {}, 'auc': {}}
        for code_index in range(ground_truth_mat.shape[1]):
            code_mask = mask_mat[:, code_index]
            code_ground_truth = ground_truth_mat[code_mask, code_index]
            code_predictions = predictions_mat[code_mask, code_index]
            vals['n'][code_index] = code_mask.sum()
            vals['auc'][code_index] = compute_auc(code_ground_truth,
                                                  code_predictions)
        return vals


@dataclass
class AdmissionAUC(Metric):

    # Show estimates for each admission for each subject (extremely large
    # table)
    admission_level: bool = field(default=False)

    # Show estimates aggregated on the subject level (very large table)
    subject_aggregate_level: bool = field(default=False)

    # Show estimates aggregated across the entire subjects and admissions.
    aggregate_level: bool = field(default=True)

    @staticmethod
    def fields():
        return ('auc', )

    @staticmethod
    def dirs():
        return (1, )

    @staticmethod
    def agg_fields():
        return (('mean', onp.nanmean), ('median', onp.nanmedian),
                ('max', onp.nanmax), ('min', onp.nanmin))

    @staticmethod
    def subject_qualifier(subject_id):
        return f'S{subject_id}'

    @classmethod
    def admission_qualifier(cls, subject_id, admission_id):
        return f'{cls.subject_qualifier(subject_id)}A{admission_id}'

    @classmethod
    def column(cls, subject_id, admission_id, field):
        clsname = cls.classname()
        return f'{clsname}.{cls.admission_qualifier(subject_id, admission_id)}.{field}'

    @classmethod
    def subject_agg_column(cls, subject_id, agg_key, field):
        return f'{cls.classname()}.{cls.subject_qualifier(subject_id)}.{agg_key}({field})'

    @classmethod
    def subject_agg_columns(cls, subject_order_gen):
        clsname = cls.classname()
        cols = []
        for subject_id in subject_order_gen():
            for field in cls.fields():
                for agg_k, _ in cls.agg_fields():
                    cols.append(
                        cls.subject_agg_column(subject_id, agg_k, field))
        return tuple(cols)

    @classmethod
    def agg_column(cls, agg_key, field):
        return f'{cls.classname()}.{agg_key}({field})'

    @classmethod
    def agg_columns(cls):
        cols = []
        for field in cls.fields():
            for agg_k, _ in cls.agg_fields():
                cols.append(cls.agg_column(agg_k, field))
        return tuple(cols)

    @classmethod
    def ordered_subjects(cls, predictions):
        return sorted(predictions)

    @classmethod
    def order(cls, predictions):
        for subject_id in cls.ordered_subjects(predictions):
            subject_predictions = predictions[subject_id]
            for admission_id in sorted(subject_predictions):
                for field in cls.fields():
                    yield (subject_id, admission_id, field)

    def columns(self, order_gen, subject_order_gen):
        cols = []
        if self.admission_level:
            cols.append(tuple(self.column(*o) for o in order_gen()))

        if self.subject_aggregate_level:
            cols.append(self.subject_agg_columns(subject_order_gen))

        if self.aggregate_level:
            cols.append(self.agg_columns())

        return sum(tuple(cols), tuple())

    def __call__(self, predictions: BatchPredictedRisks):
        auc = {}
        for subject_id in sorted(predictions):
            subject_predictions = predictions[subject_id]
            subject_auc = {}
            for admission_id in sorted(subject_predictions):
                prediction = subject_predictions[admission_id]
                auc_score = compute_auc(prediction.get_outcome(),
                                        prediction.prediction)
                subject_auc[admission_id] = auc_score
            auc[subject_id] = subject_auc
        return {'auc': auc}

    def subject_agg_row(self, result, subject_order_gen):
        row = []
        for subject_id in subject_order_gen():
            for field in self.fields():
                field_data = onp.array(list(
                    result[field][subject_id].values()))
                for _, agg_f in self.agg_fields():
                    row.append(agg_f(field_data))
        return tuple(row)

    def agg_row(self, result):
        row = []
        for field in self.fields():
            fdata = result[field]
            data = list(v for sdata in fdata.values() for v in sdata.values())
            data = onp.array(data)
            for _, agg_f in self.agg_fields():
                row.append(agg_f(data))
        return tuple(row)

    def row(self, result: Dict[str, Any], order_gen, subject_order_gen):
        row = []
        if self.admission_level:
            row.append(tuple(result[f][s][a] for s, a, f in order_gen()))
        if self.subject_aggregate_level:
            row.append(self.subject_agg_row(result, subject_order_gen))
        if self.aggregate_level:
            row.append(self.agg_row(result))
        return sum(tuple(row), tuple())

    def to_dict(self, predictions: BatchPredictedRisks):
        order_gen = lambda: self.order(predictions)
        subject_order_gen = lambda: self.ordered_subjects(predictions)
        result = self(predictions)
        cols = self.columns(order_gen, subject_order_gen)
        rows = self.row(result, order_gen, subject_order_gen)
        return dict(zip(cols, rows))

    def value_extractor(self, keys):
        subject_id = keys['subject_id']
        admission_id = keys['admission_id']
        field = keys.get('field', self.fields()[0])
        column = self.column(subject_id, admission_id, field)
        return self.from_df_functor(column, self.field_dir(field))

    def subject_aggregate_extractor(self, keys):
        field = keys.get('field', self.fields()[0])
        column = self.subject_agg_column(keys['subject_id'], keys['aggregate'],
                                         field)
        return self.from_df_functor(column, self.field_dir(field))

    def aggregate_extractor(self, keys):
        agg = keys['aggregate']
        field = keys.get('field', self.fields()[0])
        column = self.agg_column(agg, field)
        return self.from_df_functor(column, self.field_dir(field))


@dataclass
class CodeGroupTopAlarmAccuracy(Metric):

    code_groups: List[List[int]]
    top_k_list: List[int]

    @staticmethod
    def fields():
        return ('acc', )

    @staticmethod
    def dirs():
        return (1, )

    def column(self, group_index, k, field):
        return f'G{group_index}k{k}.{field}'

    def order(self):
        for k in self.top_k_list:
            for gi in range(len(self.code_groups)):
                for field in self.fields():
                    yield gi, k, field

    def columns(self):
        return tuple(self.column(gi, k, f) for gi, k, f in self.order())

    def __call__(self, predictions: BatchPredictedRisks):
        top_k_list = sorted(self.top_k_list)

        ground_truth = []
        preds = []

        for subject_risks in predictions.values():
            for pred in subject_risks.values():
                preds.append(pred.prediction)
                ground_truth.append(pred.get_outcome())

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

    def value_extractor(self, keys: Dict[str, str]):
        k = keys['k']
        gi = keys['group_index']
        assert k in self.top_k_list, f"k={k} is not in {self.top_k_list}"
        assert gi < len(
            self.code_groups), f"group_index ({gi}) >= len(self.code_groups)"
        field = keys.get('field', self.fields()[0])
        colname = self.column(gi, k, field)
        return self.from_df_functor(colname, self.field_dir(field))


@dataclass
class OtherMetrics(Metric):

    def __init__(self):
        super().__init__(None)

    def __call__(self, some_flat_dict: Dict[str, float]):
        return some_flat_dict

    def row(self, some_flat_dict: Dict[str, float]) -> Tuple[float]:
        return tuple(map(some_flat_dict.get, sorted(some_flat_dict)))

    def to_dict(self, some_flat_dict):
        return some_flat_dict

    def to_df(self, index: int, some_flat_dict: Dict[str, float]):
        some_flat_dict = {
            k: v.item() if hasattr(v, 'item') else v
            for k, v in some_flat_dict.items()
        }
        return pd.DataFrame(some_flat_dict, index=[index])


@dataclass
class MetricsCollection:
    metrics: List[Metric]
    other_metrics: OtherMetrics = OtherMetrics()

    def to_df(self,
              iteration: int,
              predictions: BatchPredictedRisks,
              other_estimated_metrics: Dict[str, float] = None):

        dfs = [m.to_df(iteration, predictions) for m in self.metrics]

        if other_estimated_metrics:
            dfs.append(
                self.other_metrics.to_df(iteration, other_estimated_metrics))
        return pd.concat(dfs, axis=1)


@dataclass
class DeLongTest(CodeLevelMetric):

    # def __post_init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    @staticmethod
    def fields():
        return {
            'code': ('n_pos', ),
            'model': ('auc', 'auc_var'),
            'pair': ('p_val', )
        }

    @staticmethod
    def fields_category():
        return ('code', 'model', 'pair')

    @staticmethod
    def column():

        def code_col(field):
            return f'{field}'

        def model_col(model, field):
            return f'{model}.{field}'

        def pair_col(model_pair, field):
            m1, m2 = model_pair
            return f'{m1}={m2}.{field}'

        return {'code': code_col, 'model': model_col, 'pair': pair_col}

    def value_extractor(self,
                        field: str,
                        code_index: Optional[int] = None,
                        code: Optional[str] = None):
        """
        Makes an extractor function to retrieve the estimated value.
        Each estimand by this metric corresponds to a particular code.
        The keys object therefore must include either 'code' with actual code
        value or 'code_index' with the corresponding index.
        The keys should have the 'field' key, with one of the following values:
            - 'n_pos': to extract number of positive cases of the AUC problem.
            - 'auc': to extract the AUC exhibited by a particular model. In
            this case keys should also have the 'model' key for the model of interest.
            - 'auc_var': same as 'auc' but returns the estimated variance of
            the AUC.
            - 'p_val': to extract the p-value of the equivalence test between
            the AUC of two models, hence the two models should be included in the keys
            as {'pairs': (model1, model2), ....}.
        """
        assert (code_index is None) != (
            code is
            None), "providing code and code_index are mutually exlusive"
        code_index = self.code2index[code] if code is not None else code_index

        if field in self.fields()['model']:
            column_gen = lambda kw: self.column()['model'](kw['model'], field)
        elif field in self.fields()['code']:
            column_gen = lambda _: self.column()['code'](field)
        else:
            column_gen = lambda kw: self.column()['pair'](tuple(
                sorted(kw['pair'])), field)

        def from_df(df, **kw):
            return df.loc[code_index, column_gen(kw)]

        return from_df

    def columns(self, clfs: List[str]):
        cols = []
        order_gen = self.order(clfs)
        for cat in self.fields_category():
            col_f = self.column()[cat]
            cols.append(tuple(col_f(*t) for t in order_gen[cat]()))
        return sum(cols, tuple())

    def _row(self, code_index: int, data: Dict[str, Any], clfs: List[str]):
        order_gen = self.order(clfs)

        def code_row():
            d = data['code']
            return tuple(d[f][code_index] for (f, ) in order_gen['code']())

        def model_row():
            d = data['model']
            return tuple(d[f][code_index][clf]
                         for clf, f in order_gen['model']())

        def pair_row():
            d = data['pair']
            return tuple(d[f][code_index][clfp]
                         for clfp, f in order_gen['pair']())

        return {'code': code_row, 'model': model_row, 'pair': pair_row}

    def rows(self, data, clfs):
        rows = []
        for code_index in sorted(self.index2code):
            row_gen = self._row(code_index, data, clfs)
            rows.append(
                sum((row_gen[cat]() for cat in self.fields_category()),
                    tuple()))
        return rows

    def to_df(self, predictions: Dict[str, BatchPredictedRisks]):
        clfs = sorted(predictions)
        cols = self.columns(clfs)
        data = self(predictions)
        df = pd.DataFrame(columns=cols)
        for idx, row in enumerate(self.rows(data, clfs)):
            df.loc[idx] = row
        return df

    def to_dict(self, predictions: Dict[str, BatchPredictedRisks]):
        clfs = sorted(predictions)
        cols = self.columns(clfs)
        data = self(predictions)
        res = {}
        for idx, row in enumerate(self.rows(data, clfs)):
            res[idx] = dict(zip(cols, row))
        return res

    @classmethod
    def _model_pairs(cls, clfs: List[str]):
        clf_pairs = []
        for i in range(len(clfs)):
            for j in range(i + 1, len(clfs)):
                clf_pairs.append((clfs[i], clfs[j]))
        return tuple(clf_pairs)

    def order(self, clfs: List[str]):

        def model_order():
            fields = self.fields()['model']
            for model in clfs:
                for field in fields:
                    yield (model, field)

        def pair_order():
            pairs = self._model_pairs(clfs)
            fields = self.fields()['pair']
            for model_pair in pairs:
                for field in fields:
                    yield (model_pair, field)

        def code_order():
            fields = self.fields()['code']
            for field in fields:
                yield (field, )

        return {'pair': pair_order, 'code': code_order, 'model': model_order}

    @classmethod
    def _extract_subjects(cls, predictions: Dict[str, BatchPredictedRisks]):
        subject_sets = [preds.get_subjects() for preds in predictions.values()]
        for s1, s2 in zip(subject_sets[:-1], subject_sets[1:]):
            assert set(s1) == set(s2), "Subjects mismatch across model outputs"
        return list(sorted(subject_sets[0]))

    @classmethod
    def _extract_grountruth_vs_predictions(cls, predictions):
        subjects = cls._extract_subjects(predictions)
        true_mat = {}
        pred_mat = {}
        mask_mat = {}
        for clf_label, clf_preds in predictions.items():
            clf_true = []
            clf_pred = []
            clf_mask = []
            for subject_id in subjects:
                subj_preds = clf_preds[subject_id]
                for adm_index in sorted(subj_preds):
                    pred = subj_preds[adm_index]
                    clf_true.append(pred.get_outcome())
                    clf_mask.append(pred.get_mask())
                    clf_pred.append(pred.prediction)

            true_mat[clf_label] = onp.vstack(clf_true)
            pred_mat[clf_label] = onp.vstack(clf_pred)
            mask_mat[clf_label] = onp.vstack(clf_mask)

        tm0 = list(true_mat.values())[0]
        mask = list(mask_mat.values())[0]
        for tm, msk in zip(true_mat.values(), mask_mat.values()):
            assert (tm0 == tm).all(), "Mismatch in ground-truth across models."
            assert (mask == msk).all(), "Mismatch in mask across models."

        return tm0, mask, pred_mat

    def __call__(self, predictions: Dict[str, BatchPredictedRisks]):
        """
        Evaluate the AUC scores for each diagnosis code for each classifier. In addition,
        conduct a pairwise test on the difference of AUC scores between each
        pair of classifiers using DeLong test. Codes that have either less than two positive cases or
        have less than two negative cases are discarded (AUC computation and difference test requirements).
        """
        # Classifier labels
        clf_labels = list(sorted(predictions.keys()))
        true_mat, mask_mat, preds = self._extract_grountruth_vs_predictions(
            predictions)
        clf_pairs = self._model_pairs(clf_labels)

        n_pos = {}  # only codewise
        auc = {}  # modelwise
        auc_var = {}  # modelwise
        p_val = {}  # pairwise

        for code_index in tqdm(range(true_mat.shape[1])):
            mask = mask_mat[:, code_index]
            code_truth = true_mat[mask, code_index]
            _n_pos = code_truth.sum()
            n_neg = len(code_truth) - _n_pos
            n_pos[code_index] = _n_pos
            # This is a requirement for pairwise testing.
            invalid_test = _n_pos < 2 or n_neg < 2

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

                preds1 = preds[clf1][mask, code_index]
                preds2 = preds[clf2][mask, code_index]
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

    def filter_results(self, results: pd.DataFrame, models: List[str],
                       min_auc: float):
        models = sorted(models)
        m_pairs = self._model_pairs(models)

        # exclude tests with nans
        pval_cols = [self.column()['pair'](pair, 'p_val') for pair in m_pairs]
        valid_tests = results.loc[:, pval_cols].isnull().max(axis=1) == 0
        results = results[valid_tests]
        logging.info(
            f'{sum(valid_tests)}/{len(valid_tests)} rows with valid tests (no NaN p-values)'
        )

        # AUC cut-off
        auc_cols = [self.column()['model'](m, 'auc') for m in models]
        accepted_aucs = results.loc[:, auc_cols].max(axis=1) > min_auc
        results = results[accepted_aucs]
        logging.info(f'{sum(accepted_aucs)}/{len(accepted_aucs)} rows with\
            an AUC higher than {min_auc} by at least one model.')

        return results

    def insignificant_difference_rows(self, results: pd.DataFrame,
                                      models: List[str], p_value: float):
        models = sorted(models)
        m_pairs = self._model_pairs(models)
        pval_cols = [self.column()['pair'](pair, 'p_val') for pair in m_pairs]
        insigificant = results.loc[:, pval_cols].min(axis=1) > p_value
        return results[insigificant]
