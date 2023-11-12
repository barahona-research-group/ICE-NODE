"""Performance metrics and loss functions."""

from typing import Dict, Optional, List, Tuple, Any, Callable
from abc import abstractmethod, ABCMeta
from dataclasses import field, dataclass
import sys
import inspect
from collections import defaultdict

from tqdm import tqdm
from absl import logging
import pandas as pd
import numpy as onp
from scipy import stats
import jax
import jax.numpy as jnp
from sklearn import metrics
import warnings

from ..ehr import (Patients, Predictions, AdmissionPrediction,
                   InpatientObservables)
from ..base import Module, Config
from .delong import FastDeLongTest
from .loss import (binary_loss, numeric_loss, colwise_binary_loss,
                   colwise_numeric_loss)


def safe_nan_func(func, x, axis):
    """Apply `func` to `x` along `axis`, ignoring NaNs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return func(x, axis=axis)


def nanaverage(A, weights, axis):
    return safe_nan_func(lambda x, axis: onp.nansum(x * weights, axis=axis) /
                         ((~onp.isnan(x)) * weights).sum(axis=axis),
                         A,
                         axis=axis)


def nanmean(A, axis=None):
    return safe_nan_func(onp.nanmean, A, axis=axis)


def nanmedian(A, axis=None):
    return safe_nan_func(onp.nanmedian, A, axis=axis)


def nanstd(A, axis=None):
    return safe_nan_func(onp.nanstd, A, axis=axis)


def nanmax(A, axis=None):
    return safe_nan_func(onp.nanmax, A, axis=axis)


def nanmin(A, axis=None):
    return safe_nan_func(onp.nanmin, A, axis=axis)


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


def nan_compute_auc(v_truth, v_preds):
    isnan = onp.isnan(v_truth) | onp.isnan(v_preds)
    v_truth = v_truth[~isnan]
    v_preds = v_preds[~isnan]
    npos = v_truth.sum()
    nneg = onp.size(v_truth) - npos
    return (npos, nneg), compute_auc(v_truth, v_preds)


class Metric(Module):
    config: Config = Config()
    patients: Patients

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._post_init()

    def _post_init(self):
        pass

    @staticmethod
    def fields():
        return tuple()

    @staticmethod
    def dirs():
        return tuple()

    def field_dir(self, field):
        return dict(zip(self.fields(), self.dirs()))[field]

    @classmethod
    def classname(cls):
        return cls.__name__

    def column(self, field):
        return f'{self.classname()}.{field}'

    def columns(self):
        return tuple(map(self.column, self.fields()))

    @abstractmethod
    def __call__(self, predictions: Predictions):
        pass

    def row(self, result: Dict[str, Any]) -> Tuple[float]:
        return tuple(map(result.get, self.fields()))

    def to_dict(self, predictions: Predictions):
        result = self(predictions)
        return dict(zip(self.columns(), self.row(result)))

    def to_df(self, index: int, predictions: Predictions):
        return pd.DataFrame(self.to_dict(predictions), index=[index])

    def value_extractor(self, keys: Dict[str, str]):
        field_name = keys.get('field', self.fields()[0])
        colname = self.column(field_name)
        return self.from_df_functor(colname, self.field_dir(field_name))

    def from_df_functor(self, colname, direction):

        def from_df(df, index=-1):
            if isinstance(df, Predictions):
                df = self.to_df(index, df)

            if index == 'best':
                if direction == 1:
                    return df[colname].argmax(), df[colname].max()
                else:
                    return df[colname].argmin(), df[colname].min()
            else:
                return df[colname].iloc[index]

        return from_df

    @classmethod
    def external_argnames(cls):
        return ['patients']


class VisitsAUC(Metric):

    @staticmethod
    def fields():
        return ('macro_auc', 'micro_auc')

    @staticmethod
    def dirs():
        return (1, 1)

    def __call__(self, predictions: Predictions) -> Tuple[float]:
        gtruth = []
        preds = []
        for patient_predictions in predictions.values():
            for p in patient_predictions.values():
                gtruth.append(p.admission.outcome.vec)
                preds.append(p.outcome.vec)

        gtruth_vec = onp.hstack(gtruth)
        preds_vec = onp.hstack(preds)
        return {
            'macro_auc':
            compute_auc(gtruth_vec, preds_vec),
            'micro_auc':
            nanmean(onp.array(list(map(
                compute_auc,
                gtruth,
                preds,
            ))))
        }


class LossMetricConfig(Config):
    dx_loss: List[str] = field(default_factory=list)
    obs_loss: List[str] = field(default_factory=list)
    lead_loss: List[str] = field(default_factory=list)


class LossMetric(Metric):
    _dx_loss: Dict[str, Callable]
    _obs_loss: Dict[str, Callable]
    _lead_loss: Dict[str, Callable]

    def __init__(self, patients, config=None, **kwargs):
        if config is None:
            config = LossMetricConfig()
        config = config.update(**kwargs)
        super().__init__(patients=patients, config=config)
        self._dx_loss = {name: binary_loss[name] for name in config.dx_loss}
        self._obs_loss = {name: numeric_loss[name] for name in config.obs_loss}
        self._lead_loss = {
            name: numeric_loss[name]
            for name in config.lead_loss
        }

    def dx_fields(self):
        return [f'dx_{name}' for name in sorted(self.config.dx_loss)]

    def obs_fields(self):
        return [f'obs_{name}' for name in sorted(self.config.obs_loss)]

    def lead_fields(self):
        return [f'lead_{name}' for name in sorted(self.config.lead_loss)]

    def fields(self):
        return self.dx_fields() + self.obs_fields() + self.lead_fields()

    def dirs(self):
        return (0, ) * len(self)

    def __len__(self):
        return len(self.fields())

    def __call__(self, predictions: Predictions):
        return {
            f'dx_{loss_key}': predictions.prediction_dx_loss(dx_loss=loss_f)
            for loss_key, loss_f in self._dx_loss.items()
        } | {
            f'obs_{loss_key}': predictions.prediction_obs_loss(obs_loss=loss_f)
            for loss_key, loss_f in self._obs_loss.items()
        } | {
            f'lead_{loss_key}':
            predictions.prediction_lead_loss(lead_loss=loss_f)
            for loss_key, loss_f in self._lead_loss.items()
        }


class AKISegmentedAdmissionConfig(Config):
    stable_window: int = 12


def nan_concat_lagging_windows(x):
    n = len(x)
    add_arr = onp.full(n - 1, onp.nan)
    x_ext = onp.concatenate((add_arr, x))
    strided = onp.lib.stride_tricks.as_strided
    nrows = len(x_ext) - n + 1
    s = x_ext.strides[0]
    return strided(x_ext, shape=(nrows, n), strides=(s, s))[:, :]


class AKISegmentedAdmissionMetric(Metric):

    def _label_aki_timestamps(self, prediction: AdmissionPrediction):
        """
        Label the observation timestamps into AKI and non-AKI.
        The label at a timestamp is:
            1. '1.0' (i.e. an AKI) if the patient has a nonzero AKI index at \
                that timestamp or within the preceding 'stable_window' hours.
            2. '0.0' (stable) if the patient has a zero AKI index at that \
                timestamp and within the preceding 'stable_window' hours.
            3. 'NaN' if there is no valid AKI index at that timestamp and \
                within the preceding 'stable_window' hours.

        Returns:
            a vector of length len(observation_timestamps) with 1 for AKI and 0
            for non-AKI.
        """

        leading_observable_config = self.patients.config.leading_observable
        aki_index = leading_observable_config.index
        observables = prediction.admission.observables
        time = onp.array(observables.time)
        mask = onp.array(observables.mask[:, aki_index])
        aki = onp.array(observables.value[:, aki_index])
        aki = onp.where(mask, aki, onp.nan)

        t_preced = nan_concat_lagging_windows(time)
        aki_preced = nan_concat_lagging_windows(aki)
        t_diff = t_preced[:, -1, None] - t_preced
        mask = t_diff <= self.config.stable_window

        aki_preced = onp.where(mask, (aki_preced > 0) * 1.0, onp.nan)

        aki_max = nanmax(aki_preced, axis=1)

        max_aki_pred = nanmax(prediction.leading_observable.value, axis=1)
        return time, aki_max, max_aki_pred

    def _segment_aki_labels(self, prediction: AdmissionPrediction):
        """
        Segment the AKI labels into periods of constant AKI or \
            constant non-AKI.

        Returns:
            a list of tuples (time, aki) where time is a vector of timestamps \
                and aki is the AKI label (1 for AKI, 0 for non-AKI, NaN for \
                unknown).
        """
        time, aki_labels, aki_preds = self._label_aki_timestamps(prediction)
        aki_labels = onp.nan_to_num(aki_labels, nan=-1)
        aki_jumps = onp.hstack((0, aki_labels[1:] - aki_labels[:-1]))
        aki_jumps_idx = onp.argwhere(aki_jumps != 0).flatten().tolist()
        indices = [0] + aki_jumps_idx + [len(aki_labels)]
        data = []
        for i1, i2 in zip(indices[:-1], indices[1:]):
            aki = aki_labels[i1]
            if aki == -1:
                aki = onp.nan

            data.append({
                'i1':
                i1,
                'i2':
                i2,
                'admission_id':
                prediction.admission.admission_id,
                'time':
                time[i1:i2],
                'next_interval_time':
                time[i2] if i2 < len(time) else onp.nan,
                'aki_label':
                aki_labels[i1],
                'aki_preds':
                aki_preds[i1:i2]
            })

        return data

    def _classify_aki_segments(self, segmented_AKI: list):
        """
        Classify the AKI segments into:
            'stable': if consists of one segment with zero AKI.
            'AKI_all', if consists of one segment with nonzero AKI.
            'AKI_emergence': if consists of more than one segment with \
                with at least one zero segment followed by a nonzero segment.
            'AKI_recovery': if consists of two segments where the first \
                segment is nonzero and the second segment is zero.
            'unknown': if there are no valid AKI labels.
        """
        aki_vals = [segment['aki_label'] for segment in segmented_AKI]

        if len(aki_vals) == 1:
            if aki_vals[0] == 0:
                return ['stable']
            elif aki_vals[0] == 1:
                return ['AKI']
            else:
                return ['unknown']

        segment_class = []
        for i, aki in enumerate(aki_vals):
            if aki == 1.0:
                segment_class.append('AKI')
            elif onp.isnan(aki):
                segment_class.append('unknown')
            elif aki == 0:
                if i < len(aki_vals) - 1:
                    if aki_vals[i + 1] == 1:
                        if i == 0:
                            segment_class.append('first_AKI_pre_emergence')
                        else:
                            segment_class.append('AKI_pre_emergence')
                    elif onp.isnan(aki_vals[i + 1]):
                        segment_class.append('unknown')
                    else:
                        raise ValueError('Unexpected value in aki_vals')
                else:
                    if aki_vals[i - 1] == 1:
                        segment_class.append('AKI_recovery')
                    else:
                        segment_class.append('unknown')
            else:
                raise ValueError('Unexpected value in aki_vals')
        return segment_class

    def _segment_classify_predictions(self, predictions: Predictions):
        """
        Segment admission intervals and classify the admissions into \
            'stable', 'AKI_all', 'AKI_emergence', 'AKI_recovery', or 'other'.
        """
        segmented_AKI = {}
        for sid in predictions:
            for aid in predictions[sid]:
                prediction = predictions[sid][aid]
                _segmented_AKI = self._segment_aki_labels(prediction)
                _class = self._classify_aki_segments(_segmented_AKI)
                for i in range(len(_segmented_AKI)):
                    _segmented_AKI[i]['class'] = _class[i]
                segmented_AKI[aid] = _segmented_AKI
        return segmented_AKI

    def _segmented_AKI_byclass(self, segmented_AKI: dict):
        """
        Segment the admissions by class.
        """
        segmented_AKI_byclass = defaultdict(list)
        for aid in segmented_AKI:
            for segment in segmented_AKI[aid]:
                _class = segment['class']
                segmented_AKI_byclass[_class].append(segment)
        return segmented_AKI_byclass

    def _filter_empty_observables(self, predictions):
        """
        Filter out admissions with empty observables.
        """
        filtered_predictions = {}
        for sid in predictions:
            filtered_predictions[sid] = {}
            for aid in predictions[sid]:
                prediction = predictions[sid][aid]
                if onp.size(prediction.observables.time) > 0:
                    filtered_predictions[sid][aid] = prediction
        return filtered_predictions

    @property
    def time_window(self):
        return [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]

    def fields(self):
        segmented_classes = [
            'stable', 'AKI', 'unknown', 'first_AKI_pre_emergence',
            'AKI_pre_emergence', 'AKI_recovery'
        ]

        fields = [f'n_{c}' for c in segmented_classes]
        fields += [f'm_{c}' for c in segmented_classes]
        time_window = self.time_window
        for time_win in time_window[:-1]:
            fields.append(f'n_emergence_{time_win}-{time_window[-1]}')
            fields.append(f'emergence_auc_{time_win}-{time_window[-1]}')
            fields.append(f'n_first_emergence_{time_win}-{time_window[-1]}')
            fields.append(f'first_emergence_auc_{time_win}-{time_window[-1]}')
            fields.append(f'n_all_emergence_{time_win}-{time_window[-1]}')
            fields.append(f'all_emergence_auc_{time_win}-{time_window[-1]}')

        return fields

    def dirs(self):
        return (1, ) * len(self.fields())

    def _apply(self, predictions: Predictions):
        some_obs = list(next(iter(
            predictions.values())).values())[0].observables
        if not isinstance(some_obs, InpatientObservables):
            predictions = predictions.defragment_observables()
        some_obs = list(next(iter(
            predictions.values())).values())[0].observables
        assert isinstance(some_obs, InpatientObservables), \
            "Observables must be InpatientObservables."
        predictions = self._filter_empty_observables(predictions)

        time_window = self.time_window
        segmented_AKI = self._segment_classify_predictions(predictions)
        segmented_AKI_byclass = self._segmented_AKI_byclass(segmented_AKI)

        n_adm_trends = {c: len(v) for c, v in segmented_AKI_byclass.items()}
        res = {f'n_{k}': v for k, v in n_adm_trends.items()}
        res.update({
            f'm_{k}': sum(len(vi['time']) for vi in v)
            for k, v in segmented_AKI_byclass.items()
        })

        aki_emergence = segmented_AKI_byclass['AKI_pre_emergence']
        first_aki_emergence = segmented_AKI_byclass['first_AKI_pre_emergence']
        stable = segmented_AKI_byclass['stable'][0]['aki_preds']

        for time_win in time_window[:-1]:
            aki_emergence_preds = []
            first_aki_emergence_preds = []
            for pred in first_aki_emergence:
                ts = pred['time']
                last_ts = pred['next_interval_time']
                mask = (last_ts - ts <= time_window[-1]) & (last_ts - ts
                                                            >= time_win)
                first_aki_emergence_preds.append(pred['aki_preds'][mask])
            for pred in aki_emergence:
                ts = pred['time']
                last_ts = pred['next_interval_time']
                mask = (last_ts - ts <= time_window[-1]) & (last_ts - ts
                                                            >= time_win)
                aki_emergence_preds.append(pred['aki_preds'][mask])

            # later emergence
            em_preds = onp.hstack([stable] + aki_emergence_preds)
            labels = onp.hstack(
                [onp.zeros(len(stable))] +
                [onp.ones(len(f)) for f in aki_emergence_preds])
            (n_emerge, _), em_auc = nan_compute_auc(labels, em_preds)

            # first emergence
            first_em_preds = onp.hstack([stable] + first_aki_emergence_preds)
            labels = onp.hstack(
                [onp.zeros(len(stable))] +
                [onp.ones(len(f)) for f in first_aki_emergence_preds])
            (n_first_emerge,
             _), fem_auc = nan_compute_auc(labels, first_em_preds)

            # all emergence
            em_preds = onp.hstack([stable] + aki_emergence_preds +
                                  first_aki_emergence_preds)
            labels = onp.hstack(
                [onp.zeros(len(stable))] +
                [onp.ones(len(f)) for f in aki_emergence_preds] +
                [onp.ones(len(f)) for f in first_aki_emergence_preds])
            (n_all_emerge, _), aem_auc = nan_compute_auc(labels, em_preds)

            res[f'n_emergence_{time_win}-{time_window[-1]}'] = n_emerge
            res[f'emergence_auc_{time_win}-{time_window[-1]}'] = em_auc

            res[f'n_first_emergence_{time_win}-{time_window[-1]}'] = n_first_emerge
            res[f'first_emergence_auc_{time_win}-{time_window[-1]}'] = fem_auc

            res[f'n_all_emergence_{time_win}-{time_window[-1]}'] = n_all_emerge
            res[f'all_emergence_auc_{time_win}-{time_window[-1]}'] = aem_auc

        return res, segmented_AKI, segmented_AKI_byclass

    def __call__(self, predictions: Predictions):
        res, segmented_AKI, segmented_AKI_byclass = self._apply(predictions)
        return res


class CodeLevelMetricConfig(Config):
    code_level: bool = True
    aggregate_level: bool = True


class CodeLevelMetric(Metric):
    _index2code: Dict[int, str]
    _code2index: Dict[str, int]

    # Show estimates per code
    def __init__(self, patients, config=None, **kwargs):
        if config is None:
            config = CodeLevelMetricConfig()
        config = config.update(**kwargs)
        super().__init__(patients=patients, config=config)

    def _post_init(self):
        index = self.patients.scheme.outcome.index
        self._code2index = index
        self._index2code = {i: c for c, i in index.items()}

    @staticmethod
    def agg_fields():
        return (('mean', nanmean), ('median', nanmedian), ('max', nanmax),
                ('min', nanmin), ('std', nanstd), ('count', onp.size))

    def code_qualifier(self, code_index):
        return f'I{code_index}C{self._index2code[code_index]}'

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
        for index in sorted(self._index2code):
            for field in self.fields():
                yield (index, field)

    def columns(self):
        cols = []
        if self.config.code_level:
            cols.append(
                tuple(self.column(idx, field) for idx, field in self.order()))
        if self.config.aggregate_level:
            cols.append(self.agg_columns())

        return sum(tuple(cols), tuple())

    def agg_row(self, result: Dict[str, Dict[int, float]]):
        row = []
        for field in self.fields():
            if isinstance(result[field], dict):
                field_vals = onp.array(list(result[field].values()))
            else:
                field_vals = result[field]
            for _, agg_f in self.agg_fields():
                row.append(agg_f(field_vals))
        return tuple(row)

    def row(self, result: Dict[str, Dict[int, float]]):
        row = []
        if self.config.code_level:
            row.append(
                tuple(result[field][index] for index, field in self.order()))
        if self.config.aggregate_level:
            row.append(self.agg_row(result))

        return sum(tuple(row), tuple())

    def value_extractor(self, keys):
        code_index = keys.get('code_index')
        code = keys.get('code')
        assert (code_index is None) != (
            code
            is None), "providing code and code_index are mutually exlusive"
        code_index = self._code2index[code] if code is not None else code_index
        column = self.column(code_index, keys['field'])
        return self.from_df_functor(column, self.field_dir(keys['field']))

    def aggregate_extractor(self, keys):
        agg = keys['aggregate']
        column = self.agg_column(agg, keys.get('field', self.fields()[0]))
        return self.from_df_functor(column, self.field_dir(keys['field']))


class ObsCodeLevelMetric(CodeLevelMetric):

    def _post_init(self):
        index = self.patients.scheme.obs.index
        self._code2index = index
        self._index2code = {i: c for c, i in index.items()}


class LeadingObsMetric(CodeLevelMetric):

    def _post_init(self):
        conf = self.patients.config.leading_observable
        self._code2index = conf.code2index
        self._index2code = conf.index2code


class CodeAUC(CodeLevelMetric):

    @staticmethod
    def agg_fields():
        return (('mean', nanmean), ('weighted_mean', nanaverage),
                ('median', nanmedian), ('max', nanmax), ('min', nanmin),
                ('std', nanstd), ('count', onp.size))

    def agg_row(self, result: Dict[str, Dict[int, float]]):
        row = []
        for field in self.fields():
            field_vals = onp.array(list(result[field].values()))
            for agg_k, agg_f in self.agg_fields():
                if agg_k.startswith('weighted'):
                    weights = onp.array(list(result['n'].values()))
                    row.append(agg_f(field_vals, weights=weights, axis=None))
                else:
                    row.append(agg_f(field_vals))

        return tuple(row)

    @staticmethod
    def fields():
        return ('auc', 'n')

    @staticmethod
    def dirs():
        return (1, 1)

    def __call__(self, predictions: Predictions):
        ground_truth = []
        preds = []

        for subj_id, admission_risks in predictions.items():
            for admission_idx in sorted(admission_risks):
                pred = admission_risks[admission_idx]
                ground_truth.append(pred.admission.outcome.vec)
                preds.append(pred.outcome.vec)

        ground_truth_mat = onp.vstack(ground_truth)
        predictions_mat = onp.vstack(preds)

        vals = {'auc': {}, 'n': {}}
        for code_index in range(ground_truth_mat.shape[1]):
            code_ground_truth = ground_truth_mat[:, code_index]
            npos = code_ground_truth.sum()
            nneg = len(code_ground_truth) - npos
            code_predictions = predictions_mat[:, code_index]
            vals['n'][code_index] = code_ground_truth.sum()

            if npos > 2 and nneg > 2:
                vals['auc'][code_index] = compute_auc(code_ground_truth,
                                                      code_predictions)
            else:
                vals['auc'][code_index] = onp.nan
        return vals


class ColwiseLossMetricConfig(Config):
    loss: List[str] = field(default_factory=list)


class CodeLevelLossMetric(CodeLevelMetric):
    _loss_functions: Dict[str, Callable]

    def _post_init(self):
        CodeLevelMetric._post_init(self)
        self._loss_functions = {
            name: colwise_binary_loss[name]
            for name in self.config.loss
        }

    def fields(self):
        return list(map(lambda n: f'dx_{n}', sorted(self.config.loss)))

    def dirs(self):
        return (0, ) * len(self.fields())

    def __call__(self, predictions: Predictions):
        loss_vals = {
            f'dx_{name}': predictions.prediction_dx_loss(loss_f)
            for name, loss_f in self._loss_functions.items()
        }
        loss_vals = {
            name: dict(zip(range(len(v)), v))
            for name, v in loss_vals.items()
        }
        return loss_vals


class ObsCodeLevelLossMetric(ObsCodeLevelMetric):
    _loss_functions: Dict[str, Callable]

    def _post_init(self):
        ObsCodeLevelMetric._post_init(self)
        self._loss_functions = {
            name: colwise_numeric_loss[name]
            for name in self.config.loss
        }

    def fields(self):
        return list(map(lambda n: f'obs_{n}', sorted(self.config.loss)))

    def dirs(self):
        return (0, ) * len(self.fields())

    def __call__(self, predictions: Predictions):
        loss_vals = {
            f'obs_{name}': predictions.prediction_obs_loss(loss_f)
            for name, loss_f in self._loss_functions.items()
        }
        loss_vals = {
            name: dict(zip(range(len(v)), v))
            for name, v in loss_vals.items()
        }
        return loss_vals


class LeadingObsLossMetric(ObsCodeLevelLossMetric):
    _loss_functions: Dict[str, Callable]

    def _post_init(self):
        LeadingObsMetric._post_init(self)
        self._loss_functions = {
            name: colwise_numeric_loss[name]
            for name in self.config.loss
        }

    def fields(self):
        return list(map(lambda n: f'lead_{n}', sorted(self.config.loss)))

    def dirs(self):
        return (0, ) * len(self.fields())

    def __call__(self, predictions: Predictions):
        loss_vals = {
            f'lead_{name}': predictions.prediction_lead_loss(loss_f)
            for name, loss_f in self._loss_functions.items()
        }
        loss_vals = {
            name: dict(zip(range(len(v)), v))
            for name, v in loss_vals.items()
        }
        return loss_vals


class LeadingObsTrends(LeadingObsMetric):

    def fields(self):
        return ('tp', 'tn', 'fp', 'fn', 'n', 'pearson', 'spearman',
                'tp_pearson', 'tp_spearman', 'tn_pearson', 'tn_spearman',
                'mae', 'rms', 'mse')

    def dirs(self):
        return (1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0)

    @classmethod
    def _compute_tp(cls, trend, trend_hat, mask):
        tp = (trend > 0) * (trend_hat > 0) * mask

        trend = pd.DataFrame(jnp.where(tp > 0, trend, jnp.nan))
        trend_hat = pd.DataFrame(jnp.where(tp > 0, trend_hat, jnp.nan))

        tp_spearman = trend.corrwith(trend_hat, axis=0,
                                     method='spearman').values
        tp_pearson = trend.corrwith(trend_hat, axis=0, method='pearson').values
        return tp.sum(axis=0) / mask.sum(axis=0), tp_pearson, tp_spearman

    @classmethod
    def _compute_tn(cls, trend, trend_hat, mask):
        tn = (trend <= 0) * (trend_hat <= 0) * mask
        trend = pd.DataFrame(jnp.where(tn > 0, trend, jnp.nan))
        trend_hat = pd.DataFrame(jnp.where(tn > 0, trend_hat, jnp.nan))

        tn_spearman = trend.corrwith(trend_hat, axis=0,
                                     method='spearman').values
        tn_pearson = trend.corrwith(trend_hat, axis=0, method='pearson').values

        return tn.sum(axis=0) / mask.sum(axis=0), tn_pearson, tn_spearman

    @classmethod
    def _compute_fp(cls, trend, trend_hat, mask):
        a = (trend <= 0) * (trend_hat > 0) * mask
        return a.sum(axis=0) / mask.sum(axis=0)

    @classmethod
    def _compute_fn(cls, trend, trend_hat, mask):
        a = (trend > 0) * (trend_hat <= 0) * mask
        return a.sum(axis=0) / mask.sum(axis=0)

    @classmethod
    def _compute_corr(cls, trend, trend_hat, mask):
        trend = pd.DataFrame(onp.where(mask > 0, trend, jnp.nan))
        trend_hat = pd.DataFrame(onp.where(mask > 0, trend_hat, jnp.nan))

        spearman = trend.corrwith(trend_hat, axis=0, method='spearman').values
        pearson = trend.corrwith(trend_hat, axis=0, method='pearson').values
        return pearson, spearman

    @classmethod
    def _compute_errors(cls, trend, trend_hat, mask):
        trend = jnp.where(mask > 0, trend, jnp.nan)
        trend_hat = jnp.where(mask > 0, trend_hat, jnp.nan)

        mae = nanmean(onp.abs(trend - trend_hat), axis=0)
        rms = onp.sqrt(nanmean((trend - trend_hat)**2, axis=0))
        mse = nanmean((trend - trend_hat)**2, axis=0)
        return mae, rms, mse

    def __call__(self, predictions: Predictions):
        obs_index = self.patients.config.leading_observable.index
        data = predictions.prediction_lead_data(obs_index)
        y = onp.array(data['y'])
        y_hat = onp.array(data['y_hat'])
        mask = onp.array(data['mask'])
        obs = onp.array(data['obs'])
        obs_mask = onp.array(data['obs_mask'])

        trend = y - obs
        trend_hat = y_hat - obs
        mask = mask * obs_mask

        n = mask.sum(axis=0)
        tp, tp_pearson, tp_spearman = self._compute_tp(trend, trend_hat, mask)
        tn, tn_pearson, tn_spearman = self._compute_tn(trend, trend_hat, mask)
        fp = self._compute_fp(trend, trend_hat, mask)
        fn = self._compute_fn(trend, trend_hat, mask)
        pearson, spearman = self._compute_corr(trend, trend_hat, mask)
        mae, rms, mse = self._compute_errors(trend, trend_hat, mask)
        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'n': n,
            'pearson': pearson,
            'spearman': spearman,
            'tp_pearson': tp_pearson,
            'tp_spearman': tp_spearman,
            'tn_pearson': tn_pearson,
            'tn_spearman': tn_spearman,
            'mae': mae,
            'rms': rms,
            'mse': mse
        }


class UntilFirstCodeAUC(CodeAUC):

    def __call__(self, predictions: Predictions):
        ground_truth = []
        preds = []
        masks = []

        for subj_id, adms_predictions in predictions.items():
            # first_occ is a vector of admission indices
            # where the code at the corresponding
            # coordinate has first appeard for the patient. the index value of
            # (-1) means that the code will not appear at all for this patient.
            first_occ = self.patients.outcome_first_occurrence(subj_id)

            for admission_id in sorted(adms_predictions):
                mask = (admission_id <= first_occ)
                pred = adms_predictions[admission_id]
                ground_truth.append(pred.admission.outcome.vec)
                preds.append(pred.outcome.vec)
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


class MetricLevelsConfig(Config):

    # Show estimates for each admission for each subject (extremely large
    # table)
    admission: bool = False

    # Show estimates aggregated on the subject level (very large table)
    subject_aggregate: bool = False

    # Show estimates aggregated across the entire subjects and admissions.
    aggregate: bool = True


class AdmissionAUC(Metric):

    def __init__(self,
                 patients: Patients,
                 config: MetricLevelsConfig = None,
                 **kwargs):
        if config is None:
            config = MetricLevelsConfig()
        config = config.update(**kwargs)
        self.patients = patients
        self.config = config

    @staticmethod
    def fields():
        return ('auc', )

    @staticmethod
    def dirs():
        return (1, )

    @staticmethod
    def agg_fields():
        return (('mean', nanmean), ('median', nanmedian), ('max', nanmax),
                ('min', nanmin), ('std', nanstd), ('count', onp.size))

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
    def ordered_subjects(cls, predictions: Predictions):
        return sorted(predictions)

    @classmethod
    def order(cls, predictions: Predictions):
        for subject_id in cls.ordered_subjects(predictions):
            subject_predictions = predictions[subject_id]
            for admission_id in sorted(subject_predictions):
                for field in cls.fields():
                    yield (subject_id, admission_id, field)

    def columns(self, order_gen, subject_order_gen):
        cols = []
        if self.config.admission:
            cols.append(tuple(self.column(*o) for o in order_gen()))

        if self.config.subject_aggregate:
            cols.append(self.subject_agg_columns(subject_order_gen))

        if self.config.aggregate:
            cols.append(self.agg_columns())

        return sum(tuple(cols), tuple())

    def __call__(self, predictions: Predictions):
        auc = {}
        for subject_id in sorted(predictions):
            subject_predictions = predictions[subject_id]
            subject_auc = {}
            for admission_id in sorted(subject_predictions):
                pred = subject_predictions[admission_id]
                auc_score = compute_auc(pred.admission.outcome.vec,
                                        pred.outcome.vec)
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
        if self.config.admission:
            row.append(tuple(result[f][s][a] for s, a, f in order_gen()))
        if self.config.subject_aggregate:
            row.append(self.subject_agg_row(result, subject_order_gen))
        if self.config.aggregate:
            row.append(self.agg_row(result))
        return sum(tuple(row), tuple())

    def to_dict(self, predictions: Predictions):
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


class CodeGroupTopAlarmAccuracyConfig(Config):
    top_k_list: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 15, 20])
    n_partitions: int = 5


class CodeGroupTopAlarmAccuracy(Metric):
    _code_groups: List[List[int]]
    config: CodeGroupTopAlarmAccuracyConfig

    def __init__(self,
                 patients: Patients,
                 train_split: List[int],
                 config: MetricLevelsConfig = None,
                 **kwargs):
        if config is None:
            config = CodeGroupTopAlarmAccuracyConfig()
        config = config.update(**kwargs)
        super().__init__(patients=patients, config=config)

        if len(train_split) == 0:
            train_split = patients.dataset.subject_ids

        self._code_groups = patients.outcome_frequency_partitions(
            config.n_partitions, train_split)

    @classmethod
    def external_argnames(cls):
        return ('train_split', 'patients')

    @staticmethod
    def fields():
        return ('acc', )

    @staticmethod
    def dirs():
        return (1, )

    def column(self, group_index, k, field):
        return f'{self.classname()}.G{group_index}k{k}.{field}'

    def order(self):
        for k in self.config.top_k_list:
            for gi in range(len(self._code_groups)):
                for field in self.fields():
                    yield gi, k, field

    def columns(self):
        return tuple(self.column(gi, k, f) for gi, k, f in self.order())

    def __call__(self, predictions: Predictions):
        top_k_list = sorted(self.config.top_k_list)

        ground_truth = []
        preds = []

        for subject_risks in predictions.values():
            for pred in subject_risks.values():
                preds.append(pred.outcome.vec)
                ground_truth.append(pred.admission.outcome.vec)

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
        for gi, code_indices in enumerate(self._code_groups):
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
        assert k in self.config.top_k_list, f"k={k} is not in {self.config.top_k_list}"
        assert gi < len(
            self._code_groups), f"group_index ({gi}) >= len(self.code_groups)"
        field = keys.get('field', self.fields()[0])
        colname = self.column(gi, k, field)
        return self.from_df_functor(colname, self.field_dir(field))


class OtherMetrics(Metric):

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
    metrics: List[Metric] = field(default_factory=list)

    def to_df(self,
              iteration: int,
              predictions: Predictions,
              other_estimated_metrics: Dict[str, float] = None):

        dfs = [m.to_df(iteration, predictions) for m in self.metrics]
        return pd.concat(dfs, axis=1)


class DeLongTestConfig(Config):
    masking: str = 'all'


class DeLongTest(CodeLevelMetric):

    def __init__(self,
                 patients,
                 config: DeLongTestConfig = DeLongTestConfig(),
                 **kwargs):
        super().__init__(patients, config=config, **kwargs)

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
            code
            is None), "providing code and code_index are mutually exlusive"
        code_index = self._code2index[code] if code is not None else code_index

        if field in self.fields()['model']:
            column_gen = lambda kw: self.column()['model'](kw['model'], field)
        elif field in self.fields()['code']:
            column_gen = lambda _: self.column()['code'](field)
        else:
            column_gen = lambda kw: self.column()['pair'](tuple(
                sorted(kw['pair'])), field)

        def from_df(df, **kw):
            if isinstance(df, dict):
                df = self.to_df(df)
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
        for code_index in sorted(self._index2code):
            row_gen = self._row(code_index, data, clfs)
            rows.append(
                sum((row_gen[cat]() for cat in self.fields_category()),
                    tuple()))
        return rows

    def to_df(self, predictions: Dict[str, Predictions]):
        clfs = sorted(predictions)
        cols = self.columns(clfs)
        data = self(predictions)
        df = pd.DataFrame(columns=cols)
        for idx, row in enumerate(self.rows(data, clfs)):
            df.loc[idx] = row
        return df

    def to_dict(self, predictions: Dict[str, Predictions]):
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
    def _extract_subjects(cls, predictions: Dict[str, Predictions]):
        subject_sets = [preds.subject_ids for preds in predictions.values()]
        for s1, s2 in zip(subject_sets[:-1], subject_sets[1:]):
            assert set(s1) == set(s2), "Subjects mismatch across model outputs"
        return list(sorted(subject_sets[0]))

    def _extract_grountruth_vs_predictions(self, predictions):

        def masks(subj_id):
            adms = self.patients.subjects[subj_id].admissions
            if self.config.masking == 'all':
                return self.patients.outcome_all_masks(subj_id)
            elif self.config.masking == 'first':
                return self.patients.outcome_first_occurrence_masks(subj_id)
            else:
                raise ValueError(f"Unknown masking type {self.config.masking}")

        subjects = self._extract_subjects(predictions)
        true_mat = {}
        pred_mat = {}
        mask_mat = {}
        for clf_label, clf_preds in predictions.items():
            clf_true = []
            clf_pred = []
            clf_mask = []
            for subject_id in subjects:
                subj_preds = [
                    clf_preds[subject_id][aid]
                    for aid in sorted(clf_preds[subject_id])
                ]
                subj_masks = masks(subject_id)
                for pred, mask in zip(subj_preds, subj_masks):
                    clf_true.append(pred.admission.outcome.vec)
                    clf_mask.append(mask)
                    clf_pred.append(pred.outcome.vec)

            true_mat[clf_label] = onp.vstack(clf_true)
            pred_mat[clf_label] = onp.vstack(clf_pred)
            mask_mat[clf_label] = onp.vstack(clf_mask).astype(bool)

        tm0 = list(true_mat.values())[0]
        mask = list(mask_mat.values())[0]
        for tm, msk in zip(true_mat.values(), mask_mat.values()):
            assert (tm0 == tm).all(), "Mismatch in ground-truth across models."
            assert (mask == msk).all(), "Mismatch in mask across models."

        return tm0, mask, pred_mat

    def __call__(self, predictions: Dict[str, Predictions]):
        """
        Evaluate the AUC scores for each diagnosis code for each classifier. \
            In addition, conduct a pairwise test on the difference of AUC \
            scores between each pair of classifiers using DeLong test. \
            Codes that have either less than two positive cases or \
            have less than two negative cases are discarded \
            (AUC computation and difference test requirements).
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
            # Update this dictionary for each AUC computed,
            # then append the results to the big list of auc.
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
