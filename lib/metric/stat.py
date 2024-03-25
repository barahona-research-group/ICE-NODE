"""Performance metrics and loss functions."""

import warnings
from abc import abstractmethod, ABCMeta
from dataclasses import field, dataclass
from datetime import datetime
from functools import cached_property
from typing import Dict, Optional, List, Tuple, Any, Callable, ClassVar, Type, Literal

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from absl import logging
from sklearn import metrics
from tqdm import tqdm

from .delong import FastDeLongTest
from .loss import (NUMERIC_LOSS, NumericLossLiteral, BINARY_LOSS, BinaryLossLiteral)
from ..base import Module, Config, Array, VxDataItem
from ..ehr import (TVxEHR, InpatientObservables, CodesVector, LeadingObservableExtractorConfig)
from ..ehr.coding_scheme import SchemeManagerView
from ..ml.artefacts import AdmissionPrediction, AdmissionsPrediction

PredictionAttributeLiteral = Literal['outcome', 'observables', 'leading_observable']


class PredictionLoss(Module, metaclass=ABCMeta):
    prediction_attribute: ClassVar[PredictionAttributeLiteral] = None

    @abstractmethod
    def item_loss(self, ground_truth: VxDataItem, prediction: VxDataItem) -> Array:
        raise NotImplementedError

    def item_weight(self, ground_truth: VxDataItem) -> float:
        return 1.0

    def __call__(self, predictions: AdmissionsPrediction) -> Array | float:
        losses = jnp.array([self.item_loss(gt, pred) for gt, pred in predictions.iter_attr(self.prediction_attribute)])
        weights = jnp.array([self.item_weight(gt) for gt in predictions.list_attr(self.prediction_attribute)[0]])
        loss = jnp.nansum(losses * (weights / jnp.sum(weights)))

        if jnp.isnan(loss):
            logging.warning('NaN obs loss detected')

        return jnp.where(jnp.isnan(loss), 0., loss)


class NumericPredictionLoss(PredictionLoss):
    loss_key: NumericLossLiteral = field(default=None, kw_only=True)

    @cached_property
    def raw_loss(self) -> Callable[[Array, Array, Array], Array]:
        return NUMERIC_LOSS[self.loss_key]

    def item_loss(self, ground_truth: InpatientObservables, prediction: InpatientObservables) -> Array:
        return self.raw_loss(ground_truth.value, prediction.value, ground_truth.mask)

    def item_weight(self, ground_truth: InpatientObservables) -> float:
        return ground_truth.mask.sum()


class LeadPredictionLoss(NumericPredictionLoss):
    prediction_attribute: ClassVar[PredictionAttributeLiteral] = 'leading_observable'


class ObsPredictionLoss(NumericPredictionLoss):
    prediction_attribute: ClassVar[PredictionAttributeLiteral] = 'observables'


class OutcomePredictionLoss(PredictionLoss):
    loss_key: BinaryLossLiteral = field(default=None, kw_only=True)
    prediction_attribute: ClassVar[PredictionAttributeLiteral] = 'outcome'

    @cached_property
    def raw_loss(self) -> Callable[[Array, Array], Array]:
        return BINARY_LOSS[self.loss_key]

    def item_loss(self, ground_truth: CodesVector, prediction: CodesVector) -> Array:
        return self.raw_loss(ground_truth.vec, prediction.vec)


def safe_nan_func(func, x, axis):
    """Apply `func` to `x` along `axis`, ignoring NaNs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return func(x, axis=axis)


def silent_nanaverage(A, weights, axis):
    return safe_nan_func(
        lambda x, axis: onp.nansum(x * weights, axis=axis) / ((~onp.isnan(x)) * weights).sum(axis=axis), A, axis=axis)


def silent_nanmean(A, axis=None):
    return safe_nan_func(onp.nanmean, A, axis=axis)


def silent_nanmedian(A, axis=None):
    return safe_nan_func(onp.nanmedian, A, axis=axis)


def silent_nanstd(A, axis=None):
    return safe_nan_func(onp.nanstd, A, axis=axis)


def silent_nanmax(A, axis=None):
    return safe_nan_func(onp.nanmax, A, axis=axis)


def silent_nanmin(A, axis=None):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def estimands(self) -> Tuple[str, ...]:
        return tuple()

    def estimands_optimal_directions(self) -> Tuple[int, ...]:
        return tuple()

    @cached_property
    def estimand_optimal_direction(self) -> Dict[str, int]:
        assert len(self.estimands()) == len(self.estimands_optimal_directions()), \
            f'{self.classname()} must define estimands and their optimal directions.'
        return dict(zip(self.estimands(), self.estimands_optimal_directions()))

    @classmethod
    def classname(cls) -> str:
        return cls.__name__

    def estimand_column_name(self, estimand: str) -> str:
        return f'{self.classname()}.{estimand}'

    @cached_property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(map(self.estimand_column_name, self.estimands()))

    @abstractmethod
    def __call__(self, predictions: AdmissionsPrediction) -> Dict[str, float]:
        pass

    def estimand_values(self, result: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(map(result.get, self.estimands()))

    def to_dict(self, predictions: AdmissionsPrediction) -> Dict[str, float]:
        result = self(predictions)
        return dict(zip(self.column_names, self.estimand_values(result)))

    def to_df(self, index: int, predictions: AdmissionsPrediction) -> pd.DataFrame:
        timenow = datetime.now()
        data = self.to_dict(predictions)
        eval_time = (datetime.now() - timenow).total_seconds()
        data.update({f'{self.classname()}.eval_time': eval_time})
        return pd.DataFrame(data, index=[index])

    def value_extractor(self, **kwargs) -> Callable[[pd.DataFrame, Optional[int]], float | Tuple[int, float]]:
        field_name = kwargs.get('estimand', self.estimands()[0])
        column_name = self.estimand_column_name(field_name)
        return self.from_df_functor(column_name, self.estimand_optimal_direction[field_name])

    def from_df_functor(self, column_name, direction) -> Callable[[pd.DataFrame, str | int], float | Tuple[int, float]]:

        def from_df(df: pd.DataFrame, index: str | int = -1) -> float | Tuple[int, float]:
            if isinstance(df, AdmissionsPrediction):
                df = self.to_df(index, df)

            if index == 'optimum':
                if direction == 1:
                    return df[column_name].argmax(), df[column_name].max()
                else:
                    return df[column_name].argmin(), df[column_name].min()
            else:
                return df[column_name].iloc[index]

        return from_df


class VisitsAUC(Metric):

    def estimands(self) -> Tuple[str, ...]:
        return 'macro_auc', 'micro_auc'

    def estimands_optimal_directions(self) -> Tuple[int, ...]:
        return 1, 1

    def __call__(self, predictions: AdmissionsPrediction) -> Dict[str, float]:
        ground_truth_vectors, prediction_vectors = [], []
        for ground_truth_outcome, predicted_outcome in predictions.iter_attr('outcome'):
            ground_truth_vectors.append(ground_truth_outcome.vec)
            prediction_vectors.append(predicted_outcome.vec)
        return {
            'macro_auc':
                compute_auc(onp.hstack(ground_truth_vectors), onp.hstack(prediction_vectors)),
            'micro_auc':
                silent_nanmean(onp.array(list(map(
                    compute_auc,
                    ground_truth_vectors,
                    prediction_vectors,
                ))))
        }


class LossMetricConfig(Config):
    loss_keys: Tuple[str] = field(default_factory=tuple)


class LossMetric(Metric):
    config: LossMetricConfig = LossMetricConfig()
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = PredictionLoss

    @cached_property
    def loss_dictionary(self) -> Dict[str, PredictionLoss]:
        return {k: self.prediction_loss_class(loss_key=k) for k in self.config.loss_keys}

    def estimands_optimal_directions(self) -> Tuple[int, ...]:
        return tuple(1 if 'r2' in k else 0 for k in self.config.loss_keys)

    def estimands(self) -> Tuple[str, ...]:
        return tuple(sorted(self.config.loss_keys))

    def __len__(self):
        return len(self.estimands())

    def __call__(self, predictions: AdmissionsPrediction) -> Dict[str, float]:
        return {k: self.prediction_loss_class(loss_key=k)(predictions) for k in self.config.loss_keys}


class ObsPredictionLossConfig(LossMetricConfig):
    loss_keys: Tuple[NumericLossLiteral, ...] = ('mse', 'mae', 'rms', 'r2', 'soft_dtw_0_1')


class ObsPredictionLossMetric(LossMetric):
    config: ObsPredictionLossConfig = ObsPredictionLossConfig()
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = ObsPredictionLoss


class OutcomePredictionLossConfig(LossMetricConfig):
    loss_keys: Tuple[BinaryLossLiteral, ...] = ('bce', 'softmax_bce', 'balanced_focal_softmax_bce')


class OutcomePredictionLossMetric(LossMetric):
    config: OutcomePredictionLossConfig = OutcomePredictionLossConfig()
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = OutcomePredictionLoss


class LeadPredictionLossConfig(ObsPredictionLossMetric):
    pass


class LeadPredictionLossMetric(ObsPredictionLossMetric):
    config: LeadPredictionLossConfig = LeadPredictionLossConfig()
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = LeadPredictionLoss


class LeadingPredictionAccuracyConfig(LeadingObservableExtractorConfig):
    aki_stage_code: str

    @staticmethod
    def from_extractor_config(config: LeadingObservableExtractorConfig, aki_stage_code: str):
        return LeadingPredictionAccuracyConfig(**config.as_dict(), aki_stage_code=aki_stage_code)


class LeadingAKIPredictionAccuracy(Metric):
    config: LeadingPredictionAccuracyConfig
    context_view: SchemeManagerView

    @property
    def lead_extractor(self):
        return dict(
            zip(range(len(self.config.leading_hours)),
                [f'{self.config.observable_code}_next_{h}hrs' for h in self.config.leading_hours]))

    @property
    def aki_stage_index(self) -> int:
        return self.context_view.scheme[self.config.scheme].index[self.config.aki_stage_code]

    @property
    def aki_binary_index(self):
        return self.context_view.scheme[self.config.scheme].index[self.config.observable_code]

    def _annotate_lead_predictions(self, prediction: pd.DataFrame,
                                   ground_truth: pd.DataFrame):
        assert len(ground_truth) > 0, 'no ground truth'

        def _last_recovery_time(p):
            if ground_truth['value'].max() == 0:
                return -onp.inf
            t = p.time
            scope = ground_truth[(ground_truth.time <= t)]

            # no occurrence, no recovery
            if len(scope) == 0 or scope['value'].max() == 0:
                return -onp.inf

            # no recovery happened yet
            if scope['value'].iloc[-1] > 0:
                return onp.inf

            recovery = ((scope['value'].iloc[:-1].values > 0) &
                        (scope['value'].iloc[1:].values == 0))
            scope = scope.iloc[1:]
            return scope[recovery]['time'].max()

        def _next_occurrence_time(p):
            t = p.time
            scope = ground_truth[(ground_truth.time > t)
                                 & (ground_truth.value > 0)]
            if len(scope) == 0:
                return onp.inf

            return scope['time'].min()

        def _next_occurrence_value(t):
            if t == onp.inf:
                return onp.nan
            return ground_truth[ground_truth.time == t]['value'].values[0]

        prediction['last_recovery_time'] = prediction.apply(
            _last_recovery_time, axis=1)
        prediction['next_occurrence_time'] = prediction.apply(
            _next_occurrence_time, axis=1)
        prediction['next_occurrence_value'] = prediction[
            'next_occurrence_time'].apply(_next_occurrence_value)

        return prediction

    def _lead_dataframe(self, prediction: AdmissionPrediction):
        """Returns a dataframe of leading predictions after filtering out non-potential (non-useful) cases:
            (1) timestamps before sufficient acquisitions
                (self.config.minimum_acquisitions) are made.
            (2) timestamps within self.config.entry_neglect_window
                hours of admission.
            (3) entire admission where no valid AKI measured after
                self.config.entry_neglect_window.
            (4) timestamps where the most recent AKI measurement is actually
                recorded positive and no recovery since then, so no point to
                early predict here.
            (5) timestamps after recovery within
                self.config.recovery_window hours.

        Columns are:
            - time: timestamp of the leading maximal prediction.
            - prediction: maximal prediction value.
            - last_recovery_time: timestamp of the last recovery.
            - next_occurrence_time: timestamp of the next occurrence of the
            AKI.
            - next_occurrence_value: value of the next occurrence of the AKI.
        """
        lead_prediction = prediction.leading_observable.to_cpu()

        # criterion (1) - minimum acquisitions, early skip function.
        if len(lead_prediction) <= self.config.minimum_acquisitions:
            return None

        # criterion (1) - minimum acquisitions
        lead_time = lead_prediction.time[self.config.minimum_acquisitions:]
        lead_critical_val = onp.nanmax(
            lead_prediction.value, axis=1)[self.config.minimum_acquisitions:]

        # criterion (2) - entry neglect window, early skip function.
        entry_neglect_mask = (lead_time > self.config.entry_neglect_window)
        if entry_neglect_mask.sum() == 0:
            return None
        else:
            lead_time = lead_time[entry_neglect_mask]
            lead_critical_val = lead_critical_val[entry_neglect_mask]

        obs_ground_truth = prediction.admission.observables.to_cpu()

        obs_index = self.aki_binary_index
        mask = obs_ground_truth.mask[:, obs_index]
        # criterion (3) - no valid AKI measured, early skip function.
        if mask.sum() == 0:
            return None

        ground_truth_val = obs_ground_truth.value[mask, obs_index]
        ground_truth_time = obs_ground_truth.time[mask]

        # criterion (3) - no valid AKI measured after entry neglect window.
        if ground_truth_time.max() < self.config.entry_neglect_window:
            return None

        prediction_df = pd.DataFrame({
            'admission_id': prediction.admission.admission_id,
            'time': lead_time,
            'value': lead_critical_val
        })
        ground_truth_df = pd.DataFrame({
            'time': ground_truth_time,
            'value': ground_truth_val
        })

        # criterion (4) - no recovery since last (current) occurrence.
        pos_ground_truth_df = ground_truth_df[ground_truth_df['value'] > 0]
        prediction_df = prediction_df[~prediction_df['time'].isin(pos_ground_truth_df['time'])]

        if len(prediction_df) == 0:
            return None

        prediction_df = self._annotate_lead_predictions(
            prediction_df, ground_truth_df)

        # criterion (4) & (5)- recovery neglect window and no recovery since
        # last occurrence.
        prediction_df = prediction_df[(prediction_df['time']
                                       > (prediction_df['last_recovery_time'] +
                                          self.config.recovery_window))]

        if len(prediction_df) == 0:
            return None

        return prediction_df

    def _lead_dataframes(self, predictions: AdmissionsPrediction):
        dataframes = []
        for patient_predictions in predictions.values():
            for prediction in patient_predictions.values():
                df = self._lead_dataframe(prediction)
                if df is not None:
                    dataframes.append(self._lead_dataframe(prediction))
        return pd.concat(dataframes)

    def _classify_timestamps(self, prediction_df: pd.DataFrame):
        """Classifies the timestamps in the dataframe into the following
        classes:
            - negative: no AKI measured.
            - first_pre_emergence: no past AKI, but in development.
            - later_pre_emergence: recovered from AKI, but new AKI in
            - recovery_window: recovered from AKI, within
                self.config.recovery_window hours.
            - recovered: recovered from AKI, after self.config.recovery_window
                hours.
            development.

        Returns a dataframe with the following columns:
            - time: timestamp of the leading maximal prediction.
            - prediction: maximal prediction value.
            - last_recovery_time: timestamp of the last recovery.
            - next_occurrence_time: timestamp of the next occurrence of the
            AKI.
            - next_occurrence_value: value of the next occurrence of the AKI.
            - class: class of the timestamp.
        """
        df = prediction_df

        df['class'] = 'unknown'
        df.loc[(df['last_recovery_time'] == -onp.inf) & (df['next_occurrence_time'] == onp.inf), 'class'] = 'negative'
        df.loc[(df['last_recovery_time'] == -onp.inf) & (
                df['next_occurrence_time'] != onp.inf), 'class'] = 'first_pre_emergence'
        df.loc[(df['last_recovery_time'] != -onp.inf) & (
                df['last_recovery_time'] + self.config.recovery_window < df['time']) & (
                       df['next_occurrence_time'] != onp.inf), 'class'] = 'later_pre_emergence'
        df.loc[(df['last_recovery_time'] != -onp.inf) & (
                df['last_recovery_time'] + self.config.recovery_window >= df['time']), 'class'] = 'recovery_window'
        df.loc[(df['last_recovery_time'] != -onp.inf) & (
                df['last_recovery_time'] + self.config.recovery_window < df['time']) & (
                       df['next_occurrence_time'] == onp.inf), 'class'] = 'recovered'
        return prediction_df

    def estimands(self):
        timestamp_class = [
            'negative', 'unknown', 'first_pre_emergence',
            'later_pre_emergence', 'recovery_window'
        ]

        fields = [f'n_timestamps_{c}' for c in timestamp_class]
        fields += [f'n_admissions_{c}' for c in timestamp_class]
        time_window = self.config.leading_hours
        t1 = time_window[-1]
        pre_emergence_types = [
            'first_pre_emergence', 'later_pre_emergence',
            'pre_emergence'
        ]
        for t0 in time_window[:-1]:
            for c in pre_emergence_types:
                fields.append(f'n_timestamps_{c}_{t0}-{t1}')
                fields.append(f'AUC_{c}_{t0}-{t1}')
        return fields

    def __call__(self, predictions: AdmissionsPrediction):
        df = self._lead_dataframes(predictions)
        df = self._classify_timestamps(df)
        df_grouped = df.groupby('class')
        res = {}
        n_timestamps = df_grouped['time'].count().to_dict()
        n_admissions = df_grouped['admission_id'].nunique().to_dict()
        for c in n_timestamps.keys():
            res[f'n_timestamps_{c}'] = n_timestamps[c]
            res[f'n_admissions_{c}'] = n_admissions[c]
        df_grouped = {k: v for k, v in df_grouped}
        df_negative = df_grouped.get('negative', df.iloc[:0])
        df_recovered = df_grouped.get('recovered', df.iloc[:0])
        df_first_pre_emergence = df_grouped.get('first_pre_emergence',
                                                df.iloc[:0])
        df_later_pre_emergence = df_grouped.get('later_pre_emergence',
                                                df.iloc[:0])

        pre_emergence_df = {
            'first_pre_emergence':
                df_first_pre_emergence,
            'later_pre_emergence':
                df_later_pre_emergence,
            'pre_emergence':
                pd.concat([df_first_pre_emergence, df_later_pre_emergence])
        }

        neg_values = onp.concatenate(
            (df_negative['value'].values, df_recovered['value'].values))
        neg_labels = onp.zeros(len(neg_values))
        time_window = self.config.leading_hours
        t1 = time_window[-1]
        for t0 in time_window[:-1]:
            for c, df in pre_emergence_df.items():
                df_t = df[(df['time'] >= t0) & (df['time'] <= t1)]
                if len(df_t) == 0:
                    continue
                pos_values = df_t['value'].values
                pos_labels = onp.ones(len(pos_values))
                values = onp.concatenate((neg_values, pos_values))
                labels = onp.concatenate((neg_labels, pos_labels))
                res[f'AUC_{c}_{t0}-{t1}'] = compute_auc(labels, values)
                res[f'n_timestamps_{c}_{t0}-{t1}'] = len(df_t)
        return res


class CodeLevelMetricConfig(Config):
    scheme: str
    code_scores: bool = True
    global_scores: bool = True


class CodeLevelMetric(Metric):
    config: CodeLevelMetricConfig
    context_view: SchemeManagerView

    @cached_property
    def codes(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @staticmethod
    def agg_fields():
        return (('mean', silent_nanmean), ('median', silent_nanmedian), ('max', silent_nanmax),
                ('min', silent_nanmin), ('std', silent_nanstd), ('count', onp.size))

    def code_qualifier(self, code_index):
        return f'I{code_index}C{self.codes[code_index]}'

    def estimand_column_name(self, code_index, estimand):
        return f'{self.classname()}.{self.code_qualifier(code_index)}.{estimand}'

    @classmethod
    def agg_column(cls, agg_key, estimand):
        return f'{cls.classname()}.{agg_key}({estimand})'

    def agg_columns(self):
        cols = []
        for est in self.estimands():
            for agg_k, _ in self.agg_fields():
                cols.append(self.agg_column(agg_k, est))
        return tuple(cols)

    def order(self):
        for index in range(len(self.codes)):
            for est in self.estimands():
                yield index, est

    def column_names(self):
        cols = []
        if self.config.code_scores:
            cols.append(
                tuple(self.estimand_column_name(idx, est) for idx, est in self.order()))
        if self.config.global_scores:
            cols.append(self.agg_columns())

        return sum(tuple(cols), tuple())

    def agg_row(self, result: Dict[str, Dict[int, float]]):
        row = []
        for est in self.estimands():
            if isinstance(result[est], dict):
                field_vals = onp.array(list(result[est].values()))
            else:
                field_vals = result[est]
            for _, agg_f in self.agg_fields():
                row.append(agg_f(field_vals))
        return tuple(row)

    def estimand_values(self, result: Dict[str, Dict[int, float]]):
        row = []
        if self.config.code_scores:
            row.append(
                tuple(result[est][index] for index, est in self.order()))
        if self.config.global_scores:
            row.append(self.agg_row(result))

        return sum(tuple(row), tuple())

    def value_extractor(self, keys):
        code_index = keys.get('code_index')
        code = keys.get('code')
        assert (code_index is None) != (
                code
                is None), "providing code and code_index are mutually exlusive"
        code_index = self.codes.index(code) if code is not None else code_index
        column = self.estimand_column_name(code_index, keys['field'])
        return self.from_df_functor(column, self.estimand_optimal_direction[keys['estimand']])

    def aggregate_extractor(self, keys):
        agg = keys['aggregate']
        column = self.agg_column(agg, keys.get('estimand', self.estimands()[0]))
        return self.from_df_functor(column, self.estimand_optimal_direction[keys['estimand']])


class ObsCodeLevelMetric(CodeLevelMetric):
    @cached_property
    def codes(self) -> Tuple[str, ...]:
        return self.context_view.scheme[self.config.scheme].codes


class CodeAUC(CodeLevelMetric):
    @cached_property
    def codes(self) -> Tuple[str, ...]:
        return self.context_view.outcome[self.config.scheme].codes

    @staticmethod
    def agg_fields() -> Tuple[Tuple[str, Callable], ...]:
        return (('mean', silent_nanmean), ('weighted_mean', silent_nanaverage),
                ('median', silent_nanmedian), ('max', silent_nanmax), ('min', silent_nanmin),
                ('std', silent_nanstd), ('count', onp.size))

    def agg_row(self, result: Dict[str, Dict[int, float]]) -> Tuple[float, ...]:
        row = []
        for field in self.estimands():
            field_vals = onp.array(list(result[field].values()))
            for agg_k, agg_f in self.agg_fields():
                if agg_k.startswith('weighted'):
                    weights = onp.array(list(result['n'].values()))
                    row.append(agg_f(field_vals, weights=weights, axis=None))
                else:
                    row.append(agg_f(field_vals))

        return tuple(row)

    def estimands(self) -> Tuple[str, ...]:
        return 'auc', 'n'

    def estimands_optimal_directions(self):
        return 1, 1

    def __call__(self, predictions: AdmissionsPrediction):
        ground_truth = []
        preds = []
        for ground_truth_outcome, predicted_outcome in predictions.iter_attr('outcome'):
            ground_truth.append(ground_truth_outcome.vec)
            preds.append(predicted_outcome.vec)

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


class MetricLevelsConfig(Config):
    # Show estimates for each admission for each subject (extremely large
    # table)
    admission: bool = False

    # Show estimates aggregated on the subject level (very large table)
    subject_aggregate: bool = False

    # Show estimates aggregated across the entire subjects and admissions.
    aggregate: bool = True


class AdmissionAUC(Metric):
    config: MetricLevelsConfig

    def __init__(self,
                 patients: TVxEHR,
                 config: MetricLevelsConfig = None,
                 **kwargs):
        if config is None:
            config = MetricLevelsConfig()
        config = config.update(**kwargs)
        self.patients = patients
        self.config = config

    def estimands(self):
        return ('auc',)

    def estimands_optimal_directions(self):
        return (1,)

    @staticmethod
    def agg_fields():
        return (('mean', silent_nanmean), ('median', silent_nanmedian), ('max', silent_nanmax),
                ('min', silent_nanmin), ('std', silent_nanstd), ('count', onp.size))

    @staticmethod
    def subject_qualifier(subject_id):
        return f'S{subject_id}'

    @classmethod
    def admission_qualifier(cls, subject_id, admission_id):
        return f'{cls.subject_qualifier(subject_id)}A{admission_id}'

    @classmethod
    def estimand_column_name(cls, subject_id, admission_id, field):
        clsname = cls.classname()
        return f'{clsname}.{cls.admission_qualifier(subject_id, admission_id)}.{field}'

    @classmethod
    def subject_agg_column(cls, subject_id, agg_key, field):
        return f'{cls.classname()}.{cls.subject_qualifier(subject_id)}.{agg_key}({field})'

    @classmethod
    def subject_agg_columns(cls, subject_order_gen):
        cols = []
        for subject_id in subject_order_gen():
            for field in cls.estimands():
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
        for field in cls.estimands():
            for agg_k, _ in cls.agg_fields():
                cols.append(cls.agg_column(agg_k, field))
        return tuple(cols)

    @classmethod
    def ordered_subjects(cls, predictions: AdmissionsPrediction):
        return sorted(p.subject_id for p in predictions.predictions)

    def order(self, predictions: AdmissionsPrediction):
        for subject_id in self.ordered_subjects(predictions):
            subject_predictions = predictions.subject_predictions[subject_id]
            for admission_id in sorted(p.admission.admission_id for p in subject_predictions):
                for field in self.estimands():
                    yield subject_id, admission_id, field

    def column_names(self, order_gen, subject_order_gen):
        cols = []
        if self.config.admission:
            cols.append(tuple(self.estimand_column_name(*o) for o in order_gen()))

        if self.config.subject_aggregate:
            cols.append(self.subject_agg_columns(subject_order_gen))

        if self.config.aggregate:
            cols.append(self.agg_columns())

        return sum(tuple(cols), tuple())

    def __call__(self, predictions: AdmissionsPrediction):
        auc = {}
        for subject_id, subject_predictions in predictions.subject_predictions.items():
            subject_auc = {}
            for admission_prediction in subject_predictions:
                auc_score = compute_auc(admission_prediction.admission.outcome.vec,
                                        admission_prediction.outcome.vec)
                subject_auc[admission_prediction.admission.admission_id] = auc_score
            auc[subject_id] = subject_auc
        return {'auc': auc}

    def subject_agg_row(self, result, subject_order_gen):
        row = []
        for subject_id in subject_order_gen():
            for field in self.estimands():
                field_data = onp.array(list(
                    result[field][subject_id].values()))
                for _, agg_f in self.agg_fields():
                    row.append(agg_f(field_data))
        return tuple(row)

    def agg_row(self, result):
        row = []
        for field in self.estimands():
            fdata = result[field]
            data = list(v for sdata in fdata.values() for v in sdata.values())
            data = onp.array(data)
            for _, agg_f in self.agg_fields():
                row.append(agg_f(data))
        return tuple(row)

    def estimand_values(self, result: Dict[str, Any], order_gen, subject_order_gen):
        row = []
        if self.config.admission:
            row.append(tuple(result[f][s][a] for s, a, f in order_gen()))
        if self.config.subject_aggregate:
            row.append(self.subject_agg_row(result, subject_order_gen))
        if self.config.aggregate:
            row.append(self.agg_row(result))
        return sum(tuple(row), tuple())

    def to_dict(self, predictions: AdmissionsPrediction):
        order_gen = lambda: self.order(predictions)
        subject_order_gen = lambda: self.ordered_subjects(predictions)
        result = self(predictions)
        cols = self.column_names(order_gen, subject_order_gen)
        rows = self.estimand_values(result, order_gen, subject_order_gen)
        return dict(zip(cols, rows))

    def value_extractor(self, keys):
        subject_id = keys['subject_id']
        admission_id = keys['admission_id']
        estimand = keys.get('estimand', self.estimands()[0])
        column = self.estimand_column_name(subject_id, admission_id, estimand)
        return self.from_df_functor(column, self.estimand_optimal_direction[estimand])

    def subject_aggregate_extractor(self, keys):
        estimand = keys.get('estimand', self.estimands()[0])
        column = self.subject_agg_column(keys['subject_id'], keys['aggregate'],
                                         estimand)
        return self.from_df_functor(column, self.estimand_optimal_direction[estimand])

    def aggregate_extractor(self, keys):
        agg = keys['aggregate']
        estimand = keys.get('estimand', self.estimands()[0])
        column = self.agg_column(agg, estimand)
        return self.from_df_functor(column, self.estimand_optimal_direction[estimand])


class CodeGroupTopAlarmAccuracyConfig(Config):
    top_k_list: List[int] = field(
        default_factory=lambda: [1, 3, 5, 10, 15, 20])
    n_partitions: int = 5


class CodeGroupTopAlarmAccuracy(Metric):
    code_groups: List[List[int]]
    config: CodeGroupTopAlarmAccuracyConfig

    @classmethod
    def external_argnames(cls):
        return ('train_split', 'patients')

    @staticmethod
    def estimands():
        return ('acc',)

    @staticmethod
    def estimands_optimal_directions():
        return (1,)

    def estimand_column_name(self, group_index, k, field):
        return f'{self.classname()}.G{group_index}k{k}.{field}'

    def order(self):
        for k in self.config.top_k_list:
            for gi in range(len(self.code_groups)):
                for field in self.estimands():
                    yield gi, k, field

    def column_names(self):
        return tuple(self.estimand_column_name(gi, k, f) for gi, k, f in self.order())

    def __call__(self, predictions: AdmissionsPrediction):
        top_k_list = sorted(self.config.top_k_list)

        ground_truth = []
        preds = []

        for ground_truth_outcome, predicted_outcome in predictions.iter_attr('outcome'):
            preds.append(predicted_outcome.vec)
            ground_truth.append(ground_truth_outcome.vec)

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

    def estimand_values(self, result: Dict[str, Dict[int, Dict[int, float]]]):
        return tuple(result[f][gi][k] for gi, k, f in self.order())

    def value_extractor(self, keys: Dict[str, str]):
        k = keys['k']
        gi = keys['group_index']
        assert k in self.config.top_k_list, f"k={k} is not in {self.config.top_k_list}"
        assert gi < len(self.code_groups), f"group_index ({gi}) >= len(self.code_groups)"
        estimand = keys.get('estimand', self.estimands()[0])
        colname = self.estimand_column_name(gi, k, estimand)
        return self.from_df_functor(colname, self.estimand_optimal_direction[estimand])


@dataclass
class MetricsCollection:
    metrics: List[Metric] = field(default_factory=list)

    def columns(self):
        return tuple(sum([m.column_names for m in self.metrics], ()))

    def to_df(self,
              iteration: int,
              predictions: AdmissionsPrediction,
              other_estimated_metrics: Dict[str, float] = None):
        dfs = [m.to_df(iteration, predictions) for m in self.metrics]
        return pd.concat(dfs, axis=1)


class DeLongTest(CodeLevelMetric):

    def estimands(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'code': ('n_pos',),
            'model': ('auc', 'auc_var'),
            'pair': ('p_val',)
        }

    @staticmethod
    def fields_category():
        return ('code', 'model', 'pair')

    def estimand_column_name(self):

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
        code_index = self.codes.index(code) if code is not None else code_index

        if field in self.estimands()['model']:
            column_gen = lambda kw: self.estimand_column_name()['model'](kw['model'], field)
        elif field in self.estimands()['code']:
            column_gen = lambda _: self.estimand_column_name()['code'](field)
        else:
            column_gen = lambda kw: self.estimand_column_name()['pair'](tuple(
                sorted(kw['pair'])), field)

        def from_df(df, **kw):
            if isinstance(df, dict):
                df = self.to_df(df)
            return df.loc[code_index, column_gen(kw)]

        return from_df

    def column_names(self, clfs: List[str]):
        cols = []
        order_gen = self.order(clfs)
        for cat in self.fields_category():
            col_f = self.estimand_column_name()[cat]
            cols.append(tuple(col_f(*t) for t in order_gen[cat]()))
        return sum(cols, tuple())

    def _row(self, code_index: int, data: Dict[str, Any], clfs: List[str]):
        order_gen = self.order(clfs)

        def code_row():
            d = data['code']
            return tuple(d[f][code_index] for (f,) in order_gen['code']())

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
        for code_index in range(len(self.codes)):
            row_gen = self._row(code_index, data, clfs)
            rows.append(
                sum((row_gen[cat]() for cat in self.fields_category()),
                    tuple()))
        return rows

    def to_df(self, predictions: Dict[str, AdmissionsPrediction]):
        clfs = sorted(predictions)
        cols = self.column_names(clfs)
        data = self(predictions)
        df = pd.DataFrame(columns=cols)
        for idx, row in enumerate(self.rows(data, clfs)):
            df.loc[idx] = row
        return df

    def to_dict(self, predictions: Dict[str, AdmissionsPrediction]):
        clfs = sorted(predictions)
        cols = self.column_names(clfs)
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
            fields = self.estimands()['model']
            for model in clfs:
                for field in fields:
                    yield (model, field)

        def pair_order():
            pairs = self._model_pairs(clfs)
            fields = self.estimands()['pair']
            for model_pair in pairs:
                for field in fields:
                    yield (model_pair, field)

        def code_order():
            fields = self.estimands()['code']
            for field in fields:
                yield (field,)

        return {'pair': pair_order, 'code': code_order, 'model': model_order}

    @classmethod
    def _extract_subjects(cls, predictions: Dict[str, AdmissionsPrediction]):
        subject_sets = [preds.subject_ids for preds in predictions.values()]
        for s1, s2 in zip(subject_sets[:-1], subject_sets[1:]):
            assert set(s1) == set(s2), "Subjects mismatch across model outputs"
        return list(sorted(subject_sets[0]))

    def _extract_grountruth_vs_predictions(self, predictions):
        true_mat = {}
        pred_mat = {}
        for clf_label, clf_preds in predictions.items():
            clf_true = []
            clf_pred = []
            for ground_truth_outcome, predicted_outcome in clf_preds.iter_attr('outcome'):
                clf_true.append(ground_truth_outcome.vec)
                clf_pred.append(predicted_outcome.vec)

            true_mat[clf_label] = onp.vstack(clf_true)
            pred_mat[clf_label] = onp.vstack(clf_pred)

        tm0 = list(true_mat.values())[0]
        for tm in true_mat.values():
            assert (tm0 == tm).all(), "Mismatch in ground-truth across models."

        return tm0, pred_mat

    def __call__(self, predictions: Dict[str, AdmissionsPrediction]):
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
        true_mat, preds = self._extract_grountruth_vs_predictions(predictions)
        clf_pairs = self._model_pairs(clf_labels)

        n_pos = {}  # only codewise
        auc = {}  # modelwise
        auc_var = {}  # modelwise
        p_val = {}  # pairwise

        for code_index in tqdm(range(true_mat.shape[1])):
            code_truth = true_mat[:, code_index]
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

    def filter_results(self, results: pd.DataFrame, models: List[str],
                       min_auc: float):
        models = sorted(models)
        m_pairs = self._model_pairs(models)

        # exclude tests with nans
        pval_cols = [self.estimand_column_name()['pair'](pair, 'p_val') for pair in m_pairs]
        valid_tests = results.loc[:, pval_cols].isnull().max(axis=1) == 0
        results = results[valid_tests]
        logging.info(
            f'{sum(valid_tests)}/{len(valid_tests)} rows with valid tests (no NaN p-values)'
        )

        # AUC cut-off
        auc_cols = [self.estimand_column_name()['model'](m, 'auc') for m in models]
        accepted_aucs = results.loc[:, auc_cols].max(axis=1) > min_auc
        results = results[accepted_aucs]
        logging.info(f'{sum(accepted_aucs)}/{len(accepted_aucs)} rows with\
            an AUC higher than {min_auc} by at least one model.')

        return results

    def insignificant_difference_rows(self, results: pd.DataFrame,
                                      models: List[str], p_value: float):
        models = sorted(models)
        m_pairs = self._model_pairs(models)
        pval_cols = [self.estimand_column_name()['pair'](pair, 'p_val') for pair in m_pairs]
        insigificant = results.loc[:, pval_cols].min(axis=1) > p_value
        return results[insigificant]
