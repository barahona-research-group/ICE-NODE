"""Performance metrics and loss functions."""

import warnings
from abc import abstractmethod
from dataclasses import field, dataclass
from datetime import datetime
from functools import cached_property
from typing import Optional, Tuple, ClassVar, Type, List, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from .delong import FastDeLongTest
from .loss import (NumericLossLiteral, BinaryLossLiteral)
from .loss_wrap import PredictionLoss, ObsPredictionLoss, OutcomePredictionLoss, LeadPredictionLoss
from ..base import Module, Config, Array
from ..ehr import (LeadingObservableExtractorConfig)
from ..ml.artefacts import AdmissionPrediction, AdmissionsPrediction


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


@dataclass
class MetricsOutput:
    name: str
    estimands: Tuple[str, ...]
    values: Tuple[float | Array, ...]
    time_elapsed: float
    aggregations: ClassVar[Tuple[Tuple[str, Callable], ...]] = (('mean', silent_nanmean), ('median', silent_nanmedian),
                                                                ('max', silent_nanmax), ('min', silent_nanmin),
                                                                ('std', silent_nanstd), ('count', onp.size))

    @property
    def dim(self) -> Optional[int]:
        if len(self.values) == 0:
            return None
        value = next(iter(self.values))
        if isinstance(value, (jnp.ndarray, onp.ndarray)) and onp.ndim(value) > 0:
            return onp.size(value)
        return None

    def columns(self, codes: Optional[Tuple[str, ...]] = None) -> List[str]:
        return self.expand_metrics_labels(codes) + self.metrics_aggregation_columns + [f'{self.name}.eval_time']

    def expand_metrics_labels(self, codes: Optional[Tuple[str, ...]]) -> List[str]:
        return sum((self.expand_metric_label(estimand, codes) for estimand in self.estimands), [])

    def expand_metric_label(self, estimand: str, codes: Optional[Tuple[str, ...]]) -> List[str]:
        if self.dim is None and codes is None:
            return [f'{self.name}.{estimand}']
        if codes is not None:
            assert self.dim is not None, 'n must be provided when codes are specified'
        expansion = codes or range(self.dim)
        return [f'{self.name}.{estimand}[{i}]' for i in expansion]

    def metric_aggregation_columns(self, estimand: str) -> List[str]:
        if self.dim is None:
            return []
        return [f'{self.name}.{k}({estimand})' for k, _ in self.aggregations]

    @property
    def metrics_aggregation_columns(self) -> List[str]:
        return sum((self.metric_aggregation_columns(estimand) for estimand in self.estimands), [])

    @property
    def aggregation_rows(self) -> List[float]:
        if self.dim is not None:
            return [agg_f(value) for value in self.values for _, agg_f in self.aggregations]
        else:
            return []

    @property
    def row(self) -> List[float]:
        if self.dim is not None:
            values = sum((onp.hstack(v).flatten().tolist() for v in self.values), [])
        else:
            values = list(self.values)
        return values + self.aggregation_rows + [self.time_elapsed]

    def estimand_column_name(self, estimand: str, index: Optional[int | str]) -> str:
        return f'{self.name}.{estimand}' + (f'[{index}]' if index is not None else '')

    def as_df(self, index: int = 0, codes: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
        return pd.DataFrame([self.row], columns=self.columns(codes), index=[index])


class Metric(Module):
    config: Config = field(default_factory=Config)
    estimands: Tuple[str, ...] = field(default_factory=tuple)

    def __call__(self, predictions: AdmissionsPrediction) -> MetricsOutput:
        time_now = datetime.now()
        values = self.apply(predictions)
        eval_time = (datetime.now() - time_now).total_seconds()
        return MetricsOutput(name=type(self).__name__, estimands=self.estimands, values=values, time_elapsed=eval_time)

    @abstractmethod
    def apply(self, predictions: AdmissionsPrediction) -> Tuple[float | Array, ...]:
        pass


class VisitsAUC(Metric):
    estimands: Tuple[str, ...] = ('macro_auc', 'micro_auc')

    def apply(self, predictions: AdmissionsPrediction) -> Tuple[float | Array, ...]:
        ground_truth_vectors, prediction_vectors = [], []
        for ground_truth_outcome, predicted_outcome in predictions.iter_attr('outcome'):
            ground_truth_vectors.append(ground_truth_outcome.vec)
            prediction_vectors.append(predicted_outcome.vec)
        macro_auc = compute_auc(onp.hstack(ground_truth_vectors), onp.hstack(prediction_vectors))
        micro_auc = silent_nanmean(onp.array(list(map(
            compute_auc,
            ground_truth_vectors,
            prediction_vectors,
        ))))
        return macro_auc, micro_auc


class LossMetricConfig(Config):
    loss_keys: Tuple[str, ...] = field(default_factory=tuple)
    per_column: bool = False


class LossMetric(Metric):
    config: LossMetricConfig = field(default_factory=lambda: LossMetricConfig())
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = PredictionLoss
    estimands: Tuple[str, ...] = field(default_factory=lambda: tuple())

    def __post_init__(self):
        self.estimands = tuple(sorted(self.config.loss_keys))

    @cached_property
    def loss_functions(self) -> List[PredictionLoss]:
        return [self.prediction_loss_class(loss_key=k, per_column=self.config.per_column) for k in self.estimands]

    def apply(self, predictions: AdmissionsPrediction) -> Tuple[float | Array, ...]:
        return tuple(f(predictions) for f in self.loss_functions)


class ObsPredictionLossConfig(LossMetricConfig):
    loss_keys: Tuple[NumericLossLiteral, ...] = ('mse', 'mae', 'rms', 'r2')


class ObsPredictionLossMetric(LossMetric):
    config: ObsPredictionLossConfig = field(default_factory=lambda: ObsPredictionLossConfig())
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = ObsPredictionLoss


class PerColumnObsPredictionLoss(ObsPredictionLossMetric):
    config: ObsPredictionLossConfig = field(default_factory=lambda: ObsPredictionLossConfig(per_column=True))


class OutcomePredictionLossConfig(LossMetricConfig):
    loss_keys: Tuple[BinaryLossLiteral, ...] = ('bce', 'softmax_bce', 'balanced_focal_softmax_bce')


class OutcomePredictionLossMetric(LossMetric):
    config: OutcomePredictionLossConfig = field(default_factory=lambda: OutcomePredictionLossConfig())
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = OutcomePredictionLoss


class LeadPredictionLossConfig(ObsPredictionLossConfig):
    pass


class LeadPredictionLossMetric(ObsPredictionLossMetric):
    config: LeadPredictionLossConfig = field(default_factory=lambda: LeadPredictionLossConfig())
    prediction_loss_class: ClassVar[Type[PredictionLoss]] = LeadPredictionLoss


class LeadingPredictionAccuracyConfig(LeadingObservableExtractorConfig):
    aki_binary_index: int = field(kw_only=True)
    observable_code: str = field(default_factory=lambda: '')
    scheme: str = field(default_factory=lambda: '')
    leading_hours: List[float] = field(default_factory=lambda: [6.0, 12.0, 24.0, 36.0, 48.0])
    entry_neglect_window: float = 6.0
    minimum_acquisitions: int = 2  # minimum number of acquisitions to consider
    recovery_window: float = 12.0


class LeadingAKIPredictionAccuracy(Metric):
    config: LeadingPredictionAccuracyConfig

    def __post_init__(self):
        timestamp_class = [
            'negative', 'unknown', 'first_pre_emergence',
            'later_pre_emergence', 'recovery_window', 'recovered'
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
        self.estimands = tuple(fields)

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
        lead_critical_val = onp.nanmax(lead_prediction.value, axis=1)[self.config.minimum_acquisitions:]

        # criterion (2) - entry neglect window, early skip function.
        entry_neglect_mask = (lead_time > self.config.entry_neglect_window)
        if entry_neglect_mask.sum() == 0:
            return None
        else:
            lead_time = lead_time[entry_neglect_mask]
            lead_critical_val = lead_critical_val[entry_neglect_mask]

        obs_ground_truth = prediction.admission.observables.to_cpu()

        obs_index = self.config.aki_binary_index
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

        # criterion (4) & (5)- recovery neglect window and no recovery since last occurrence.
        prediction_df = prediction_df[
            (prediction_df['time'] > (prediction_df['last_recovery_time'] + self.config.recovery_window))]

        if len(prediction_df) == 0:
            return None

        return prediction_df

    def _lead_dataframes(self, predictions: AdmissionsPrediction):
        dataframes = []
        for prediction in predictions.sorted_predictions:
            df = self._lead_dataframe(prediction)
            if df is not None:
                dataframes.append(df)
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

    def apply(self, predictions: AdmissionsPrediction) -> Tuple[float, ...]:
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
        return tuple(map(lambda k: res.get(k, float('nan')), self.estimands))


class CodeAUC(Metric):
    estimands: Tuple[str, ...] = ('auc',)

    #
    def __call__(self, predictions: AdmissionsPrediction):
        ground_truth = []
        preds = []
        for ground_truth_outcome, predicted_outcome in predictions.iter_attr('outcome'):
            ground_truth.append(ground_truth_outcome.vec)
            preds.append(predicted_outcome.vec)

        ground_truth_mat = onp.vstack(ground_truth)
        predictions_mat = onp.vstack(preds)
        array = []
        for code_index in range(ground_truth_mat.shape[1]):
            code_ground_truth = ground_truth_mat[:, code_index]
            n_pos = code_ground_truth.sum()
            n_neg = len(code_ground_truth) - n_pos
            code_predictions = predictions_mat[:, code_index]
            if n_pos > 2 and n_neg > 2:
                array.append(compute_auc(code_ground_truth, code_predictions))
            else:
                array.append(onp.nan)
        return onp.array(array)


class AdmissionAUC(Metric):
    estimands: Tuple[str, ...] = ('auc',)

    def apply(self, predictions: AdmissionsPrediction) -> Tuple[Array, ...]:
        auc_values = [compute_auc(g_truth.vec, pred.vec) for g_truth, pred in predictions.iter_attr('outcome')]
        return onp.array(auc_values),


class CodeGroupTopAlarmAccuracyConfig(Config):
    partitions: List[List[int]]
    top_k_list: Tuple[int, ...] = (1, 3, 5, 10, 15, 20)

    def __post_init__(self):
        self.top_k_list = tuple(sorted(self.top_k_list))


class CodeGroupTopAlarmAccuracy(Metric):
    config: CodeGroupTopAlarmAccuracyConfig
    estimands: Tuple[str, ...] = ()

    def __post_init__(self):
        self.estimands = tuple(
            f'G{gi}k{k}_ACC' for gi in range(len(self.config.partitions)) for k in self.config.top_k_list)

    def apply(self, predictions: AdmissionsPrediction) -> Tuple[float, ...]:
        ground_truth = []
        preds = []
        for ground_truth_outcome, predicted_outcome in predictions.iter_attr('outcome'):
            preds.append(predicted_outcome.vec)
            ground_truth.append(ground_truth_outcome.vec)

        preds = onp.vstack(preds)
        ground_truth = onp.vstack(ground_truth).astype(bool)
        topk_risks = onp.argpartition(preds * -1, self.config.top_k_list, axis=1)
        true_positive = {}
        for k in self.config.top_k_list:
            topk_risks_i = topk_risks[:, :k]
            topk_risks_k = onp.zeros_like(preds, dtype=bool)
            onp.put_along_axis(topk_risks_k, topk_risks_i, True, 1)
            true_positive[k] = (topk_risks_k & ground_truth)
        acc = []
        for gi, code_indices in enumerate(self.config.partitions):
            group_true = ground_truth[:, tuple(code_indices)]
            for k in self.config.top_k_list:
                group_tp = true_positive[k][:, tuple(code_indices)]
                acc.append(group_tp.sum() / group_true.sum())

        return tuple(acc)


@dataclass
class MetricsCollectionOutput:
    metrics: Tuple[MetricsOutput, ...]
    codes: Tuple[Optional[Tuple[str, ...]], ...] = ()

    def __post_init__(self):
        if len(self.codes) == 0:
            self.codes = tuple([None] * len(self.metrics))

    @property
    def columns(self) -> List[str]:
        return sum((metric.columns(c) for metric, c in zip(self.metrics, self.codes)), [])

    @property
    def row(self) -> List[float]:
        return sum((metric.row for metric in self.metrics), [])

    def as_df(self, index: int | str = 0) -> pd.DataFrame:
        return pd.DataFrame([self.row], columns=self.columns, index=[index])


class MetricsCollection(Metric):
    metrics: Tuple[Metric, ...] = field(default_factory=tuple)
    codes: Tuple[Optional[Tuple[str, ...]], ...] = field(default_factory=tuple)
    estimands: Tuple[str, ...] = ()

    def __call__(self, predictions: AdmissionsPrediction) -> MetricsCollectionOutput:
        output = tuple(m(predictions) for m in self.metrics)
        return MetricsCollectionOutput(metrics=output, codes=self.codes)

    def apply(self, predictions: AdmissionsPrediction) -> Tuple[float, ...]:
        raise NotImplementedError


#     def filter_results(self, results: pd.DataFrame, models: List[str],
#                        min_auc: float):
#         models = sorted(models)
#         m_pairs = self._model_pairs(models)
#
#         # exclude tests with nans
#         pval_cols = [self.estimand_column_name()['pair'](pair, 'p_val') for pair in m_pairs]
#         valid_tests = results.loc[:, pval_cols].isnull().max(axis=1) == 0
#         results = results[valid_tests]
#         logging.info(
#             f'{sum(valid_tests)}/{len(valid_tests)} rows with valid tests (no NaN p-values)'
#         )
#
#         # AUC cut-off
#         auc_cols = [self.estimand_column_name()['model'](m, 'auc') for m in models]
#         accepted_aucs = results.loc[:, auc_cols].max(axis=1) > min_auc
#         results = results[accepted_aucs]
#         logging.info(f'{sum(accepted_aucs)}/{len(accepted_aucs)} rows with\
#             an AUC higher than {min_auc} by at least one model.')
#
#         return results
#
#     def insignificant_difference_rows(self, results: pd.DataFrame,
#                                       models: List[str], p_value: float):
#         models = sorted(models)
#         m_pairs = self._model_pairs(models)
#         pval_cols = [self.estimand_column_name()['pair'](pair, 'p_val') for pair in m_pairs]
#         insigificant = results.loc[:, pval_cols].min(axis=1) > p_value
#         return results[insigificant]

class DeLongTestOutput(MetricsOutput):
    name: str = field(default_factory=lambda: 'DeLongTest', kw_only=True)
    aggregations: ClassVar[Tuple[Tuple[str, Callable], ...]] = field(default_factory=tuple, kw_only=True)
    estimands: Tuple[str, ...] = field(default_factory=tuple, kw_only=True)
    values: Tuple[float | Array, ...] = field(default_factory=tuple, kw_only=True)

    code_n_pos: Array
    models_auc: Tuple[Array, ...]
    models_auc_var: Tuple[Array, ...]
    model_pairs_p_val: Tuple[Array, ...]
    models: Tuple[str, ...]
    model_pairs: Tuple[Tuple[str, str], ...]

    @property
    def dim(self) -> Optional[int]:
        return len(self.code_n_pos)

    def codes_dataframe(self, codes: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
        data = {'code_index': onp.arange(self.dim),
                'code': codes or list(range(self.dim)),
                'n_pos': self.code_n_pos}
        return pd.DataFrame(data)

    def models_dataframe(self, codes: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
        auc_array = onp.vstack(self.models_auc)
        auc_var_array = onp.vstack(self.models_auc_var)
        auc_columns = self.expand_metric_label(f'auc', codes)
        auc_var_columns = self.expand_metric_label(f'auc_var', codes)
        data = dict(zip(auc_columns, auc_array.T)) | dict(zip(auc_var_columns, auc_var_array.T))
        return pd.DataFrame(data, index=list(self.models))

    def model_pairs_dataframe(self, codes: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
        p_val_array = onp.vstack(self.model_pairs_p_val)
        data = dict(zip(self.expand_metric_label(f'p_val', codes), p_val_array.T))
        model_a, model_b = zip(*self.model_pairs)
        return pd.DataFrame(data | {
            'model_a': model_a,
            'model_b': model_b
        }, index=list(range(len(self.model_pairs)))).set_index(['model_a', 'model_b'])


class DeLongTest(Metric):
    estimands: Tuple[str, ...] = ()

    @staticmethod
    def model_pairs(models: Tuple[str, ...]) -> Tuple[Tuple[str, str], ...]:
        return tuple((models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models)))

    @staticmethod
    def extract_grountruth_vs_predictions(predictions: Dict[str, AdmissionsPrediction]) -> Tuple[
        Array, Dict[str, Array]]:
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

    def __call__(self, predictions: Dict[str, AdmissionsPrediction]) -> DeLongTestOutput:
        """
        Evaluate the AUC scores for each diagnosis code for each classifier. \
            In addition, conduct a pairwise test on the difference of AUC \
            scores between each pair of classifiers using DeLong test. \
            Codes that have either less than two positive cases or \
            have less than two negative cases are discarded \
            (AUC computation and difference test requirements).
        """
        # Classifier labels
        models = tuple(sorted(predictions.keys()))
        models_index = dict(zip(models, range(len(models))))
        model_pairs = self.model_pairs(models)
        true_mat, preds = self.extract_grountruth_vs_predictions(predictions)
        n_pos = true_mat.sum(axis=0)
        n_neg = len(true_mat) - n_pos
        auc = onp.full((len(models), true_mat.shape[1]), onp.nan)
        var = onp.full((len(models), true_mat.shape[1]), onp.nan)
        p_val = onp.full((len(model_pairs), true_mat.shape[1]), onp.nan)

        time_now = datetime.now()
        for i in tqdm(range(true_mat.shape[1])):
            code_truth = true_mat[:, i]
            # This is a requirement for pairwise testing.
            invalid_test = n_pos[i] < 2 or n_neg[i] < 2
            for pair_i, (model_a, model_b) in enumerate(model_pairs):
                if n_pos[i] < 2 or n_neg[i] < 2:
                    continue
                preds_a = preds[model_a][:, i]
                preds_b = preds[model_b][:, i]
                auc_a, auc_b, var_a, var_b, p_val = FastDeLongTest.delong_roc_test(code_truth, preds_a, preds_b)
                p_val[pair_i, i] = p_val
                auc[models_index[model_a], i] = auc_a
                auc[models_index[model_b], i] = auc_b
                var[models_index[model_a], i] = var_a
                var[models_index[model_b], i] = var_b

        return DeLongTestOutput(
            models=models,
            model_pairs=model_pairs,
            code_n_pos=n_pos,
            models_auc=tuple(auc),
            models_auc_var=tuple(var),
            model_pairs_p_val=tuple(p_val),
            time_elapsed=(datetime.now() - time_now).total_seconds(),
            name=type(self).__name__,
            estimands=(),
            values=()
        )
