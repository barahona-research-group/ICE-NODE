from __future__ import annotations

import logging
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Optional, Union, Tuple, Callable, Any, Iterable, Literal

import equinox as eqx
import numpy as np
import tables as tbl
from jax import numpy as jnp

from lib import Config
from lib.base import VxData, VxDataItem, Array, Module
from lib.ehr import Admission, CodesVector, InpatientObservables
from lib.utils import tree_hasnan


class PatientAdmissionTrajectory(VxData):
    observables: Optional[InpatientObservables] = None
    leading_observable: Optional[InpatientObservables] = None


class ModelBehaviouralMetrics(VxData):
    pass


class TrajectoryConfig(Config):
    sampling_rate: float = 0.5  # 0.5 means every 30 minutes.


PredictionAttribute = Literal['outcome', 'observables', 'leading_observable']


class AdmissionPrediction(VxData):
    subject_id: str
    admission: Admission  # ground_truth
    observables: Optional[InpatientObservables] = None
    leading_observable: Optional[InpatientObservables] = None
    outcome: Optional[CodesVector] = None
    trajectory: PatientAdmissionTrajectory = field(default_factory=PatientAdmissionTrajectory)
    model_behavioural_metrics: ModelBehaviouralMetrics = field(default_factory=ModelBehaviouralMetrics)

    def has_nans(self):
        return tree_hasnan((self.observables, self.leading_observable, self.outcome))


# Loss aggregation types.
# 1. 'concat': concatenate the predictions, pass to the loss function, receive then return a scalar.
#   Rank-based loss should follow this type.
# 2. 'mean': accumulate the loss for each prediction, return an averaging scalar.
# 3. 'struct_time': apply the loss function to temporally structured data, return a scalar for every temporal vector
#   pairs (i.e. the ground-truth vs. prediction). DTW-based loss should follow this type.
# 4. 'struct_prob_div': apply the loss function to structured data with probability distributions, return a scalar
#   for every probability distribution parameter set pairs. KL-divergence-based loss should follow this type.
# 5. 'struct_prob_div_time': apply the loss function to structured data with probability distributions and temporally
#   structured data, return a scalar for every temporal vector pairs (i.e. the ground-truth vs. prediction).
#       KL-divergence-based loss with DTW should follow this type.

LossAggreagationType = Literal['concat', 'mean', 'struct_time', 'struct_prob_div', 'struct_prob_div_time']


class LossWrapperConfig(Config):
    aggregation: LossAggreagationType = 'mean'

    def __post_init__(self):
        if self.aggregation not in ['concat', 'mean']:
            if self.aggregation not in ['struct_time', 'struct_prob_div', 'struct_prob_div_time']:
                raise NotImplementedError(f'Aggregation type {self.aggregation} not implemented yet.')
            else:
                raise ValueError(f'Invalid aggregation type: {self.aggregation}.')


class LossWrapper(Module):
    config: LossWrapperConfig
    loss_fn: Callable[[VxDataItem, VxDataItem], Array]
    concatenate: Optional[Callable[[Iterable[VxDataItem]], VxDataItem]] = None

    def __post_init__(self):
        if self.config.aggregation == 'concat' and self.concatenate is None:
            raise ValueError('concatenation function must be provided for aggregation type "concat"')

    @staticmethod
    def codes_concat_wrapper(loss_fn: Callable[[Array, Array], Array]) -> LossWrapper:
        return LossWrapper(LossWrapperConfig('concat'), loss_fn, lambda l: jnp.hstack([v.vec for v in l]))

    @staticmethod
    def observables_mean_wrapper(loss_fn: Callable[[Array, Array, Array], Array]) -> LossWrapper:
        def obs_loss(obs_true: InpatientObservables, obs_pred: InpatientObservables) -> Array:
            return loss_fn(obs_true.value, obs_pred.value, obs_true.mask)

        return LossWrapper(LossWrapperConfig('mean'), obs_loss, None)

    def __call__(self, ground_truth: Iterable[VxDataItem], predictions: Iterable[VxDataItem]) -> Array:
        if self.config.aggregation == 'concat':
            loss = self.loss_fn(self.concatenate(ground_truth), self.concatenate(predictions))
        elif self.config.aggregation == 'mean':
            losses = jnp.array([self.loss_fn(gt, pred) for gt, pred in zip(ground_truth, predictions)])
            loss = jnp.nanmean(losses)
        else:
            raise NotImplementedError(f'Aggregation type {self.config.aggregation} not implemented yet.')

        if jnp.isnan(loss):
            logging.warning('NaN obs loss detected')

        return jnp.where(jnp.isnan(loss), 0., loss)


class AdmissionsPrediction(VxData):
    predictions: Tuple[AdmissionPrediction, ...] = field(default_factory=tuple)

    def save(self, path: Union[str, Path]):
        """
        Save the predictions to a file.

        Args:
            path: the path to the file where the predictions will be saved.

        Returns:
            None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tbl.open_file(str(path.with_suffix('.h5')), 'w') as store:
            return self.to_hdf_group(store.root)

    @staticmethod
    def load(path: Union[str, Path]) -> AdmissionsPrediction:
        """
        Load predictions from a file.

        Args:
            path: the path to the file containing the predictions.

        Returns:
            The loaded predictions.
        """
        with tbl.open_file(str(Path(path).with_suffix('.h5')), 'r') as store:
            return AdmissionsPrediction.from_hdf_group(store.root)

    def add(self, *args, **kwargs) -> AdmissionsPrediction:
        return eqx.tree_at(lambda p: p.predictions, self, self.predictions + (AdmissionPrediction(*args, **kwargs),))

    def __iter__(self) -> Iterable[AdmissionPrediction]:
        return iter(self.predictions)

    def iter_attr(self, attr: str) -> Iterable[Tuple[VxDataItem, VxDataItem]]:
        return ((getattr(p.admission, attr), getattr(p, attr)) for p in self)

    def list_attr(self, attr: str) -> Tuple[Tuple[VxDataItem, ...], Tuple[VxDataItem, ...]]:
        ground_truth, predictions = tuple(zip(*self.iter_attr(attr)))
        return ground_truth, predictions

    def __len__(self):
        return len(self.predictions)

    def aggregate(self, operand: Callable[[AdmissionPrediction], Any],
                  aggregation: Callable[[Iterable[Any]], Any]) -> Any:
        return aggregation([operand(p) for p in self])

    @cached_property
    def average_interval_hours(self):
        """
        Calculate the average interval hours of the predictions.

        Returns:
            The average interval hours.
        """
        return self.aggregate(lambda p: p.admission.interval_hours, lambda l: np.mean(np.array(l)))

    def filter_nans(self) -> AdmissionsPrediction:
        """
        Filter out predictions with NaN values.

        Returns:
            the filtered predictions (a copy of the original predictions).
        """
        if len(self) == 0:
            return self

        cleaned = tuple()
        nan_detected = False
        for p in self:
            if not p.has_nans():
                cleaned += (p,)
            else:
                nan_detected = True
                logging.warning('Skipping prediction with NaNs: '
                                f'subject_id={p.subject_id}, admission_id={p.admission.admission_id} '
                                f'interval_hours= {p.admission.interval_hours}. '
                                'Note: long intervals is a likely reason to destabilise the model')
        clean_predictions = AdmissionsPrediction(cleaned)
        if nan_detected:
            logging.warning(f'Average interval_hours: {clean_predictions.average_interval_hours}')

        if len(cleaned) == 0 and len(self) > 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return clean_predictions

    def apply_loss(self, attribute: PredictionAttribute, loss: LossWrapper) -> Array:
        """
        Calculate the loss of the predictions.

        Args:
            attribute: the attribute to calculate the loss for.
            loss: the loss function.

        Returns:
            The loss.
        """
        ground_truth, predictions = self.list_attr(attribute)
        return loss(ground_truth, predictions)
