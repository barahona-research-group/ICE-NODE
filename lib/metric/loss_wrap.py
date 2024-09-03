import logging
from abc import ABCMeta, abstractmethod
from dataclasses import field
from functools import cached_property
from typing import Optional, Callable, ClassVar, List

import jax.numpy as jnp

from .loss import NumericLossLiteral, NUMERIC_LOSS, BinaryLossLiteral, PROB_BINARY_LOSS, ProbNumericLossLiteral, \
    ProbLossSignature, PROB_NUMERIC_LOSS, LOGITS_BINARY_LOSS
from ..base import Array, np_module, VxDataItem, Module, Config
from ..ehr import InpatientObservables, CodesVector
from ..ehr.tvx_concepts import ObservablesDistribution
from ..ml.artefacts import PredictionAttribute, AdmissionsPrediction


class PredictionLoss(Module, metaclass=ABCMeta):
    config: Config = field(default_factory=Config)
    prediction_attribute: ClassVar[PredictionAttribute] = None
    admission_attribute: ClassVar[PredictionAttribute] = None
    per_column: bool = False

    @abstractmethod
    def item_loss(self, ground_truth: VxDataItem, prediction: VxDataItem) -> Array:
        raise NotImplementedError

    def item_weight(self, ground_truth: VxDataItem) -> float:
        return 1.0

    def aggregate_loss(self, predictions: AdmissionsPrediction) -> Array | float:
        weights = [self.item_weight(gt) for gt in
                   predictions.list_attr(self.prediction_attribute, self.admission_attribute)[0]]
        weights = [w / sum(weights) for w in weights]

        losses = jnp.array([(self.item_loss(gt, pred) * w if w > 0. else 0.) for (gt, pred), w in
                            zip(predictions.iter_attr(self.prediction_attribute, self.admission_attribute), weights)])
        loss = jnp.nansum(losses)

        if jnp.isnan(loss):
            logging.warning('NaN obs loss detected')
            return 0.
        return loss

    def per_column_loss(self, ground_truths: List[VxDataItem], predictions: List[VxDataItem]) -> Array:
        raise NotImplementedError

    def __call__(self, predictions: AdmissionsPrediction) -> Array | float:
        if not self.per_column:
            return self.aggregate_loss(predictions)
        else:
            gts, preds = zip(*predictions.iter_attr(self.prediction_attribute, self.admission_attribute))
            return self.per_column_loss(gts, preds)


class NumericPredictionLoss(PredictionLoss):
    loss_key: NumericLossLiteral = field(default=None, kw_only=True)

    @cached_property
    def raw_loss(self) -> Callable[[Array, Array, Array, Optional[int]], Array]:
        return NUMERIC_LOSS[self.loss_key]

    def item_loss(self, ground_truth: InpatientObservables, prediction: InpatientObservables) -> Array:
        ground_truth_val = np_module(ground_truth.value).nan_to_num(ground_truth.value, nan=0.0)
        if ground_truth_val.dtype == bool:
            ground_truth_val = ground_truth_val.astype(float)
        return self.raw_loss(ground_truth_val, prediction.value, ground_truth.mask, None)

    def per_column_loss(self, ground_truths: List[InpatientObservables],
                        predictions: List[InpatientObservables]) -> Array:
        predictions_vecs = []
        ground_truth_vecs = []
        masks_vecs = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            predictions_vecs.append(prediction.value)
            ground_truth_vecs.append(ground_truth.value)
            masks_vecs.append(ground_truth.mask)
        predictions_array = jnp.vstack(predictions_vecs)
        ground_truth_array = jnp.vstack(ground_truth_vecs)
        masks_array = jnp.vstack(masks_vecs)
        return self.raw_loss(ground_truth_array, predictions_array, masks_array, 0)

    def item_weight(self, ground_truth: InpatientObservables) -> float:
        return ground_truth.mask.sum()


class LeadPredictionLoss(NumericPredictionLoss):
    loss_key: NumericLossLiteral | BinaryLossLiteral = field(default=None, kw_only=True)
    prediction_attribute: ClassVar[PredictionAttribute] = 'leading_observable'

    @cached_property
    def raw_loss(self) -> Callable[[Array, Array, Array], Array]:
        if self.loss_key in PROB_BINARY_LOSS:
            return PROB_BINARY_LOSS[self.loss_key]
        else:
            return NUMERIC_LOSS[self.loss_key]


class ObsPredictionLoss(NumericPredictionLoss):
    prediction_attribute: ClassVar[PredictionAttribute] = 'observables'

class ImputationLoss(NumericPredictionLoss):
    admission_attribute: ClassVar[PredictionAttribute] = 'observables'
    prediction_attribute: ClassVar[PredictionAttribute] = 'imputed_observables'

class ProbObsPredictionLoss(ObsPredictionLoss):
    loss_key: ProbNumericLossLiteral = field(default=None, kw_only=True)
    prediction_attribute: ClassVar[PredictionAttribute] = 'observables'

    def item_loss(self, ground_truth: InpatientObservables, prediction: ObservablesDistribution) -> Array:
        ground_truth_val = np_module(ground_truth.value).nan_to_num(ground_truth.value, nan=0.0)
        if ground_truth_val.dtype == bool:
            ground_truth_val = ground_truth_val.astype(float)
        ground_truth_std = ground_truth_val * 0.0 + 0.1
        return self.raw_loss((ground_truth_val, ground_truth_std), (prediction.mean, prediction.std), ground_truth.mask,
                             None)

    @cached_property
    def raw_loss(self) -> ProbLossSignature:
        return PROB_NUMERIC_LOSS[self.loss_key]

    def per_column_loss(self, ground_truths: List[InpatientObservables],
                        predictions: List[ObservablesDistribution]) -> Array:
        predictions_mean_vecs = []
        predictions_std_vecs = []
        ground_truth_mean_vecs = []
        ground_truth_std_vecs = []
        masks_vecs = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            predictions_mean_vecs.append(prediction.mean)
            predictions_std_vecs.append(prediction.std)
            ground_truth_mean_vecs.append(ground_truth.value)
            ground_truth_std_vecs.append(ground_truth.value * 0.0 + 0.1)
            masks_vecs.append(ground_truth.mask)

        predictions_mean_array = jnp.vstack(predictions_mean_vecs)
        predictions_std_array = jnp.vstack(predictions_std_vecs)
        ground_truth_mean_array = jnp.vstack(ground_truth_mean_vecs)
        if ground_truth_mean_array.dtype == bool:
            ground_truth_mean_array = ground_truth_mean_array.astype(float)

        ground_truth_std_array = jnp.vstack(ground_truth_std_vecs)
        masks_array = jnp.vstack(masks_vecs)

        return self.raw_loss((ground_truth_mean_array, ground_truth_std_array),
                             (predictions_mean_array, predictions_std_array), masks_array, 0)


class AdjustedProbObsPredictionLoss(ProbObsPredictionLoss):
    prediction_attribute: ClassVar[PredictionAttribute] = 'adjusted_observables'
    admission_attribute: ClassVar[PredictionAttribute] = 'observables'


class OutcomePredictionLoss(PredictionLoss):
    loss_key: BinaryLossLiteral = field(default=None, kw_only=True)
    prediction_attribute: ClassVar[PredictionAttribute] = 'outcome'

    @cached_property
    def raw_loss(self) -> Callable[[Array, Array, Optional[int]], Array]:
        return LOGITS_BINARY_LOSS[self.loss_key]

    def item_loss(self, ground_truth: CodesVector, prediction: CodesVector) -> Array:
        ground_truth_vec = ground_truth.vec
        if ground_truth_vec.dtype == bool:
            ground_truth_vec = ground_truth_vec.astype(float)
        return self.raw_loss(ground_truth_vec, prediction.vec, None)

    def per_column_loss(self, predictions: List[CodesVector], ground_truths: List[CodesVector]) -> Array:
        predictions_vecs = []
        ground_truth_vecs = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            predictions_vecs.append(prediction.vec)
            ground_truth_vecs.append(ground_truth.vec)
        predictions_array = jnp.vstack(predictions_vecs)
        ground_truth_array = jnp.vstack(ground_truth_vecs)
        return self.raw_loss(ground_truth_array, predictions_array, 0)
