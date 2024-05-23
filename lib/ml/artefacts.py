from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Optional, Union, Tuple, Callable, Any, Iterable, Literal, Dict, List

import equinox as eqx
import numpy as np
import tables as tbl

from lib.base import VxData, VxDataItem
from lib.ehr import Admission, CodesVector, InpatientObservables
from lib.utils import tree_hasnan


class ModelBehaviouralMetrics(VxData):
    pass


PredictionAttribute = Literal['outcome', 'observables', 'leading_observable', 'adjusted_observables']


class AdmissionPrediction(VxData):
    admission: Optional[Admission]
    observables: Optional[InpatientObservables] = None
    leading_observable: Optional[InpatientObservables] = None
    outcome: Optional[CodesVector] = None
    trajectory: Optional[InpatientObservables] = None
    model_behavioural_metrics: Optional[ModelBehaviouralMetrics] = None

    def add(self, **kwargs) -> AdmissionPrediction:
        return eqx.combine(self, type(self)(admission=None, **kwargs),
                           is_leaf=lambda x: x is None or isinstance(x, (Admission,
                                                                         InpatientObservables, CodesVector,
                                                                         ModelBehaviouralMetrics)))

    def has_nans(self):
        return tree_hasnan((self.observables, self.leading_observable, self.outcome))


class AdmissionPredictionPair(VxData):
    subject_id: str
    prediction: AdmissionPrediction


class AdmissionsPrediction(VxData):
    predictions: Tuple[AdmissionPredictionPair, ...] = field(default_factory=tuple)

    @cached_property
    def subject_predictions(self) -> Dict[str, List[AdmissionPrediction]]:
        dictionary = defaultdict(list)
        for prediction in self.predictions:
            dictionary[prediction.subject_id].append(prediction.prediction)
        return dictionary

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

    def add(self, subject_id: str, prediction: AdmissionPrediction) -> AdmissionsPrediction:
        prediction_item = AdmissionPredictionPair(subject_id=subject_id, prediction=prediction)
        return eqx.tree_at(lambda p: p.predictions, self, self.predictions + (prediction_item,))

    @cached_property
    def sorted_predictions(self) -> Tuple[AdmissionPrediction, ...]:
        prediction_list = []
        for subject_id in sorted(self.subject_predictions.keys()):
            for prediction in sorted(self.subject_predictions[subject_id], key=lambda p: p.admission.admission_id):
                prediction_list.append(prediction)
        return tuple(prediction_list)

    def __iter__(self) -> Iterable[AdmissionPrediction]:
        return iter(self.sorted_predictions)

    def iter_attr(self, attr: str, adm_attr: Optional[str] = None) -> Iterable[Tuple[VxDataItem, VxDataItem]]:
        if adm_attr is None:
            adm_attr = attr
        return ((getattr(p.admission, adm_attr), getattr(p, attr)) for p in self if getattr(p, attr) is not None)

    def list_attr(self, attr: str, adm_attr: Optional[str] = None) -> Tuple[Tuple[VxDataItem, ...], Tuple[VxDataItem, ...]]:
        ground_truth, predictions = tuple(zip(*self.iter_attr(attr, adm_attr)))
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
        for p in self.predictions:
            if not p.prediction.has_nans():
                cleaned += (p,)
            else:
                nan_detected = True
                logging.warning('Skipping prediction with NaNs: '
                                f'subject_id={p.subject_id}, admission_id={p.prediction.admission.admission_id} '
                                f'interval_hours= {p.prediction.admission.interval_hours}. '
                                'Note: long intervals is a likely reason to destabilise the model')
        clean_predictions = AdmissionsPrediction(predictions=cleaned)
        if nan_detected:
            logging.warning(f'Average interval_hours: {clean_predictions.average_interval_hours}')

        if len(cleaned) == 0 and len(self) > 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return clean_predictions
