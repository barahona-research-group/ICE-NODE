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


class PredictedTrajectory(VxData):
    pass


PredictionAttribute = Literal['outcome', 'observables', 'leading_observable']


class AdmissionPrediction(VxData):
    admission: Optional[Admission] = None  # ground_truth
    observables: Optional[InpatientObservables] = None
    leading_observable: Optional[InpatientObservables] = None
    outcome: Optional[CodesVector] = None
    trajectory: Optional[PredictedTrajectory] = None
    model_behavioural_metrics: Optional[ModelBehaviouralMetrics] = None

    def has_nans(self):
        return tree_hasnan((self.observables, self.leading_observable, self.outcome))


class AdmissionsPrediction(VxData):
    predictions: Tuple[Tuple[str, AdmissionPrediction], ...] = field(default_factory=tuple)

    @cached_property
    def subject_predictions(self) -> Dict[str, List[AdmissionPrediction]]:
        dictionary = defaultdict(list)
        for subject_id, prediction in self.predictions:
            dictionary[subject_id].append(prediction)
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
        prediction_item = (subject_id, prediction)
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
        for subject_id, p in self.predictions:
            if not p.has_nans():
                cleaned += ((subject_id, p),)
            else:
                nan_detected = True
                logging.warning('Skipping prediction with NaNs: '
                                f'subject_id={subject_id}, admission_id={p.admission.admission_id} '
                                f'interval_hours= {p.admission.interval_hours}. '
                                'Note: long intervals is a likely reason to destabilise the model')
        clean_predictions = AdmissionsPrediction(cleaned)
        if nan_detected:
            logging.warning(f'Average interval_hours: {clean_predictions.average_interval_hours}')

        if len(cleaned) == 0 and len(self) > 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return clean_predictions
