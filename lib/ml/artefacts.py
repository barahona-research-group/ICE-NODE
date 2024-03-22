from __future__ import annotations

import logging
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Optional, Union, Tuple, Callable, Any, Iterable

import equinox as eqx
import numpy as np
import tables as tbl
from jax import numpy as jnp

from lib import Config
from lib.base import VxData
from lib.ehr import Admission, CodesVector, InpatientObservables
from lib.ehr.tvx_ehr import outcome_first_occurrence
from lib.utils import tree_hasnan


class PatientAdmissionTrajectory(VxData):
    observables: Optional[InpatientObservables] = None
    leading_observable: Optional[InpatientObservables] = None


class ModelBehaviouralMetrics(VxData):
    pass


class AdmissionPrediction(VxData):
    subject_id: str
    admission: Admission # ground_truth
    observables: Optional[InpatientObservables] = None
    leading_observable: Optional[InpatientObservables] = None
    outcome: Optional[CodesVector] = None
    trajectory: PatientAdmissionTrajectory = field(default_factory=PatientAdmissionTrajectory)
    model_behavioural_metrics: ModelBehaviouralMetrics

    def has_nans(self):
        return tree_hasnan((self.observables, self.leading_observable, self.outcome))


class AdmissionPredictionCollection(VxData):
    predictions: Tuple[AdmissionPrediction, ...] = field(default_factory=tuple)
    """
    A dictionary-like class for storing and manipulating prediction data.

    This class extends the built-in `dict` class and provides additional methods
    for saving, loading, filtering, and analyzing prediction data.


    Methods:
        save: save the predictions to a file.
        load: load predictions from a file.
        add: add a prediction to the collection.
        subject_ids: get a list of subject IDs in the predictions.
        get_predictions: get a list of predictions for the specified subject IDs.
        associative_regularisation: calculate the associative regularization term.
        get_patient_trajectories: get the trajectories of patients in the predictions.
        average_interval_hours: calculate the average interval hours of the predictions.
        filter_nans: filter out admission predictions with NaN values.
        has_dx_predictions: check if the predictions have diagnosis predictions.
        has_auxiliary_loss: check if the predictions have auxiliary loss.
        prediction_dx_loss: calculate the diagnosis loss (the predicted outcome) of the predictions.
        predicted_observables_list: get a list of *predicted* observables.
        predicted_leading_observables_list: get a list of *predicted* leading observables.
        observables_list: get a list of *ground-truth* observables in the predictions.
        leading_observables_list: get a list of *ground-truth* leading observables in the predictions.
        prediction_obs_loss: calculate the observation loss of the predictions.
        prediction_lead_loss: calculate the leading observation loss of the predictions.
        prediction_lead_data: get the data for leading observation loss calculation.
        outcome_first_occurrence_masks: get the masks for the first occurrence of outcomes.
    """

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
    def load(path: Union[str, Path]) -> AdmissionPredictionCollection:
        """
        Load predictions from a file.

        Args:
            path: the path to the file containing the predictions.

        Returns:
            The loaded predictions.
        """
        with tbl.open_file(str(Path(path).with_suffix('.h5')), 'r') as store:
            return AdmissionPredictionCollection.from_hdf_group(store.root)

    def add(self, *args, **kwargs) -> AdmissionPredictionCollection:
        return eqx.tree_at(lambda p: p.predictions, self, self.predictions + (AdmissionPrediction(*args, **kwargs),))

    def __iter__(self):
        return iter(self.predictions)

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

    def filter_nans(self) -> AdmissionPredictionCollection:
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
        clean_predictions = AdmissionPredictionCollection(cleaned)
        if nan_detected:
            logging.warning(f'Average interval_hours: {clean_predictions.average_interval_hours}')

        if len(cleaned) == 0 and len(self) > 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return clean_predictions


    def prediction_dx_loss(self, dx_loss):
        """
        Calculate the diagnosis loss of the predictions.

        Args:
            dx_loss: the diagnosis loss function.

        Returns:
            The diagnosis loss.
        """
        if not self.has_dx_predictions:
            return 0.

        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        l_outcome = [a.outcome.vec for a in adms]
        l_pred = [p.outcome.vec for p in preds]
        l_mask = [jnp.ones_like(a.outcome.vec, dtype=bool) for a in adms]
        loss_v = jnp.array(list(map(dx_loss, l_outcome, l_pred, l_mask)))
        loss_v = jnp.nanmean(loss_v)
        return jnp.where(jnp.isnan(loss_v), 0., loss_v)

    @property
    def predicted_observables_list(self):
        """
        Get a list of *predicted* observables.

        Returns:
            A list of *predicted* observables.
        """
        preds = self.get_predictions()
        obs_l = []

        for sid in sorted(self.keys()):
            for aid in sorted(self[sid].keys()):
                obs = self[sid][aid].observables
                if isinstance(obs, list):
                    obs_l.extend(obs)
                else:
                    obs_l.append(obs)

        return obs_l

    @property
    def predicted_leading_observables_list(self):
        """
        Get a list of *predicted* leading observables.

        Returns:
            A list of *predicted* leading observables.
        """
        preds = self.get_predictions()
        obs_l = []

        for sid in sorted(self.keys()):
            for aid in sorted(self[sid].keys()):
                obs = self[sid][aid].leading_observable
                if isinstance(obs, list):
                    obs_l.extend(obs)
                else:
                    obs_l.append(obs)

        return obs_l

    @property
    def observables_list(self):
        """
        Get a list of *ground-truth* observables in the predictions.

        Returns:
            A list of *ground-truth*  observables.
        """
        preds = self.get_predictions()
        obs_l = []

        for sid in sorted(self.keys()):
            for aid in sorted(self[sid].keys()):
                obs = self[sid][aid].admission.observables
                if isinstance(obs, list):
                    obs_l.extend(obs)
                else:
                    obs_l.append(obs)

        return obs_l

    @property
    def leading_observables_list(self):
        """
        Get a list of *ground-truth* leading observables in the predictions.

        Returns:
            A list of *ground-truth* leading observables.
        """
        preds = self.get_predictions()
        obs_l = []

        for sid in sorted(self.keys()):
            for aid in sorted(self[sid].keys()):
                obs = self[sid][aid].admission.leading_observable
                if isinstance(obs, list):
                    obs_l.extend(obs)
                else:
                    obs_l.append(obs)

        return obs_l

    def prediction_obs_loss(self, obs_loss):
        """
        Calculate the observation loss of the predictions.

        Args:
            obs_loss: the observation loss function.

        Returns:
            The observation loss.
        """
        l_true = [obs.value for obs in self.observables_list]
        l_mask = [obs.mask for obs in self.observables_list]
        l_pred = [obs.value for obs in self.predicted_observables_list]

        true = jnp.vstack(l_true)
        mask = jnp.vstack(l_mask)
        pred = jnp.vstack(l_pred)
        loss_v = obs_loss(true, pred, mask)
        if jnp.isnan(loss_v):
            logging.warning('NaN obs loss detected')
        return jnp.where(jnp.isnan(loss_v), 0., loss_v)

    def prediction_lead_loss(self, lead_loss):
        """
        Calculate the leading observation loss of the predictions.

        Args:
            lead_loss: the leading observation loss function.

        Returns:
            The leading observation loss.
        """
        loss_v = []
        weight_v = []
        for pred in self.get_predictions():
            adm = pred.admission
            if isinstance(pred.leading_observable, list):
                pred_l = pred.leading_observable
            else:
                pred_l = [pred.leading_observable]

            if isinstance(adm.leading_observable, list):
                adm_l = adm.leading_observable
            else:
                adm_l = [adm.leading_observable]

            assert len(pred_l) == len(adm_l)

            for pred_lo, adm_lo in zip(pred_l, adm_l):
                for i in range(len(adm_lo.time)):
                    y = adm_lo.value[i]
                    y_hat = pred_lo.value[i]
                    mask = adm_lo.mask[i]
                    loss_v.append(lead_loss(y, y_hat, mask))
                    weight_v.append(mask.sum())

        loss_v = jnp.array(loss_v)
        weight_v = jnp.array(weight_v)
        weight_v = jnp.where(jnp.isnan(loss_v), 0.0, weight_v)
        if weight_v.sum() == 0:
            return 0.

        weight_v = weight_v / weight_v.sum()
        return jnp.nansum(loss_v * weight_v)

    def prediction_lead_data(self, obs_index):
        """
        Get the data for leading observation loss calculation.

        Args:
            obs_index: the coding scheme index of the particular observation in leading_observable.

        Returns:
            The data for leading observation loss calculation.
        """
        preds = self.get_predictions()

        y = []
        y_hat = []
        mask = []

        obs = []
        obs_mask = []
        for pred in preds:
            adm = pred.admission
            if isinstance(pred.leading_observable, list):
                pred_leading_observable = pred.leading_observable
            else:
                pred_leading_observable = [pred.leading_observable]

            if isinstance(adm.leading_observable, list):
                adm_leading_observable = adm.leading_observable
            else:
                adm_leading_observable = [adm.leading_observable]

            if isinstance(adm.observables, list):
                adm_observables = adm.observables
            else:
                adm_observables = [adm.observables]

            for pred_lo, adm_lo, adm_obs in zip(pred_leading_observable,
                                                adm_leading_observable,
                                                adm_observables):
                for i in range(len(adm_lo.time)):
                    mask.append(adm_lo.mask[i])
                    y.append(adm_lo.value[i])
                    y_hat.append(pred_lo.value[i])
                    obs.append(adm_obs.value[i][obs_index])
                    obs_mask.append(adm_obs.mask[i][obs_index])

        y = jnp.vstack(y)
        y_hat = jnp.vstack(y_hat)
        mask = jnp.vstack(mask)
        obs = jnp.vstack(obs)
        obs_mask = jnp.vstack(obs_mask)
        return {
            'y': y,
            'y_hat': y_hat,
            'mask': mask,
            'obs': obs,
            'obs_mask': obs_mask,
        }

    def outcome_first_occurrence_masks(self, subject_id):
        preds = self[subject_id]
        adms = [preds[aid].admission for aid in sorted(preds.keys())]
        first_occ_adm_id = outcome_first_occurrence(adms)
        return [first_occ_adm_id == a.admission_id for a in adms]


class TrajectoryConfig(Config):
    sampling_rate: float = 0.5  # 0.5 means every 30 minutes.
