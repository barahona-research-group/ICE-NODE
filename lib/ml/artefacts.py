from __future__ import annotations

import logging
import pickle
from copy import deepcopy
from pathlib import Path

from typing import Optional, List, Dict, Union

import equinox as eqx
import numpy as np
from jax import numpy as jnp, tree_util as jtu

from lib import Config
from lib.base import Data
from lib.ehr import Admission, CodesVector, InpatientObservables
from lib.ehr.tvx_ehr import outcome_first_occurrence
from lib.utils import tree_hasnan, tqdm_constructor


class AdmissionPrediction(Data):
    """
    Represents predictions associated with an admission in a healthcare system.

    Attributes:
        admission (Admission): the admission information.
        outcome (Optional[CodesVector]): the *predicted* outcome.
        observables (Optional[List[InpatientObservables]]): the list of *predicted* inpatient observables/measurements.
        leading_observable (Optional[List[InpatientObservables]]): the list of *predicted* time-leading inpatient observables/measurements.
        trajectory (Optional[PatientTrajectory]): the patient *predicted* trajectory, which is recorded based on TrajectoryConfig configuration.
        associative_regularisation (Optional[Dict[str, jnp.ndarray]]): the associative regularisation data through the inference.
        auxiliary_loss (Optional[Dict[str, jnp.ndarray]]): some auxiliary loss information.
        other (Optional[Dict[str, jnp.ndarray]]): other additional data.

    Methods:
        has_nans(): checks if any of the attributes contain NaN values.
        defragment_observables(): in case each of `observables` and `leading_observable` are sharded over lists, combine them into one object for each.
    """

    admission: Admission
    outcome: Optional[CodesVector] = None
    observables: Optional[List[InpatientObservables]] = None
    leading_observable: Optional[List[InpatientObservables]] = None
    trajectory: Optional[PatientTrajectory] = None
    associative_regularisation: Optional[Dict[str, jnp.ndarray]] = None
    auxiliary_loss: Optional[Dict[str, jnp.ndarray]] = None
    other: Optional[Dict[str, jnp.ndarray]] = None

    def has_nans(self):
        return tree_hasnan((self.outcome, self.observables, self.other,
                            self.leading_observable))

    def defragment_observables(self):
        updated = self
        updated = eqx.tree_at(lambda x: x.observables, updated,
                              InpatientObservables.concat(updated.observables))
        updated = eqx.tree_at(
            lambda x: x.leading_observable, updated,
            InpatientObservables.concat(updated.leading_observable))
        updated = eqx.tree_at(
            lambda x: x.admission.observables, updated,
            InpatientObservables.concat(updated.admission.observables))
        updated = eqx.tree_at(
            lambda x: x.admission.leading_observable, updated,
            InpatientObservables.concat(updated.admission.leading_observable))
        return updated


class Predictions(dict):
    """
    A dictionary-like class for storing and manipulating prediction data.

    This class extends the built-in `dict` class and provides additional methods
    for saving, loading, filtering, and analyzing prediction data.


    Methods:
        save: save the predictions to a file.
        load: load predictions from a file.
        add: add a prediction to the collection.
        defragment_observables: defragment the sharded observables in each admission prediction.
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
        path = path.with_suffix('.pickle')
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: Union[str, Path]) -> jtu.pytree:
        """
        Load predictions from a file.

        Args:
            path: the path to the file containing the predictions.

        Returns:
            The loaded predictions.
        """
        path = Path(path)
        with open(path.with_suffix('.pickle'), 'rb') as file:
            return pickle.load(file)

    def add(self, subject_id: str, prediction: AdmissionPrediction) -> None:
        """
        Add an admission prediction to the collection.

        Args:
            subject_id: the ID of the subject.
            prediction: the prediction to be added.

        Returns:
            None
        """
        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][prediction.admission.admission_id] = prediction

    def defragment_observables(self):
        """
        Defragment the sharded observables in each admission prediction.

        Returns:
            the defragmented predictions (a copy of the original predictions).
        """
        preds = deepcopy(self)
        for sid in preds.keys():
            for aid in preds[sid].keys():
                assert isinstance(preds[sid][aid].observables, list), \
                    'Observables must be a list'
                assert isinstance(preds[sid][aid].leading_observable, list), \
                    'Leading observables must be a list'
                preds[sid][aid] = self[sid][aid].defragment_observables()

        return preds

    def _defragment_observables(self):
        """
        Internal method for defragmenting observables.

        Returns:
            None
        """
        for sid in tqdm_constructor(self.keys()):
            for aid in self[sid].keys():
                self[sid][aid] = self[sid][aid].defragment_observables()

    @property
    def subject_ids(self):
        """
        Get a list of subject IDs in the predictions.

        Returns:
            a list of subject IDs.
        """
        return sorted(self.keys())

    def get_predictions(self, subject_ids: Optional[List[str]] = None):
        """
        Get a list of predictions for the specified subject IDs.

        Args:
            subject_ids: the list of subject IDs. If None, all subject IDs will be used.

        Returns:
            A list of predictions.
        """
        if subject_ids is None:
            subject_ids = self.keys()
        return sum((list(self[sid].values()) for sid in subject_ids), [])

    def associative_regularisation(self, regularisation: Config):
        """
        Calculate the associative regularization term.

        Args:
            regularisation: the regularization configuration.

        Returns:
            The associative regularization term.
        """
        preds = self.get_predictions()
        reg = []
        for p in preds:
            if p.associative_regularisation is not None:
                for k, term in p.associative_regularisation.items():
                    reg.append(term * regularisation.get(k))
        return sum(reg)

    def get_patient_trajectories(self, subject_ids: Optional[List[str]] = None):
        """
        Get the trajectories of patients in the predictions.

        Args:
            subject_ids: the list of subject IDs. If None, all subject IDs will be used.

        Returns:
            A dictionary mapping subject IDs to patient trajectories.
        """
        if subject_ids is None:
            subject_ids = sorted(self.keys())

        trajectories = {}
        for sid in subject_ids:
            adm_ids = sorted(self[sid].keys())
            trajectories[sid] = PatientTrajectory.concat(
                [self[sid][aid].trajectory for aid in adm_ids])
        return trajectories

    def average_interval_hours(self):
        """
        Calculate the average interval hours of the predictions.

        Returns:
            The average interval hours.
        """
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        return np.mean([a.interval_hours for a in adms])

    def filter_nans(self):
        """
        Filter out predictions with NaN values.

        Returns:
            the filtered predictions (a copy of the original predictions).
        """
        if len(self) == 0:
            return self

        cleaned = Predictions()
        nan_detected = False
        for sid, spreds in self.items():
            for aid, apred in spreds.items():
                if not apred.has_nans():
                    cleaned.add(sid, apred)
                else:
                    nan_detected = True
                    logging.warning('Skipping prediction with NaNs: '
                                    f'subject_id={sid}, admission_id={aid} '
                                    'interval_hours= '
                                    f'{apred.admission.interval_hours}. '
                                    'Note: long intervals is a likely '
                                    'reason to destabilise the model')
        if nan_detected:
            logging.warning(f'Average interval_hours: '
                            f'{cleaned.average_interval_hours()}')

        if len(cleaned) == 0 and len(self) > 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return cleaned

    @property
    def has_dx_predictions(self):
        """
        Check if the predictions have diagnosis predictions.

        Returns:
            True if all predictions have diagnosis predictions, False otherwise.
        """
        preds = self.get_predictions()
        return all([p.outcome is not None for p in preds])

    @property
    def has_auxiliary_loss(self):
        """
        Check if the predictions have auxiliary loss.

        Returns:
            True if all predictions have auxiliary loss, False otherwise.
        """
        preds = self.get_predictions()
        return all([p.auxiliary_loss is not None for p in preds])

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
