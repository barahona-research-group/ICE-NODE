from __future__ import annotations
from typing import List, Optional, Dict
import logging
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from .dataset import MIMIC4ICUDataset
from .inpatient_concepts import (InpatientAdmission, Inpatient,
                                 InpatientObservables, InpatientInterventions,
                                 CodesVector)


class AdmissionPrediction(eqx.Module):
    outcome: CodesVector
    observables: List[InpatientObservables]


class AdmissionResults(eqx.Module):
    admission: InpatientAdmission
    prediction: AdmissionPrediction
    other: Dict[str, jnp.ndarray] = None


class InpatientsPredictions(dict):

    def add(self,
            subject_id: int,
            admission: InpatientAdmission,
            prediction: AdmissionPrediction,
            other: Optional[Dict[str, jnp.ndarray]] = None):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][admission.admission_id] = AdmissionResults(
            admission=admission, prediction=prediction, other=other)

    def get_subjects(self):
        return sorted(self.keys())

    def get_predictions(self, subject_id):
        predictions = self[subject_id]
        return list(map(predictions.get, sorted(predictions)))

    def subject_prediction_loss(self, subject_id, outcome_loss, obs_loss):
        (outcome_true, outcome_pred, obs_true, obs_pred,
         obs_mask) = [], [], [], [], []
        for r in self[subject_id].values():
            outcome_true.append(r.admission.outcome.vec)
            outcome_pred.append(r.prediction.outcome_vec)
            obs_true.append(r.admission.observables.value)
            obs_pred.append(r.prediction.observables.value)
            obs_mask.append(r.admission.observables.mask)

        outcome_true = jnp.vstack(outcome_true)
        outcome_pred = jnp.vstack(outcome_pred)
        obs_true = jnp.vstack(obs_true)
        obs_pred = jnp.vstack(obs_pred)
        obs_mask = jnp.vstack(obs_mask)

        return outcome_loss(outcome_true, outcome_pred) + obs_loss(
            obs_true, obs_pred, obs_mask)

    def prediction_loss(self, outcome_loss, obs_loss):
        loss = [
            self.subject_prediction_loss(subject_id, outcome_loss, obs_loss)
            for subject_id in self.keys()
        ]
        return jnp.nanmean(jnp.array(loss))


class Inpatients(eqx.Module):
    dataset: MIMIC4ICUDataset
    subjects: Dict[int, Inpatient]

    def __init__(self,
                 dataset: MIMIC4ICUDataset,
                 subject_ids: List[int] = None,
                 num_workers=1):
        super().__init__()
        self.dataset = dataset
        logging.debug('Loading subjects..')
        if subject_ids is None:
            subject_ids = dataset.subject_ids
        subjects_list = dataset.to_subjects(subject_ids, num_workers)
        logging.debug('[DONE] Loading subjects')
        self.subjects = {s.subject_id: s for s in subjects_list}

    def n_admissions(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        return sum(len(self.subjects[s].admissions) for s in subject_ids)

    def n_segments(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(
            len(a.interventions.time) for s in subject_ids
            for a in self.subjects[s].admissions)

    def n_obs_times(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(
            len(o.time) for s in subject_ids
            for a in self.subjects[s].admissions for o in a.observables)

    def intervals_sum(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        delta = lambda t: t[-1][1] - t[0][0]

        return sum(
            delta(a.interventions.time) for s in subject_ids
            for a in self.subjects[s].admissions)

    def p_obs(self, subject_ids=None):
        if subject_ids is None:

            subject_ids = self.subjects.keys()
        return sum(o.mask.sum() for s in subject_ids
                   for a in self.subjects[s].admissions
                   for o in a.observables) / self.n_obs_times() / len(
                       self.dataset.scheme.obs)

    def obs_coocurrence_matrix(self):
        obs = []
        for s in self.subjects.values():
            for a in s.admissions:
                for o in a.observables:
                    if len(o.time) > 0:
                        obs.append(o.mask)
        obs = jnp.vstack(obs, dtype=int)
        return obs.T @ obs

    def size_in_bytes(self):
        is_arr = eqx.filter(self.subjects, eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects, is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def _unscaled_observation(self, obs_l: List[InpatientObservables]):
        obs_scaler = self.dataset.preprocessing_history[0]['obs']['scaler']
        unscaled_obs = []
        for obs in obs_l:
            value = obs.value
            obs_idx = np.arange(value.shape[1])
            mu = obs_scaler.mean.loc[obs_idx].values
            sigma = obs_scaler.std.loc[obs_idx].values
            unscaled_obs.append(
                eqx.tree_at(lambda o: o.value, obs, value * sigma + mu))
        return unscaled_obs

    def _unscaled_input(self, input_: InpatientInterventions):
        # (T, D)
        scaled_rate = input_.segmented_input
        input_idx = np.arange(scaled_rate[0].shape[0])
        input_scaler = self.dataset.preprocessing_history[0]['input']['scaler']
        mx = input_scaler.max_val.loc[input_idx].values
        unscaled_rate = [rate * mx for rate in scaled_rate]
        return eqx.tree_at(lambda o: o.segmented_input, input_, unscaled_rate)

    def _unscaled_admission(self, inpatient_admission: InpatientAdmission):
        return eqx.tree_at(
            lambda o: o.observables, inpatient_admission,
            self._unscaled_observation(inpatient_admission.observables))

    def unscaled_subject(self, subject_id: int):
        s = self.subjects[subject_id]
        adms = s.admissions
        adms = [self._unscaled_admission(a) for a in adms]
        return eqx.tree_at(lambda o: o.admissions, s, adms)

    def subject_size_in_bytes(self, subject_id):
        is_arr = eqx.filter(self.subjects[subject_id], eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects[subject_id], is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def to_jax_arrays(self, subject_ids: Optional[List[int]] = None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        subjects = {i: self.subjects[i] for i in subject_ids}
        arrs, others = eqx.partition(subjects, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        subjects = eqx.combine(arrs, others)
        return eqx.tree_at(lambda o: o.subjects, self, subjects)

    def outcome_frequency_vec(self, subjects: List[int]):
        return sum(self.subjects[i].outcome_frequency_vec() for i in subjects)

    def outcome_frequency_partitions(self, n_partitions, subjects: List[int]):
        frequency_vec = self.outcome_frequency_vec(subjects)
        frequency_vec = frequency_vec / frequency_vec.sum()
        sorted_codes = np.argsort(frequency_vec)
        frequency_vec = frequency_vec[sorted_codes]
        cumsum = np.cumsum(frequency_vec)
        partitions = np.linspace(0, 1, n_partitions + 1)[1:-1]
        splitters = np.searchsorted(cumsum, partitions)
        return np.hsplit(sorted_codes, splitters)


if __name__ == '__main__':
    pass
    # from .dataset import load_dataset
    # logging.root.level = logging.DEBUG
    # m4inpatient_dataset = load_dataset('M4ICU')
    # inpatients = Inpatients(m4inpatient_dataset)
