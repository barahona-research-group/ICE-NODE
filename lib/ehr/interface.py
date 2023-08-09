from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ..utils import tqdm_constructor
from .dataset import MIMIC4ICUDataset
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, CodesVector)


def outcome_first_occurrence(sorted_admissions: List[Admission]):
    first_occurrence = np.empty_like(sorted_admissions[0].outcome.vec,
                                     dtype=int)
    first_occurrence[:] = -1
    for adm in sorted_admissions:
        update_mask = (first_occurrence < 0) & adm.outcome.vec
        first_occurrence[update_mask] = adm.admission_id
    return first_occurrence


class AdmissionPrediction(eqx.Module):
    admission: Admission
    outcome: Optional[CodesVector] = None
    observables: Optional[List[InpatientObservables]] = None
    other: Optional[Dict[str, jnp.ndarray]] = None


class Predictions(dict):

    def add(self, subject_id: int, prediction: AdmissionPrediction):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][prediction.admission.admission_id] = prediction

    @property
    def subject_ids(self):
        return sorted(self.keys())

    def get_predictions(self, subject_ids: Optional[List[int]] = None):
        if subject_ids is None:
            subject_ids = self.keys()
        return sum((list(self[sid].values()) for sid in subject_ids), [])

    def prediction_loss(self, dx_loss=None, obs_loss=None):
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        loss = {}
        if dx_loss is not None:
            dx_true = jnp.vstack([a.outcome.vec for a in adms])
            dx_pred = jnp.vstack([p.outcome.vec for p in preds])
            dx_mask = jnp.ones_like(dx_true, dtype=bool)
            loss['dx_loss'] = dx_loss(dx_true, dx_pred, dx_mask)
        if obs_loss is not None:
            obs_true = sum((a.observables for a in adms), [])
            obs_true = InpatientObservables.concat(obs_true)
            obs_pred = sum((p.observables for p in preds), [])
            obs_pred = InpatientObservables.concat(obs_pred)

            loss['obs_loss'] = obs_loss(obs_true.value, obs_pred.value,
                                        obs_true.mask)

        return loss

    def outcome_first_occurrence_masks(self, subject_id):
        preds = self[subject_id]
        adms = [preds[aid].admission for aid in sorted(preds.keys())]
        first_occ_adm_id = outcome_first_occurrence(adms)
        return [first_occ_adm_id == a.admission_id for a in adms]


class Patients(eqx.Module):
    dataset: MIMIC4ICUDataset
    subjects: Optional[Dict[int, Patient]]

    def __init__(self,
                 dataset: MIMIC4ICUDataset,
                 subjects: Optional[Dict[int, Patient]] = None):
        super().__init__()
        self.dataset = dataset
        self.subjects = subjects

    def random_splits(self,
                      splits: List[float],
                      random_seed: int = 42,
                      balanced: str = 'subjects'):
        return self.dataset.random_splits(splits, self.subjects.keys(),
                                          random_seed, balanced)

    def __len__(self):
        return len(self.subjects)

    def load_subjects(self,
                      subject_ids: Optional[List[int]] = None,
                      num_workers: int = 1,
                      **demographic_flags):
        if subject_ids is None:
            subject_ids = self.dataset.subject_ids
        subjects = self.dataset.to_subjects(subject_ids, num_workers)

        subjects = {s.subject_id: s for s in subjects}
        subjects = {
            sid: s.set_demographic_vector_attributes(**demographic_flags)
            for sid, s in subjects.items()
        }
        return Patients(dataset=self.dataset, subjects=subjects)

    @property
    def demographic_vector_size(self):
        return next(iter(self.subjects.values())).demographic_vector_size

    def _subject_to_device(self, subject_id: int):
        s = self.subjects[subject_id]
        arrs, others = eqx.partition(s, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        return eqx.combine(arrs, others)

    def device_batch(self, subject_ids: Optional[List[int]] = None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        subjects = {
            i: jax.block_until_ready(self._subject_to_device(i))
            for i in tqdm_constructor(subject_ids,
                                      desc="Loading to device",
                                      unit='subject',
                                      leave=False)
        }
        return Patients(dataset=self.dataset, subjects=subjects)

    def batch_gen(self,
                  subject_ids,
                  batch_n_admissions: int,
                  ignore_first_admission: bool = False):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        n_splits = self.n_admissions(
            subject_ids, ignore_first_admission) // batch_n_admissions
        if n_splits == 0:
            n_splits = 1
        p_splits = np.linspace(0, 1, n_splits + 1)[1:-1]

        subject_ids = np.array(subject_ids)

        c_subject_id = self.dataset.colname['adm']['subject_id']
        adm_df = self.dataset.df['adm']
        adm_df = adm_df[adm_df[c_subject_id].isin(subject_ids)]

        n_adms = adm_df.groupby(c_subject_id).size()
        if ignore_first_admission:
            n_adms = n_adms - 1
        w_adms = n_adms.loc[subject_ids] / n_adms.sum()
        weights = w_adms.values.cumsum()
        splits = np.searchsorted(weights, p_splits)
        splits = [a.tolist() for a in np.split(subject_ids, splits)]
        splits = [s for s in splits if len(s) > 0]

        for split in splits:
            yield self.device_batch(split)

    def n_admissions(self,
                     subject_ids=None,
                     ignore_first_admission: bool = False):
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        if ignore_first_admission:
            return sum(
                len(self.subjects[s].admissions) - 1 for s in subject_ids)
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

    def interval_days(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_days for s in subject_ids
                   for a in self.subjects[s].admissions)

    def interval_hours(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_hours for s in subject_ids
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

    def _unscaled_admission(self, inpatient_admission: Admission):
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

    def outcome_first_occurrence_masks(self, subject_id):
        adms = self.subjects[subject_id].admissions
        first_occ_adm_id = outcome_first_occurrence(adms)
        return [first_occ_adm_id == a.admission_id for a in adms]

    def outcome_all_masks(self, subject_id):
        adms = self.subjects[subject_id].admissions
        if isinstance(adms[0].outcome.vec, jnp.ndarray):
            _np = jnp
        else:
            _np = np
        return [_np.ones_like(a.outcome.vec, dtype=bool) for a in adms]


if __name__ == '__main__':
    pass
    # from .dataset import load_dataset
    # logging.root.level = logging.DEBUG
    # m4inpatient_dataset = load_dataset('M4ICU')
    # inpatients = Inpatients(m4inpatient_dataset)
