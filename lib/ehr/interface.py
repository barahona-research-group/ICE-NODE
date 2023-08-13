from __future__ import annotations
from typing import List, Optional, Dict, Union
import logging
import pickle
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from ..utils import tqdm_constructor, tree_hasnan, translate_path
from .dataset import MIMIC4ICUDataset
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, DemographicVectorConfig)
from .coding_scheme import CodesVector


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

    def has_nans(self):
        return tree_hasnan((self.outcome, self.observables, self.other))


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

    def average_interval_hours(self):
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        return np.mean([a.interval_hours for a in adms])

    def filter_nans(self):
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

        if len(cleaned) == 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return cleaned

    def prediction_dx_loss(self, dx_loss):
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        l_outcome = [a.outcome.vec for a in adms]
        l_pred = [p.outcome.vec for p in preds]
        l_mask = [jnp.ones_like(a.outcome.vec, dtype=bool) for a in adms]
        outcome = jnp.vstack(l_outcome)
        pred = jnp.vstack(l_pred)
        mask = jnp.vstack(l_mask)
        return dx_loss(outcome, pred, mask)

    def prediction_obs_loss(self, obs_loss):
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        obs_true = [o.value for a in adms for o in a.observables]
        obs_mask = [o.mask for a in adms for o in a.observables]
        obs_pred = [o.value for p in preds for o in p.observables]
        obs_true = jnp.vstack(obs_true)
        obs_mask = jnp.vstack(obs_mask)
        obs_pred = jnp.vstack(obs_pred)
        return obs_loss(obs_true, obs_pred, obs_mask)

    def outcome_first_occurrence_masks(self, subject_id):
        preds = self[subject_id]
        adms = [preds[aid].admission for aid in sorted(preds.keys())]
        first_occ_adm_id = outcome_first_occurrence(adms)
        return [first_occ_adm_id == a.admission_id for a in adms]


class Patients(eqx.Module):
    dataset: MIMIC4ICUDataset
    demographic_vector_config: DemographicVectorConfig
    subjects: Optional[Dict[int, Patient]]

    def __init__(self,
                 dataset: MIMIC4ICUDataset,
                 demographic_vector_config: DemographicVectorConfig,
                 subjects: Optional[Dict[int, Patient]] = None):
        super().__init__()
        self.dataset = dataset
        self.demographic_vector_config = demographic_vector_config
        self.subjects = subjects

    def random_splits(self,
                      splits: List[float],
                      random_seed: int = 42,
                      balanced: str = 'subjects'):
        return self.dataset.random_splits(splits, self.subjects.keys(),
                                          random_seed, balanced)

    def __len__(self):
        return len(self.subjects)

    def save(self, path: Union[str, Path], overwrite: bool = False):
        suffix = '.pickle'
        path = Path(path)
        if path.suffix != suffix:
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f'File {path} already exists.')
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: Union[str, Path]) -> jtu.pytree:
        suffix = '.pickle'
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'Not a file: {path}')
        if path.suffix != suffix:
            raise ValueError(f'Not a {suffix} file: {path}')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_subjects(self,
                      subject_ids: Optional[List[int]] = None,
                      num_workers: int = 1):
        if subject_ids is None:
            subject_ids = self.dataset.subject_ids
        subjects = self.dataset.to_subjects(
            subject_ids,
            num_workers=num_workers,
            demographic_vector_config=self.demographic_vector_config)

        subjects = {s.subject_id: s for s in subjects}
        return Patients(
            dataset=self.dataset,
            demographic_vector_config=self.demographic_vector_config,
            subjects=subjects)

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
        return Patients(
            dataset=self.dataset,
            demographic_vector_config=self.demographic_vector_config,
            subjects=subjects)

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

    def d2d_interval_days(self, subject_ids=None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.d2d_interval_days for s in subject_ids
                   for a in self.subjects[s].admissions)

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
            value = obs_scaler.unscale(obs.value)
            unscaled_obs.append(eqx.tree_at(lambda o: o.value, obs, value))
        return unscaled_obs

    def _unscaled_input(self, input_: InpatientInterventions):
        # (T, D)
        scaled_rate = input_.segmented_input
        input_scaler = self.dataset.preprocessing_history[0]['input']['scaler']
        unscaled_rate = [input_scaler.unscale(r) for r in scaled_rate]
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

    def outcome_first_occurrence(self, subject_id):
        return outcome_first_occurrence(self.subjects[subject_id].admissions)

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
