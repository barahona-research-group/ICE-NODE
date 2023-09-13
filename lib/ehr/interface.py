from __future__ import annotations
from typing import List, Optional, Dict, Union, Any, Callable
import logging
import pickle
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import dask
from ..utils import tqdm_constructor, tree_hasnan, write_config, load_config
from ..base import Config, Data, Module
from .dataset import Dataset, DatasetScheme, DatasetConfig
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, DemographicVectorConfig,
                       LeadingObservableConfig)
from .coding_scheme import CodesVector


def outcome_first_occurrence(sorted_admissions: List[Admission]):
    first_occurrence = np.empty_like(
        sorted_admissions[0].outcome.vec,
        dtype=type[sorted_admissions[0].admission_id])
    first_occurrence[:] = -1
    for adm in sorted_admissions:
        update_mask = (first_occurrence < 0) & adm.outcome.vec
        first_occurrence[update_mask] = adm.admission_id
    return first_occurrence


class PatientTrajectory(Data):
    time: jnp.ndarray
    state: jnp.ndarray

    @staticmethod
    def concat(trajectories: List[PatientTrajectory]):
        if isinstance(trajectories[0], list):
            trajectories = [
                PatientTrajectory.concat(traj) for traj in trajectories
            ]
        return PatientTrajectory(
            time=jnp.hstack([traj.time for traj in trajectories]),
            state=jnp.vstack([traj.state for traj in trajectories]))


class AdmissionPrediction(Data):
    admission: Admission
    outcome: Optional[CodesVector] = None
    observables: Optional[List[InpatientObservables]] = None
    leading_observable: Optional[List[InpatientObservables]] = None
    trajectory: Optional[PatientTrajectory] = None
    associative_regularisation: Optional[Dict[str, jnp.ndarray]] = None
    other: Optional[Dict[str, jnp.ndarray]] = None

    def has_nans(self):
        return tree_hasnan((self.outcome, self.observables, self.other,
                            self.leading_observable))


class Predictions(dict):

    def add(self, subject_id: str, prediction: AdmissionPrediction):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][prediction.admission.admission_id] = prediction

    @property
    def subject_ids(self):
        return sorted(self.keys())

    def get_predictions(self, subject_ids: Optional[List[str]] = None):
        if subject_ids is None:
            subject_ids = self.keys()
        return sum((list(self[sid].values()) for sid in subject_ids), [])

    def associative_regularisation(self, regularisation: Config):
        preds = self.get_predictions()
        reg = []
        for p in preds:
            if p.associative_regularisation is not None:
                for k, term in p.associative_regularisation.items():
                    reg.append(term * regularisation.get(k))
        return sum(reg)

    def get_patient_trajectories(self,
                                 subject_ids: Optional[List[str]] = None):
        if subject_ids is None:
            subject_ids = sorted(self.keys())

        trajectories = {}
        for sid in subject_ids:
            adm_ids = sorted(self[sid].keys())
            trajectories[sid] = PatientTrajectory.concat(
                [self[sid][aid].trajectory for aid in adm_ids])
        return trajectories

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

        if len(cleaned) == 0 and len(self) > 0:
            logging.warning('No predictions left after NaN filtering')
            raise ValueError('No predictions left after NaN filtering')

        return cleaned

    def prediction_dx_loss(self, dx_loss):
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        l_outcome = [a.outcome.vec for a in adms]
        l_pred = [p.outcome.vec for p in preds]
        l_mask = [jnp.ones_like(a.outcome.vec, dtype=bool) for a in adms]
        loss_v = jnp.array(list(map(dx_loss, l_outcome, l_pred, l_mask)))
        loss_v = jnp.nanmean(loss_v)
        return jnp.where(jnp.isnan(loss_v), 0., loss_v)

    def _numeric_loss(self, loss_fn, attr: str):
        preds = self.get_predictions()
        adms = [r.admission for r in preds]
        l_true = [o.value for a in adms for o in getattr(a, attr)]
        l_mask = [o.mask for a in adms for o in getattr(a, attr)]
        l_pred = [o.value for p in preds for o in getattr(p, attr)]
        true = jnp.vstack(l_true)
        mask = jnp.vstack(l_mask)
        pred = jnp.vstack(l_pred)
        loss_v = loss_fn(true, pred, mask)
        return jnp.where(jnp.isnan(loss_v), 0., loss_v)

    def prediction_obs_loss(self, obs_loss):
        return self._numeric_loss(obs_loss, 'observables')

    def prediction_lead_loss(self, lead_loss):
        preds = self.get_predictions()
        loss_v = []
        for pred in preds:
            adm = pred.admission
            for pred_lo, adm_lo in zip(pred.leading_observable,
                                       adm.leading_observable):
                for i in range(len(adm_lo.time)):
                    m = adm_lo.mask[i]
                    if m.sum() < len(m) // 2:
                        continue
                    y = adm_lo.value[i][m]
                    y_hat = pred_lo.value[i][m]
                    loss_v.append(lead_loss(y, y_hat) / m.sum())
        loss_v = jnp.array(loss_v)
        loss_v = jnp.nanmean(loss_v)
        return jnp.where(jnp.isnan(loss_v), 0., loss_v)

    def prediction_lead_data(self, obs_index):
        preds = self.get_predictions()

        y = []
        y_hat = []
        mask = []

        obs = []
        obs_mask = []
        for pred in preds:
            adm = pred.admission
            for pred_lo, adm_lo, adm_obs in zip(pred.leading_observable,
                                                adm.leading_observable,
                                                adm.observables):
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


class InterfaceConfig(Config):
    demographic_vector: DemographicVectorConfig
    leading_observable: Optional[LeadingObservableConfig]
    scheme: Dict[str, str]
    cache: Optional[str]

    def __init__(self,
                 demographic_vector: DemographicVectorConfig,
                 leading_observable: Optional[LeadingObservableConfig] = None,
                 dataset_scheme: Optional[DatasetScheme] = None,
                 scheme: Optional[Dict[str, str]] = None,
                 cache: Optional[str] = None,
                 **interface_scheme_kwargs):
        super().__init__()
        self.demographic_vector = demographic_vector
        self.leading_observable = leading_observable
        if scheme is None:
            self.scheme = dataset_scheme.make_target_scheme_config(**scheme)
        else:
            self.scheme = scheme

        self.cache = cache


class Patients(Module):
    config: InterfaceConfig
    dataset: Dataset
    subjects: Dict[str, Patient]
    _scheme: DatasetScheme

    def __init__(self,
                 config: InterfaceConfig,
                 dataset: Dataset,
                 subjects: Dict[str, Patient] = {}):
        super().__init__(config=config)
        self.dataset = dataset
        self.subjects = subjects
        self._scheme = dataset.scheme.make_target_scheme(config.scheme)

    @property
    def subject_ids(self):
        return sorted(self.subjects.keys())

    @classmethod
    def external_argnames(cls):
        return ['dataset', 'subjects']

    @property
    def scheme(self):
        return self._scheme

    @property
    def schemes(self):
        return (self.dataset.scheme, self._scheme)

    def random_splits(self,
                      splits: List[float],
                      random_seed: int = 42,
                      balanced: str = 'subjects'):
        return self.dataset.random_splits(splits, self.subjects.keys(),
                                          random_seed, balanced)

    def __len__(self):
        return len(self.subjects)

    @staticmethod
    def equal_config(path, config=None, dataset_config=None, subject_ids=None):
        path = Path(path)
        try:
            cached_dsconfig = load_config(
                path.with_suffix('.dataset.config.json'))
            cached_config = load_config(path.with_suffix('.config.json'))
            cached_ids = load_config(path.with_suffix('.subject_ids.json'))
        except FileNotFoundError:
            return False
        pairs = []
        for a, b in [(config, cached_config),
                     (dataset_config, cached_dsconfig),
                     (subject_ids, cached_ids)]:
            if a is None:
                continue
            if issubclass(type(a), Config):
                a = a.to_dict()
            pairs.append((a, b))

        for a, b in pairs:
            if a != b:
                logging.debug('Config mismatch:')
                logging.debug(f'a:  {a}')
                logging.debug(f'b:  {b}')
                return False

        return all(a == b for a, b in pairs)

    def save(self, path: Union[str, Path], overwrite: bool = False):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.save(path.with_suffix('.dataset'), overwrite)

        # rest = eqx.tree_at(lambda x: x.dataset, self, None)
        subj_path = path.with_suffix('.subjects.pickle')
        if subj_path.exists():
            if overwrite:
                subj_path.unlink()
            else:
                raise RuntimeError(f'File {subj_path} already exists.')
        with open(subj_path, 'wb') as file:
            pickle.dump(self.subjects, file)

        write_config(self.dataset.config.to_dict(),
                     path.with_suffix('.dataset.config.json'))
        write_config(self.config.to_dict(), path.with_suffix('.config.json'))
        write_config(self.subject_ids, path.with_suffix('.subject_ids.json'))

    @staticmethod
    def load(path: Union[str, Path]) -> jtu.pytree:
        path = Path(path)
        with open(path.with_suffix('.subjects.pickle'), 'rb') as file:
            subjects = pickle.load(file)
        dataset = Dataset.load(path.with_suffix('.dataset'))
        config = load_config(path.with_suffix('.config.json'))
        config = Config.from_dict(config)
        return Patients.import_module(config,
                                      dataset=dataset,
                                      subjects=subjects)

    @staticmethod
    def try_load_cached(config: InterfaceConfig,
                        dataset_config: DatasetConfig,
                        dataset_generator: Callable[[DatasetConfig], Dataset],
                        subject_subset: Optional[List[str]] = None,
                        num_workers: int = 8):
        if config.cache is None or not Patients.equal_config(
                config.cache, config, dataset_config, subject_subset):
            if config.cache is not None:
                logging.info('Cache does not match config, ignoring cache.')
            logging.info('Loading subjects from scratch.')

            with dask.config.set(scheduler='processes',
                                 num_workers=num_workers):
                interface = Patients(config, dataset_generator(dataset_config))
                interface = interface.load_subjects(num_workers=num_workers,
                                                    subject_ids=subject_subset)

            if config.cache is not None:
                interface.save(config.cache, overwrite=True)

            return interface
        else:
            logging.info('Loading cached subjects.')
            return Patients.load(config.cache)

    def load_subjects(self,
                      subject_ids: Optional[List[str]] = None,
                      num_workers: int = 1):
        if subject_ids is None:
            subject_ids = self.dataset.subject_ids

        subjects = self.dataset.to_subjects(
            subject_ids,
            num_workers=num_workers,
            demographic_vector_config=self.config.demographic_vector,
            leading_observable_config=self.config.leading_observable,
            target_scheme=self._scheme)

        subjects = {s.subject_id: s for s in subjects}
        interface = eqx.tree_at(lambda x: x.subjects, self, subjects)

        return interface

    def _subject_to_device(self, subject_id: str):
        s = self.subjects[subject_id]
        arrs, others = eqx.partition(s, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        return eqx.combine(arrs, others)

    def device_batch(self, subject_ids: Optional[List[str]] = None):
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        subjects = {
            i: self._subject_to_device(i)
            for i in tqdm_constructor(subject_ids,
                                      desc="Loading to device",
                                      unit='subject',
                                      leave=False)
        }
        return eqx.tree_at(lambda x: x.subjects, self, subjects)

    def epoch_splits(self,
                     subject_ids: Optional[List[str]],
                     batch_n_admissions: int,
                     ignore_first_admission: bool = False):
        if subject_ids is None:
            subject_ids = list(self.subjects.keys())

        n_splits = self.n_admissions(
            subject_ids, ignore_first_admission) // batch_n_admissions
        if n_splits == 0:
            n_splits = 1
        p_splits = np.linspace(0, 1, n_splits + 1)[1:-1]

        subject_ids = np.array(subject_ids,
                               dtype=type(list(self.subjects.keys())[0]))

        c_subject_id = self.dataset.colname['adm'].subject_id
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
        return splits

    def batch_gen(self,
                  subject_ids,
                  batch_n_admissions: int,
                  ignore_first_admission: bool = False):
        splits = self.epoch_splits(subject_ids, batch_n_admissions,
                                   ignore_first_admission)
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

        return sum(self.subjects[s].d2d_interval_days for s in subject_ids)

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
                       self._scheme.obs)

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

    def unscaled_subject(self, subject_id: str):
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

    def outcome_frequency_vec(self, subjects: List[str]):
        return sum(self.subjects[i].outcome_frequency_vec() for i in subjects)

    def outcome_frequency_partitions(self, n_partitions, subjects: List[str]):
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
