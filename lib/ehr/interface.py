from __future__ import annotations

import logging
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Dict, Union, Callable

import dask
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .coding_scheme import CodesVector
from .concepts import (Admission, Patient, InpatientObservables,
                       InpatientInterventions, DemographicVectorConfig,
                       LeadingObservableExtractorConfig)
from .dataset import Dataset, DatasetScheme, DatasetConfig
from ..base import Config, Data, Module
from ..utils import tqdm_constructor, tree_hasnan, write_config, load_config


def outcome_first_occurrence(sorted_admissions: List[Admission]):
    """
    Find the first occurrence admission index of each outcome in a list of sorted admissions.

    Args:
        sorted_admissions (List[Admission]): a list of sorted Admission objects.

    Returns:
        np.ndarray: an array containing the admission index of the first occurrence of each outcome for each admission. -1 means no occurrence.
    """
    first_occurrence = np.empty_like(
        sorted_admissions[0].outcome.vec,
        dtype=type[sorted_admissions[0].admission_id])
    first_occurrence[:] = -1
    for adm in sorted_admissions:
        update_mask = (first_occurrence < 0) & adm.outcome.vec
        first_occurrence[update_mask] = adm.admission_id
    return first_occurrence


class TrajectoryConfig(Config):
    sampling_rate: float = 0.5  # 0.5 means every 30 minutes.


class PatientTrajectory(InpatientObservables):
    """
    Represents a patient's trajectory of observations over time.

    Attributes:
        time (jnp.ndarray): array of timestamps for the observations.
        value (jnp.ndarray): array of observation values.
    """

    time: jnp.ndarray
    value: jnp.ndarray

    def __init__(self, time: jnp.ndarray, value: jnp.ndarray):
        if isinstance(time, jnp.ndarray):
            _np = jnp
        else:
            _np = np

        self.time = time
        self.value = value
        self.mask = _np.ones_like(value, dtype=bool)

    @staticmethod
    def concat(trajectories: List[PatientTrajectory]):
        if isinstance(trajectories[0], list):
            trajectories = [
                PatientTrajectory.concat(traj) for traj in trajectories
            ]
        return PatientTrajectory(
            time=jnp.hstack([traj.time for traj in trajectories]),
            value=jnp.vstack([traj.state for traj in trajectories]))


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


class InterfaceConfig(Config):
    """
    Configuration class for the interface.

    Attributes:
        demographic_vector (DemographicVectorConfig): configuration for the demographic vector.
        leading_observable (Optional[LeadingObservableExtractorConfig]): configuration for the leading observable (optional).
        scheme (Dict[str, str]): dictionary representing the scheme.
        cache (Optional[str]): cache path to store the interface after generation (optional).
        time_binning (Optional[int]): time binning configuration to aggregate observables/measurements over intervals (optional).
    """

    demographic_vector: DemographicVectorConfig
    leading_observable: Optional[LeadingObservableExtractorConfig]
    scheme: Dict[str, str]
    cache: Optional[str]
    time_binning: Optional[int] = None

    def __init__(self,
                 demographic_vector: DemographicVectorConfig,
                 leading_observable: Optional[LeadingObservableExtractorConfig] = None,
                 dataset_scheme: Optional[DatasetScheme] = None,
                 scheme: Optional[Dict[str, str]] = None,
                 cache: Optional[str] = None,
                 time_binning: Optional[int] = None,
                 **interface_scheme_kwargs):
        super().__init__()
        self.demographic_vector = demographic_vector
        self.leading_observable = leading_observable
        self.time_binning = time_binning
        if scheme is None:
            self.scheme = dataset_scheme.make_target_scheme_config(**scheme)
        else:
            self.scheme = scheme

        self.cache = cache


class Patients(Module):
    """
    A class representing a collection of patients in the EHR system, in ML-compliant format.

    Attributes:
        config (InterfaceConfig): the configuration for the interface.
        dataset (Dataset): the dataset containing the patient data.
        subjects (Dict[str, Patient]): a dictionary of patient objects, where the keys are subject IDs.
        _scheme (DatasetScheme): the target scheme for the dataset.

    Methods:
        subject_ids: returns a sorted list of subject IDs.
        scheme: returns the target scheme for the dataset.
        schemes: returns a tuple containing the dataset scheme and the target scheme.
        random_splits: generates random splits of the dataset.
        __len__: returns the number of patients in the dataset.
        equal_config: checks if the given configuration is equal to the cached configuration.
        save: saves the Patients object to disk.
        load: loads a Patients object from disk.
        try_load_cached: tries to load a cached Patients object, or creates a new one if the cache does not match the configuration.
        load_subjects: loads the subjects from the dataset.
        device_batch: loads the subjects and moves them to the device (e.g. GPU).
        epoch_splits: generates epoch splits for the subjects.
        batch_gen: generates batches of subjects.
        n_admissions: returns the total number of admissions for the given subjects.
        n_segments: returns the total number of segments for the given subjects.
        n_obs_times: returns the total number of observation timestamps for the given subjects.
        d2d_interval_days: returns the total number of days between the first and last discharge.
        interval_days: returns the total number of days between admissions for the given subjects.
        interval_hours: returns the total number of hours between admissions for the given subjects.
        p_obs: returns the proportion of present observations in the timestamped vectorized representation in the dataset.
        obs_coocurrence_matrix: returns the co-occurrence (or co-presence) matrix of the observables.
        size_in_bytes: returns the size of the Patients object in bytes.
        _unscaled_observation: unscales the observation values, undos the preprocessing scaling.
        _unscaled_leading_observable: unscales the leading observable values, undos the preprocessing scaling.
    """

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
        """Get the list of subject IDs."""
        return sorted(self.subjects.keys())

    @classmethod
    def external_argnames(cls):
        """Get the external argument names."""
        return ['dataset', 'subjects']

    @property
    def scheme(self):
        """Get the scheme."""
        return self._scheme

    @property
    def schemes(self):
        """Get the schemes."""
        return (self.dataset.scheme, self._scheme)

    def random_splits(self,
                      splits: List[float],
                      random_seed: int = 42,
                      balanced: str = 'subjects'):
        """Generate random splits of the dataset.

        Args:
            splits (List[float]): list of split ratios.
            random_seed (int, optional): random seed for reproducibility. Defaults to 42.
            balanced (str, optional): balancing strategy. Defaults to 'subjects'.

        Returns:
            Tuple[List[str]]: Tuple of lists containing the split subject IDs.
        """
        return self.dataset.random_splits(splits, self.subjects.keys(),
                                          random_seed, balanced)

    def __len__(self):
        """Get the number of subjects."""
        return len(self.subjects)

    @staticmethod
    def equal_config(path, config=None, dataset_config=None, subject_ids=None):
        """Check if the configuration is equal to the cached configuration.

        Args:
            path: path to the configuration file.
            config: interface configuration.
            dataset_config: Dataset configuration.
            subject_ids: list of subject IDs.

        Returns:
            bool: True if the configurations are equal, False otherwise.
        """
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
        """Save the Patients object to disk.

        Args:
            path (Union[str, Path]): path to save the Patients object.
            overwrite (bool, optional): whether to overwrite existing files. Defaults to False.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.save(path.with_suffix('.dataset'), overwrite)

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
    def load(path: Union[str, Path]) -> Patients:
        """Load the Patients object from disk.

        Args:
            path (Union[str, Path]): path to load the Patients object from.

        Returns:
            Patients: loaded Patients object.
        """
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
        """Try to load the Patients object from cache, or create a new one if cache is not available or does not match the current config.

        Args:
            config (InterfaceConfig): Interface configuration.
            dataset_config (DatasetConfig): Dataset configuration.
            dataset_generator (Callable[[DatasetConfig], Dataset]): Dataset generator function, used to apply further post-processing on the dataset before generating the Patients object.
            subject_subset (Optional[List[str]], optional): list of subject IDs to load. Defaults to None.
            num_workers (int, optional): number of workers for parallel processing. Defaults to 8.

        Returns:
            Patients: loaded or created Patients object.
        """
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
        """Load subjects from the dataset.

        Args:
            subject_ids (Optional[List[str]], optional): list of subject IDs to load. Defaults to None.
            num_workers (int, optional): number of workers for parallel processing. Defaults to 1.

        Returns:
            Patients: Patients object with loaded subjects.
        """
        if subject_ids is None:
            subject_ids = self.dataset.subject_ids

        subjects = self.dataset.to_subjects(
            subject_ids,
            num_workers=num_workers,
            demographic_vector_config=self.config.demographic_vector,
            leading_observable_config=self.config.leading_observable,
            target_scheme=self._scheme,
            time_binning=self.config.time_binning)

        subjects = {s.subject_id: s for s in subjects}

        interface = eqx.tree_at(lambda x: x.subjects, self, subjects)

        return interface

    def device_batch(self, subject_ids: Optional[List[str]] = None):
        """Load subjects and move them to the device. If subject_ids is None, load all subjects.

        Args:
            subject_ids (Optional[List[str]], optional): list of subject IDs to load. Defaults to None.

        Returns:
            Patients: Patients object with subjects loaded and moved to the device.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        subjects = {
            i: self.subjects[i].to_device()
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
        """Generate epoch splits for training.

        Args:
            subject_ids (Optional[List[str]]): list of subject IDs to split.
            batch_n_admissions (int): number of admissions per batch.
            ignore_first_admission (bool, optional): whether to ignore the first admission from the counts. Defaults to False.

        Returns:
            List[List[str]]: List of lists containing the split subject IDs.
        """
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
        """Generate batches of subjects.

        Args:
            subject_ids: list of subject IDs.
            batch_n_admissions (int): number of admissions per batch.
            ignore_first_admission (bool, optional): whether to ignore the first admission from the counts. Defaults to False.

        Yields:
            Patients: Patients object with a batch of subjects.
        """
        splits = self.epoch_splits(subject_ids, batch_n_admissions,
                                   ignore_first_admission)
        for split in splits:
            yield self.device_batch(split)

    def n_admissions(self,
                     subject_ids=None,
                     ignore_first_admission: bool = False):
        """Get the total number of admissions.

        Args:
            subject_ids: list of subject IDs.
            ignore_first_admission (bool, optional): Whether to ignore the first admission from the counts. Defaults to False.

        Returns:
            int: Total number of admissions.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        if ignore_first_admission:
            return sum(
                len(self.subjects[s].admissions) - 1 for s in subject_ids)
        return sum(len(self.subjects[s].admissions) for s in subject_ids)

    def n_segments(self, subject_ids=None):
        """Get the total number of segments.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of segments.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(
            len(a.interventions.time) for s in subject_ids
            for a in self.subjects[s].admissions)

    def n_obs_times(self, subject_ids=None):
        """Get the total number of observation times.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of observation times.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(
            len(o.time) for s in subject_ids
            for a in self.subjects[s].admissions for o in a.observables)

    def d2d_interval_days(self, subject_ids=None):
        """Get the total number of days between first discharge and last discharge.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of days between first discharge and last discharge.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(self.subjects[s].d2d_interval_days for s in subject_ids)

    def interval_days(self, subject_ids=None):
        """Get the total number of days in-hospital.

        Args:
            subject_ids: List of subject IDs.

        Returns:
            int: total number of days in-hospital.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_days for s in subject_ids
                   for a in self.subjects[s].admissions)

    def interval_hours(self, subject_ids=None):
        """Get the total number of hours in-hospital.

        Args:
            subject_ids: List of subject IDs.

        Returns:
            int: Total number of hours in-hospital.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_hours for s in subject_ids
                   for a in self.subjects[s].admissions)

    def p_obs(self, subject_ids=None):
        """Get the proportion of present observation over all unique timestamps for the given subject IDs.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            float: proportion of observables presence per unique timestamp.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        return sum(o.mask.sum() for s in subject_ids
                   for a in self.subjects[s].admissions
                   for o in a.observables) / self.n_obs_times() / len(
            self._scheme.obs)

    def obs_coocurrence_matrix(self):
        """Compute the co-occurrence (or co-presence) matrix of observables.

        Returns:
            jnp.ndarray: co-occurrence (or co-presence) matrix of observables.
        """
        obs = []
        for s in self.subjects.values():
            for a in s.admissions:
                for o in a.observables:
                    if len(o.time) > 0:
                        obs.append(o.mask)
        obs = jnp.vstack(obs, dtype=int)
        return obs.T @ obs

    def size_in_bytes(self):
        """Get the size of the Patients object in bytes.

        Returns:
            int: size of the Patients object in bytes.
        """
        is_arr = eqx.filter(self.subjects, eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects, is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def _unscaled_observation(self, obs_l: List[InpatientObservables]):
        """Unscale the observation values, undo the preprocessing scaling.

        Args:
            obs_l (List[InpatientObservables]): list of InpatientObservables.

        Returns:
            List[InpatientObservables]: list of unscaled InpatientObservables.
        """
        obs_scaler = self.dataset.scalers_history['obs']
        if not isinstance(obs_l, list):
            obs_l = [obs_l]
        unscaled_obs = []
        for obs in obs_l:
            value = obs_scaler.unscale(obs.value)
            unscaled_obs.append(eqx.tree_at(lambda o: o.value, obs, value))
        return unscaled_obs

    def _unscaled_leading_observable(self, lead_l: List[InpatientObservables]):
        """Unscale the leading observable values, undo the preprocessing scaling.

        Args:
            lead_l (List[InpatientObservables]): list of InpatientObservables.

        Returns:
            List[InpatientObservables]: ist of unscaled InpatientObservables.
        """
        lead_scaler = self.dataset.scalers_history['obs']
        obs_index = self.config.leading_observable.index
        if not isinstance(lead_l, list):
            lead_l = [lead_l]
        unscaled_lead = []
        for lead in lead_l:
            value = lead_scaler.unscale_code(lead.value, obs_index)
            unscaled_lead.append(eqx.tree_at(lambda o: o.value, lead, value))
        return unscaled_lead

    def _unscaled_input(self, input_: InpatientInterventions):
        """Unscale the input values, undo the preprocessing scaling.

        Args:
            input_ (InpatientInterventions): input rates to unscale.

        Returns:
            InpatientInterventions: unscaled input rates.
        """
        # (T, D)
        scaled_rate = input_.segmented_input
        input_scaler = self.dataset.scalers_history['int_input']
        unscaled_rate = [input_scaler.unscale(r) for r in scaled_rate]
        return eqx.tree_at(lambda o: o.segmented_input, input_, unscaled_rate)

    def _unscaled_admission(self, inpatient_admission: Admission):
        """Unscale the admission observations, leading observables, and interventions.

        Args:
            inpatient_admission (Admission): the admission object to unscale its relevant components.

        Returns:
            Admission: unscaled admission object (from a copy of the original).
        """
        adm = eqx.tree_at(
            lambda o: o.observables, inpatient_admission,
            self._unscaled_observation(inpatient_admission.observables))
        adm = eqx.tree_at(
            lambda o: o.interventions, adm,
            self._unscaled_input(inpatient_admission.interventions))
        adm = eqx.tree_at(
            lambda o: o.leading_observable, adm,
            self._unscaled_leading_observable(
                inpatient_admission.leading_observable))
        return adm

    def unscaled_subject(self, subject_id: str):
        """Apply the unscaling to all admissions in a subject.

        Args:
            subject_id (str): subject ID.

        Returns:
            Patient: patient object with the unscaled relevant components (from a copy of the original).
        """
        s = self.subjects[subject_id]
        adms = s.admissions
        adms = [self._unscaled_admission(a) for a in adms]
        return eqx.tree_at(lambda o: o.admissions, s, adms)

    def subject_size_in_bytes(self, subject_id):
        """Get the size of the subject object in bytes.

        Args:
            subject_id (str): subject ID.

        Returns:
            int: size of the subject object in bytes.
        """
        is_arr = eqx.filter(self.subjects[subject_id], eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects[subject_id], is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def outcome_frequency_vec(self, subjects: List[str]):
        """Get the outcome frequency vector for a list of subjects.

        Args:
            subjects (List[str]): list of subject IDs.

        Returns:
            jnp.ndarray: outcome frequency vector.
        """
        return sum(self.subjects[i].outcome_frequency_vec() for i in subjects)

    def outcome_frequency_partitions(self, n_partitions, subjects: List[str]):
        """
        Get the outcome codes partitioned by their frequency of occurrence into `n_partitions` partitions. The codes in each partition contributes to 1 / n_partitions of the all outcome occurrences.
        
        Args:
            n_partitions (int): number of partitions.
            subjects (List[str]): list of subject IDs.

        Returns:
            List[List[int]]: list of outcome codes partitioned by frequency into `n_partitions` partitions.
        
        """
        frequency_vec = self.outcome_frequency_vec(subjects)
        frequency_vec = frequency_vec / frequency_vec.sum()
        sorted_codes = np.argsort(frequency_vec)
        frequency_vec = frequency_vec[sorted_codes]
        cumsum = np.cumsum(frequency_vec)
        partitions = np.linspace(0, 1, n_partitions + 1)[1:-1]
        splitters = np.searchsorted(cumsum, partitions)
        return np.hsplit(sorted_codes, splitters)

    def outcome_first_occurrence(self, subject_id):
        """Get the first occurrence admission index of each outcome for a subject. If an outcome does not occur, the index is set to -1.

        Args:
            subject_id (str): subject ID.

        Returns:
            int: first occurrence admission index of each outcome for a subject.
        """
        return outcome_first_occurrence(self.subjects[subject_id].admissions)

    def outcome_first_occurrence_masks(self, subject_id):
        """Get a list of masks indicating whether an outcome occurs for a subject for the first time.

        Args:
            subject_id (str): subject ID.

        Returns:
            List[bool]: list of masks indicating whether an outcome occurs for a subject for the first time.

        """
        adms = self.subjects[subject_id].admissions
        first_occ_adm_id = outcome_first_occurrence(adms)
        return [first_occ_adm_id == a.admission_id for a in adms]

    def outcome_all_masks(self, subject_id):
        """Get a list of full-masks with the same shape as the outcome vector."""
        adms = self.subjects[subject_id].admissions
        if isinstance(adms[0].outcome.vec, jnp.ndarray):
            _np = jnp
        else:
            _np = np
        return [_np.ones_like(a.outcome.vec, dtype=bool) for a in adms]
