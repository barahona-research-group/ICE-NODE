from __future__ import annotations

import logging
import pickle
from abc import ABCMeta, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Dict, Union, Callable, Tuple

import dask
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd

from .dataset import Dataset, DatasetScheme, DatasetConfig, DatasetSchemeConfig, ReportAttributes, \
    AbstractDatasetTransformation, AbstractDatasetPipeline
from .tvx_concepts import (Admission, Patient, InpatientObservables,
                           InpatientInterventions, DemographicVectorConfig,
                           LeadingObservableExtractorConfig, SegmentedPatient)
from ..base import Config, Module
from ..utils import tqdm_constructor, write_config, load_config


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


class TVxEHRSplitsConfig(Config):
    split_quantiles: List[float]
    seed: int = 0
    balance: str = 'subjects'
    discount_first_admission: bool = False


class TVxEHRSampleConfig(Config):
    n_subjects: int
    seed: int = 0
    offset: int = 0


class IQROutlierRemoverConfig(Config):
    outlier_q1: float = 0.25
    outlier_q2: float = 0.75
    outlier_iqr_scale: float = 1.5
    outlier_z1: float = -2.5
    outlier_z2: float = 2.5


class ScalerConfig(Config):
    use_float16: bool = True


class OutlierRemoversConfig(Config):
    obs: Optional[IQROutlierRemoverConfig] = None


class OutlierRemovers(eqx.Module):
    obs: Optional[eqx.Module] = None


class ScalersConfig(Config):
    obs: Optional[ScalerConfig] = None
    icu_inputs: Optional[ScalerConfig] = None


class Scalers(eqx.Module):
    obs: Optional[eqx.Module] = None
    icu_inputs: Optional[eqx.Module] = None


class DatasetNumericalProcessorsConfig(Config):
    scalers: Optional[ScalersConfig] = ScalersConfig()
    outlier_removers: Optional[OutlierRemoversConfig] = OutlierRemoversConfig()


class DatasetNumericalProcessors(eqx.Module):
    outlier_removers: OutlierRemovers = OutlierRemovers()
    scalers: Scalers = Scalers()


class TVxEHRConfig(Config):
    """
    Configuration class for the interface.

    Attributes:
        demographic_vector (DemographicVectorConfig): configuration for the demographic vector.
        leading_observable (Optional[LeadingObservableExtractorConfig]): configuration for the leading observable (optional).
        scheme (Dict[str, str]): dictionary representing the scheme.
        time_binning (Optional[int]): time binning configuration to aggregate observables/measurements over intervals (optional).
    """
    scheme: DatasetSchemeConfig
    demographic_vector: DemographicVectorConfig
    splits: Optional[TVxEHRSplitsConfig] = None
    numerical_processors: DatasetNumericalProcessorsConfig = DatasetNumericalProcessorsConfig()
    interventions: bool = False
    observables: bool = False
    interventions_segmentation: bool = False
    time_binning: Optional[float] = None


class TVxReportAttributes(ReportAttributes):
    tvx_concept: str = None


class AbstractTVxTransformation(AbstractDatasetTransformation, metaclass=ABCMeta):
    name: str = None

    @abstractmethod
    def __call__(self, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        pass

    @staticmethod
    def static_report(report: Tuple[TVxReportAttributes, ...], **report_attributes) -> Tuple[TVxReportAttributes, ...]:
        return report + (TVxReportAttributes(**report_attributes),)

    def report(self, report: Tuple[TVxReportAttributes, ...], **kwargs) -> Tuple[TVxReportAttributes, ...]:
        return self.static_report(report, **self.meta, **kwargs)


class TrainableTransformation(AbstractTVxTransformation, metaclass=ABCMeta):

    # dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (RandomSplits, SetIndex, SetCodeIntegerIndices)

    @staticmethod
    def get_admission_ids(tv_ehr: TVxEHR) -> List[str]:
        c_subject_id = tv_ehr.dataset.config.tables.static.subject_id_alias
        c_admission_id = tv_ehr.dataset.config.tables.admissions.admission_id_alias
        admissions = tv_ehr.dataset.tables.admissions[[c_subject_id]]
        assert c_admission_id in admissions.index.names, f"Column {c_admission_id} not found in admissions table index."
        training_subject_ids = tv_ehr.splits[0]
        return admissions[admissions[c_subject_id].isin(training_subject_ids)].index.unique().tolist()

    @abstractmethod
    def __call__(self, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        pass


class AbstractTVxPipeline(AbstractDatasetPipeline, metaclass=ABCMeta):
    transformations: List[AbstractTVxTransformation]

    @abstractmethod
    def __call__(self, tv_ehr: TVxEHR) -> Tuple[TVxEHR, pd.DataFrame]:
        pass


class TVxEHR(Module):
    """
    A class representing a collection of patients in the EHR system, in ML-compliant format.

    Attributes:
        config (TVxEHRConfig): the configuration for the interface.
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

    config: TVxEHRConfig
    dataset: Dataset
    dataset_numerical_processors: DatasetNumericalProcessors = DatasetNumericalProcessors()
    subjects: Optional[Dict[str, Patient]] = None
    splits: Optional[Tuple[Tuple[str]]] = None

    @property
    def subject_ids(self):
        """Get the list of subject IDs."""
        return sorted(self.subjects.keys())

    @classmethod
    def external_argnames(cls):
        """Get the external argument names."""
        return ['dataset', 'subjects']

    @cached_property
    def scheme(self):
        """Get the scheme."""
        return self.dataset.scheme.make_target_scheme(self.config.scheme)

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
    def equal_config(path,
                     config=None,
                     dataset_config=None,
                     subject_ids=None):
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
    def load(path: Union[str, Path]) -> TVxEHR:
        """Load the Patients object from disk.

        Args:
            path (Union[str, Path]): path to load the Patients object from.

        Returns:
            TVxEHR: loaded Patients object.
        """
        path = Path(path)
        with open(path.with_suffix('.subjects.pickle'), 'rb') as file:
            subjects = pickle.load(file)
        dataset = Dataset.load(path.with_suffix('.dataset'))
        config = load_config(path.with_suffix('.config.json'))
        config = Config.from_dict(config)
        return TVxEHR.import_module(config,
                                    dataset=dataset,
                                    subjects=subjects)

    @staticmethod
    def try_load_cached(config: TVxEHRConfig,
                        dataset_config: DatasetConfig,
                        dataset_generator: Callable[[DatasetConfig], Dataset],
                        subject_subset: Optional[List[str]] = None,
                        num_workers: int = 8):
        """Try to load the Patients object from cache, or create a new one if cache is not available or does not match the current config.

        Args:
            config (TVxEHRConfig): Interface configuration.
            dataset_config (DatasetConfig): Dataset configuration.
            dataset_generator (Callable[[DatasetConfig], Dataset]): Dataset generator function, used to apply further post-processing on the dataset before generating the Patients object.
            subject_subset (Optional[List[str]], optional): list of subject IDs to load. Defaults to None.
            num_workers (int, optional): number of workers for parallel processing. Defaults to 8.

        Returns:
            TVxEHR: loaded or created Patients object.
        """
        if config.cache is None or not TVxEHR.equal_config(
                config.cache, config, dataset_config, subject_subset):
            if config.cache is not None:
                logging.info('Cache does not match config, ignoring cache.')
            logging.info('Loading subjects from scratch.')

            with dask.config.set(scheduler='processes',
                                 num_workers=num_workers):
                interface = TVxEHR(config, dataset_generator(dataset_config))
                interface = interface.load_subjects(num_workers=num_workers,
                                                    subject_ids=subject_subset)

            if config.cache is not None:
                interface.save(config.cache, overwrite=True)

            return interface
        else:
            logging.info('Loading cached subjects.')
            return TVxEHR.load(config.cache)

    def load_subjects(self,
                      subject_ids: Optional[List[str]] = None,
                      num_workers: int = 1):
        """Load subjects from the dataset.

        Args:
            subject_ids (Optional[List[str]], optional): list of subject IDs to load. Defaults to None.
            num_workers (int, optional): number of workers for parallel processing. Defaults to 1.

        Returns:
            TVxEHR: Patients object with loaded subjects.
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
            TVxEHR: Patients object with subjects loaded and moved to the device.
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


class SegmentedTVxEHR(TVxEHR):
    subjects: Dict[str, SegmentedPatient]

## TODO:
# [ ] Four modes of temporal EHR access:
#   - [ ] Discharge timestamps access.
#   - [ ] Interval-based (segmented) access.
#   - [ ] Timestamp-based access.
#   - [ ] Time-binning.


## TODO:
# Roadmap:
# [ ] Add support for FHIR resources.
# [ ] Add support for medication and prescriptions.
# [ ] Add support for referrals and locations.
# [ ] Add examples folder.
# [ ] Add support for process-mining models.
