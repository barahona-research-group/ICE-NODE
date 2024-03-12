from __future__ import annotations

import logging
import pickle
from abc import ABCMeta, abstractmethod
from dataclasses import field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Dict, Union, Callable, Tuple, Type, ClassVar

import dask
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd

from . import OutcomeExtractor
from .coding_scheme import FileBasedOutcomeExtractor, CodesVector
from .dataset import Dataset, DatasetScheme, DatasetConfig, DatasetSchemeConfig, ReportAttributes, \
    AbstractTransformation, AbstractDatasetPipeline, AbstractDatasetRepresentation, Report
from .tvx_concepts import (Admission, Patient, InpatientObservables,
                           InpatientInterventions, DemographicVectorConfig,
                           LeadingObservableExtractorConfig, SegmentedPatient, StaticInfo, InpatientInput)
from ..base import Config, Data, Module, FlatConfig
from ..utils import tqdm_constructor, write_config, load_config


class CodedValueProcessorConfig(Config):
    pass


class CodedValueScalerConfig(CodedValueProcessorConfig):
    use_float16: bool


class TVxEHRSplitsConfig(Config):
    split_quantiles: List[float]
    seed: int = 0
    balance: str = 'subjects'
    discount_first_admission: bool = False


class TVxEHRSampleConfig(FlatConfig):
    n_subjects: int
    seed: int = 0
    offset: int = 0


class IQROutlierRemoverConfig(FlatConfig):
    outlier_q1: float = 0.25
    outlier_q2: float = 0.75
    outlier_iqr_scale: float = 1.5
    outlier_z1: float = -2.5
    outlier_z2: float = 2.5


class ScalerConfig(Config):
    use_float16: bool = True


class OutlierRemoversConfig(Config):
    obs: Optional[IQROutlierRemoverConfig] = None


class ScalersConfig(Config):
    obs: Optional[ScalerConfig] = None
    icu_inputs: Optional[ScalerConfig] = None


class DatasetNumericalProcessorsConfig(Config):
    scalers: Optional[ScalersConfig] = ScalersConfig()
    outlier_removers: Optional[OutlierRemoversConfig] = OutlierRemoversConfig()


class CodedValueProcessor(Module):
    config: CodedValueProcessorConfig
    table_name: Optional[str] = None
    code_column: Optional[str] = None
    value_column: Optional[str] = None

    def fit(self, dataset: Dataset, admission_ids: List[str],
            table_name: str, code_column: str, value_column: str) -> 'CodedValueProcessor':
        df = getattr(dataset.tables, table_name)
        c_adm_id = getattr(dataset.config.tables, table_name).admission_id_alias
        df = df[[code_column, value_column, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]

        fitted = self
        for k, v in self._extract_stats(df, code_column, value_column).items():
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v)

        for k, v in {'table_name': table_name, 'code_column': code_column, 'value_column': value_column}.items():
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v)
        return fitted

    @abstractmethod
    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        pass

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass

    @property
    def series_dict(self) -> Dict[str, pd.Series]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pd.Series)}

    @property
    def processing_target(self) -> Dict[str, str]:
        return {'table_name': self.table_name, 'code_column': self.code_column, 'value_column': self.value_column}

    def save_series(self, path: Path, key: str):
        for k, v in self.series_dict.items():
            v.to_hdf(path, f'{key}/{k}', format='table')

    @staticmethod
    def load_series(path: Path, key: str):
        with pd.HDFStore(path, mode='r') as store:
            return {k.split('/')[-1]: store[k] for k in store.keys() if k.startswith(key)}

    def save_config(self, path: Path, key: str):
        config = self.config.to_dict()
        config['classname'] = self.__class__.__name__
        pd.DataFrame(config, index=[0]).to_hdf(path, f'{key}/config', format='table')
        pd.DataFrame(self.processing_target, index=[0]).to_hdf(path, f'{key}/target', format='table')

    @staticmethod
    def load_config(path: Path, key: str) -> Tuple[CodedValueProcessorConfig, str, Dict[str, str]]:
        with pd.HDFStore(path, mode='r') as store:
            config_data = store[f"{key}/config"].loc[0].to_dict()
            classname = config_data.pop('classname')
            target = store[f"{key}/target"].loc[0].to_dict()
            return CodedValueProcessorConfig.from_dict(config_data), classname, target

    def save(self, path: Path, key: str):
        self.save_series(path, f'{key}/series')
        self.save_config(path, f'{key}/config')

    @staticmethod
    def load(path: Path, key: str) -> CodedValueProcessor:
        config, classname, target = CodedValueProcessor.load_config(path, f'{key}/config')
        series = CodedValueProcessor.load_series(path, f'{key}/series')
        return Module.import_module(config=config, classname=classname, **series, **target)

    def equals(self, other: CodedValueProcessor):
        return self.config == other.config and self.processing_target == other.processing_target and \
            all(getattr(self, k).equals(getattr(other, k)) for k in self.series_dict.keys())

    def __eq__(self, other):
        return self.equals(other)


class CodedValueScaler(CodedValueProcessor):
    config: CodedValueScalerConfig

    @property
    @abstractmethod
    def original_dtype(self) -> np.dtype:
        pass

    @abstractmethod
    def unscale(self, array: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        pass


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


class OutlierRemovers(eqx.Module):
    obs: Optional[CodedValueProcessor] = None


class Scalers(eqx.Module):
    obs: Optional[CodedValueScaler] = None
    icu_inputs: Optional[CodedValueScaler] = None


class DatasetNumericalProcessors(eqx.Module):
    outlier_removers: OutlierRemovers = OutlierRemovers()
    scalers: Scalers = Scalers()

    def save(self, path: Path, key: str):
        if self.outlier_removers.obs is not None:
            self.outlier_removers.obs.save(path, f'{key}/outlier_removers/obs')

        if self.scalers.obs is not None:
            self.scalers.obs.save(path, f'{key}/scalers/obs')

        if self.scalers.icu_inputs is not None:
            self.scalers.icu_inputs.save(path, f'{key}/scalers/icu_inputs')

    @staticmethod
    def load(path: Path, key: str) -> DatasetNumericalProcessors:
        scalers = {}
        outlier_removers = {}
        with pd.HDFStore(path, mode='r') as store:
            if f'{key}/outlier_removers' in store:
                if f'{key}/outlier_removers/obs' in store:
                    outlier_removers['obs'] = CodedValueProcessor.load(path, f'{key}/outlier_removers')
            if f'{key}/scalers' in store:
                if f'{key}/scalers/obs' in store:
                    scalers['obs'] = CodedValueScaler.load(path, f'{key}/scalers/obs')
                if f'{key}/scalers/icu_inputs' in store:
                    scalers['icu_inputs'] = CodedValueScaler.load(path, f'{key}/scalers/icu_inputs')
        return DatasetNumericalProcessors(outlier_removers=OutlierRemovers(**outlier_removers),
                                          scalers=Scalers(**scalers))


class TVxEHRSchemeConfig(DatasetSchemeConfig):
    outcome: Optional[str] = None


class TVxEHRScheme(DatasetScheme):
    config: TVxEHRSchemeConfig

    @cached_property
    def outcome(self) -> Optional[OutcomeExtractor]:
        return OutcomeExtractor.from_name(self.config.outcome) if self.config.outcome else None

    @staticmethod
    def validate_outcome_scheme(dx_discharge: str, outcome: str) -> bool:
        return outcome in FileBasedOutcomeExtractor.supported_outcomes(dx_discharge)

    @staticmethod
    def validate_mapping(source: DatasetScheme, target: TVxEHRScheme):
        target_schemes = target.scheme_dict
        for key, source_scheme in source.scheme_dict.items():
            assert source_scheme.mapper_to(
                target_schemes[key].name
            ), f"Cannot map {key} from {source_scheme.name} to {target_schemes[key].name}"
        if target.outcome is not None:
            assert TVxEHRScheme.validate_outcome_scheme(
                target.dx_discharge.name, target.outcome.name
            ), f"Outcome {target.outcome.name} not supported for {target.dx_discharge.name}"

    def demographic_vector_size(self, demographic_vector_config: DemographicVectorConfig):
        size = 0
        if demographic_vector_config.gender:
            size += len(self.gender)
        if demographic_vector_config.age:
            size += 1
        if demographic_vector_config.ethnicity:
            size += len(self.ethnicity)
        return size

    def dx_mapper(self, source_scheme: DatasetScheme):
        return source_scheme.dx_discharge.mapper_to(self.dx_discharge.name)

    def ethnicity_mapper(self, source_scheme: DatasetScheme):
        return source_scheme.ethnicity.mapper_to(self.ethnicity.name)

    def gender_mapper(self, source_scheme: DatasetScheme):
        return source_scheme.gender.mapper_to(self.gender.name)

    def icu_procedures_mapper(self, source_scheme: DatasetScheme):
        return source_scheme.icu_procedures.mapper_to(self.icu_procedures.name)

    def hosp_procedures_mapper(self, source_scheme: DatasetScheme):
        return source_scheme.hosp_procedures.mapper_to(self.hosp_procedures.name)

    @staticmethod
    def supported_target_scheme_options(dataset_scheme: DatasetScheme):
        supported_attr_targets = {
            k: (v.name,) + v.supported_targets
            for k, v in dataset_scheme.scheme_dict.items()
        }
        supported_outcomes = FileBasedOutcomeExtractor.supported_outcomes(dataset_scheme.dx_discharge.name)
        supported_attr_targets['outcome'] = supported_outcomes
        return supported_attr_targets


class TVxEHRConfig(Config):
    """
    Configuration class for the interface.

    Attributes:
        demographic (DemographicVectorConfig): configuration for the demographic vector.
        leading_observable (Optional[LeadingObservableExtractorConfig]): configuration for the leading observable (optional).
        scheme (Dict[str, str]): dictionary representing the scheme.
        time_binning (Optional[int]): time binning configuration to aggregate observables/measurements over intervals (optional).
    """
    scheme: TVxEHRSchemeConfig
    demographic: DemographicVectorConfig
    sample: Optional[TVxEHRSampleConfig] = None
    splits: Optional[TVxEHRSplitsConfig] = None
    numerical_processors: DatasetNumericalProcessorsConfig = DatasetNumericalProcessorsConfig()
    interventions: bool = False
    observables: bool = False
    time_binning: Optional[float] = None
    leading_observable: Optional[LeadingObservableExtractorConfig] = None
    interventions_segmentation: bool = False


class TVxReportAttributes(ReportAttributes):
    tvx_concept: str = None

    @staticmethod
    def _t(object_or_type: Union[Type, Data]) -> str:
        return object_or_type.__name__ if isinstance(object_or_type, type) else object_or_type.__class__.__name__

    @classmethod
    def ehr_prefix(cls) -> str:
        return cls._t(TVxEHR)

    @classmethod
    def subjects_prefix(cls) -> str:
        return f'{cls.ehr_prefix()}.Dict[str, {cls._t(Patient)}]'

    @classmethod
    def subject_prefix(cls) -> str:
        return f'{cls.subjects_prefix()}[i]'

    @classmethod
    def static_info_prefix(cls) -> str:
        return f'{cls.subject_prefix()}.{cls._t(StaticInfo)}'

    @classmethod
    def admissions_prefix(cls) -> str:
        return f'{cls.subject_prefix()}.List[{cls._t(Admission)}]'

    @classmethod
    def inpatient_input_prefix(cls, attribute) -> str:
        return f'{cls.admissions_prefix()}[j].{cls._t(InpatientInterventions)}.{attribute}({cls._t(InpatientInput)})'

    @classmethod
    def admission_attribute_prefix(cls, attribute: str, attribute_type: str | type) -> str:
        if isinstance(attribute_type, type):
            attribute_type = cls._t(attribute_type)

        return f'{cls.admissions_prefix()}[j].{attribute}({attribute_type})'

    @classmethod
    def admission_codes_attribute(cls, attribute) -> str:
        return f'{cls.admission_attribute_prefix(attribute, CodesVector)}'


class TVxReport(Report):
    incident_class: ClassVar[Type[TVxReportAttributes]] = TVxReportAttributes


class AbstractTVxTransformation(AbstractTransformation, metaclass=ABCMeta):
    @classmethod
    def skip(cls, dataset: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        return dataset, report.add(transformation=cls, operation='skip')


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


class AbstractTVxPipeline(AbstractDatasetPipeline, metaclass=ABCMeta):
    transformations: List[AbstractTVxTransformation] = field(kw_only=True)
    report_class: ClassVar[Type[TVxReport]] = TVxReport


class TVxEHR(AbstractDatasetRepresentation):
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

    config: TVxEHRConfig = field(kw_only=True)
    dataset: Dataset = field(kw_only=True)
    numerical_processors: DatasetNumericalProcessors = DatasetNumericalProcessors()
    splits: Optional[Tuple[Tuple[str]]] = None
    subjects: Optional[Dict[str, Patient]] = None

    @property
    def header(self) -> Tuple[Tuple[Tuple[str]], DatasetNumericalProcessors]:
        return self.splits, self.numerical_processors

    @staticmethod
    def compile_header(splits: Optional[Tuple[Tuple[str]]],
                       numerical_processors: DatasetNumericalProcessors) -> Tuple[Tuple[Tuple[str]], DatasetNumericalProcessors]:
        return splits, numerical_processors

    #
    # def save_header(self, path: Path, key: str):
    #     pd.DataFrame()


    @property
    def subject_ids(self):
        """Get the list of subject IDs."""
        return sorted(self.subjects.keys())

    def equals(self, other: 'TVxEHR'):
        return self.config == other.config and self.dataset.equals(
            other.dataset) and self.subjects == other.subjects and \
            self.equal_report(
                other) and self.splits == other.splits and self.numerical_processors == other.numerical_processors

    @cached_property
    def scheme(self):
        """Get the scheme."""
        scheme = TVxEHRScheme(config=self.config.scheme)
        TVxEHRScheme.validate_mapping(self.dataset.scheme, scheme)
        return scheme

    @cached_property
    def gender_mapper(self):
        return self.scheme.gender_mapper(self.dataset.scheme)

    @cached_property
    def ethnicity_mapper(self):
        return self.scheme.ethnicity_mapper(self.dataset.scheme)

    @cached_property
    def dx_mapper(self):
        return self.scheme.dx_mapper(self.dataset.scheme)

    @cached_property
    def icu_procedures_mapper(self):
        return self.scheme.icu_procedures_mapper(self.dataset.scheme)

    @cached_property
    def hosp_procedures_mapper(self):
        return self.scheme.hosp_procedures_mapper(self.dataset.scheme)

    @cached_property
    def subjects_sorted_admission_ids(self) -> Dict[str, List[str]]:
        c_admittime = self.dataset.config.tables.admissions.admission_time_alias
        c_subject_id = self.dataset.config.tables.admissions.subject_id_alias

        # For each subject get the list of adm sorted by admission date.
        return self.dataset.tables.admissions.groupby(c_subject_id).apply(
            lambda x: x.sort_values(c_admittime).index.to_list()).to_dict()

    @cached_property
    def admission_ids(self) -> List[str]:
        return sum(self.subjects_sorted_admission_ids.values(), [])

    @cached_property
    def admission_dates(self) -> Dict[str, Tuple[date, date]]:
        admissions = self.dataset.tables.admissions
        c_admittime = self.dataset.config.tables.admissions.admission_time_alias
        c_dischtime = self.dataset.config.tables.admissions.discharge_time_alias
        return dict(zip(admissions.index, zip(admissions[c_admittime], admissions[c_dischtime])))

    def __len__(self):
        """Get the number of subjects."""
        return len(self.subjects) if self.subjects is not None else 0

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
            demographic_vector_config=self.config.demographic,
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
# [x] Add examples folder.
# [ ] Add support for process-mining models.
