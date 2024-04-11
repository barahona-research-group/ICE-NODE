from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple, Type, ClassVar, Iterable, Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
import tables as tbl

from . import OutcomeExtractor
from .coding_scheme import CodesVector
from .dataset import Dataset, DatasetScheme, DatasetSchemeConfig, ReportAttributes, \
    AbstractTransformation, AbstractDatasetPipeline, AbstractDatasetRepresentation, Report
from .tvx_concepts import (Admission, Patient, InpatientObservables,
                           InpatientInterventions, DemographicVectorConfig,
                           LeadingObservableExtractorConfig, SegmentedPatient, StaticInfo, InpatientInput,
                           AdmissionDates)
from ..base import Config, VxData, Module, FlatConfig
from ..utils import tqdm_constructor


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
    scalers: Optional[ScalersConfig] = field(default_factory=ScalersConfig)
    outlier_removers: Optional[OutlierRemoversConfig] = field(default_factory=OutlierRemoversConfig)


class CodedValueProcessor(Module):
    config: CodedValueProcessorConfig
    table_name: Optional[str] = None
    code_column: Optional[str] = None
    value_column: Optional[str] = None

    def table_getter(self, dataset: Dataset) -> pd.DataFrame:
        return getattr(dataset.tables, self.table_name)

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
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v,
                                 is_leaf=lambda x: x is None)
        return fitted

    @abstractmethod
    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        pass

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass

    @property
    def series_dict(self) -> Dict[str, pd.Series]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, pd.Series) and len(v) > 0}

    @property
    def processing_target(self) -> Dict[str, str]:
        return {'table_name': self.table_name, 'code_column': self.code_column, 'value_column': self.value_column}

    def save_series(self, store: tbl.Group):
        h5file = store._v_file
        for k, v in self.series_dict.items():
            v.to_hdf(h5file.filename, h5file.create_group(store, k)._v_pathname, format='table')

    @staticmethod
    def load_series(store: tbl.Group):
        h5filepath = store._v_file.filename
        return {k: pd.read_hdf(h5filepath, key=getattr(store, k)._v_pathname) for k in store._v_groups}

    def save_config(self, store: tbl.Group):
        config = self.config.to_dict()
        h5file = store._v_file

        config['classname'] = self.__class__.__name__
        pd.DataFrame(config, index=[0]).to_hdf(h5file.filename, h5file.create_group(store, 'config')._v_pathname,
                                               format='table')
        pd.DataFrame(self.processing_target, index=[0]).to_hdf(h5file.filename,
                                                               h5file.create_group(store, 'target')._v_pathname,
                                                               format='table')

    @staticmethod
    def load_config(store: tbl.Group) -> Tuple[CodedValueProcessorConfig, str, Dict[str, str]]:
        config_data = pd.read_hdf(store._v_file.filename,
                                  key=getattr(store, "config")._v_pathname).loc[0].to_dict()
        classname = config_data.pop('classname')
        target = pd.read_hdf(store._v_file.filename,
                             key=getattr(store, "target")._v_pathname).loc[0].to_dict()
        return CodedValueProcessorConfig.from_dict(config_data), classname, target

    def save(self, store: tbl.Group):
        h5file = store._v_file
        self.save_series(h5file.create_group(store, 'series'))
        self.save_config(h5file.create_group(store, 'config'))

    @staticmethod
    def load(store: tbl.Group) -> CodedValueProcessor:
        config, classname, target = CodedValueProcessor.load_config(getattr(store, 'config'))
        series = CodedValueProcessor.load_series(getattr(store, 'series'))
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

    def equals(self, other: OutlierRemovers) -> bool:
        return self.obs is None and other.obs is None or self.obs.equals(other.obs)


class Scalers(eqx.Module):
    obs: Optional[CodedValueScaler] = None
    icu_inputs: Optional[CodedValueScaler] = None

    def equals(self, other: Scalers) -> bool:
        return self.obs is None and other.obs is None or self.obs.equals(other.obs) and \
            self.icu_inputs is None and other.icu_inputs is None or self.icu_inputs.equals(other.icu_inputs)


class DatasetNumericalProcessors(eqx.Module):
    outlier_removers: OutlierRemovers = OutlierRemovers()
    scalers: Scalers = Scalers()

    def save(self, store: tbl.Group):
        h5file = store._v_file
        outliers = h5file.create_group(store, 'outlier_removers')
        scalers = h5file.create_group(store, 'scalers')
        if self.outlier_removers.obs is not None:
            self.outlier_removers.obs.save(h5file.create_group(outliers, 'obs'))

        for k in ['obs', 'icu_inputs']:
            if getattr(self.scalers, k) is not None:
                getattr(self.scalers, k).save(h5file.create_group(scalers, k))

    @staticmethod
    def load(store: tbl.Group) -> DatasetNumericalProcessors:
        outlier_removers = {k: CodedValueProcessor.load(v) for k, v in
                            store.outlier_removers._v_groups.items()}
        scalers = {k: CodedValueScaler.load(v) for k, v in getattr(store, 'scalers')._v_groups.items()}
        return DatasetNumericalProcessors(outlier_removers=OutlierRemovers(**outlier_removers),
                                          scalers=Scalers(**scalers))

    def equals(self, other: DatasetNumericalProcessors) -> bool:
        return self.outlier_removers.equals(other.outlier_removers) and self.scalers.equals(other.scalers)

    def __eq__(self, other):
        return self.equals(other)


class TVxEHRSchemeConfig(DatasetSchemeConfig):
    outcome: Optional[str] = None


class TVxEHRScheme(DatasetScheme):
    config: TVxEHRSchemeConfig

    @cached_property
    def outcome(self) -> Optional[OutcomeExtractor]:
        return self.context_view.outcome[self.config.outcome] if self.config.outcome else None

    @staticmethod
    def validate_outcome_scheme(outcome_scheme: OutcomeExtractor, dx_discharge: str) -> bool:
        context = outcome_scheme.context_view
        return context.supported_outcome(outcome_scheme.name, dx_discharge)

    @staticmethod
    def validate_mapping(source: DatasetScheme, target: TVxEHRScheme):
        target_schemes = target.scheme_dict
        for key, source_scheme in source.scheme_dict.items():
            assert source_scheme.mapper_to(
                target_schemes[key].name
            ), f"Cannot map {key} from {source_scheme.name} to {target_schemes[key].name}"

        if target.outcome is not None:
            assert TVxEHRScheme.validate_outcome_scheme(
                target.outcome, target.dx_discharge.name
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
    def check_target_scheme_support(dataset_scheme: DatasetScheme) -> Dict[str, Tuple[str, ...]]:
        supported_attr_targets = {
            k: v.supported_targets
            for k, v in dataset_scheme.scheme_dict.items()
        }
        supported_outcomes = dataset_scheme.context_view.supported_outcomes(dataset_scheme.dx_discharge.name)
        return supported_attr_targets | {'outcome': supported_outcomes}


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
    numerical_processors: DatasetNumericalProcessorsConfig = field(default_factory=DatasetNumericalProcessorsConfig)
    interventions: bool = False
    observables: bool = False
    time_binning: Optional[float] = None
    leading_observable: Optional[LeadingObservableExtractorConfig] = None
    interventions_segmentation: bool = False
    admission_minimum_los: Optional[float] = None


class TVxReportAttributes(ReportAttributes):
    tvx_concept: str = None

    @staticmethod
    def _t(object_or_type: Union[Type, VxData]) -> str:
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


_SplitsType = Tuple[Tuple[str, ...], ...]


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
    numerical_processors: DatasetNumericalProcessors = field(default_factory=DatasetNumericalProcessors)
    splits: Optional[_SplitsType] = None
    subjects: Optional[Dict[str, Patient]] = None
    patient_class: ClassVar[Type[Patient]] = Patient
    report_class: ClassVar[Type[TVxReport]] = TVxReport

    @property
    def header(self) -> Dict[str, Any]:
        return super().header | {'splits': self.splits, 'numerical_processors': self.numerical_processors,
                                 'subjects_list': self.subject_ids}

    def equal_header(self, other: AbstractDatasetRepresentation) -> bool:
        h1, h2 = self.header, other.header
        return super().equal_header(other) and h1['splits'] == h2['splits'] and \
            h1['numerical_processors'].equals(h2['numerical_processors']) and h1['subjects_list'] == h2['subjects_list']

    @property
    def subject_ids(self):
        """Get the list of subject IDs."""
        return sorted(self.subjects.keys()) if self.subjects is not None else []

    def equals(self, other: 'TVxEHR'):
        return self.equal_header(other) and self.dataset.equals(other.dataset) and self.subjects == other.subjects

    @cached_property
    def scheme(self):
        """Get the scheme."""
        scheme = TVxEHRScheme(config=self.config.scheme, context_view=self.dataset.scheme_manager.view())
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

        admission_ids = sum(self.subjects_sorted_admission_ids.values(), [])
        # Check unique admission IDs across all subjects.
        if len(admission_ids) != len(set(admission_ids)):
            raise ValueError("Duplicate admission IDs found.")

        return admission_ids

    def subject_admission_demographics(self, subject_id: str) -> Dict[str, jnp.ndarray]:
        return self.subjects[subject_id].admission_demographics(self.config.demographic)

    @cached_property
    def admission_demographics(self) -> Dict[str, jnp.ndarray]:
        return {admission_id: admission_demo for subject_id in self.subjects for admission_id, admission_demo in
                self.subject_admission_demographics(subject_id).items()}

    @cached_property
    def admission_dates(self) -> Dict[str, AdmissionDates]:
        admissions = self.dataset.tables.admissions
        c_admittime = self.dataset.config.tables.admissions.admission_time_alias
        c_dischtime = self.dataset.config.tables.admissions.discharge_time_alias
        return admissions.apply(lambda x: AdmissionDates(x[c_admittime], x[c_dischtime]), axis=1).to_dict()

    def __len__(self):
        """Get the number of subjects."""
        return len(self.subjects) if self.subjects is not None else 0

    def save_splits(self, group: tbl.Group):
        h5file = group._v_file
        for i, split in enumerate(self.splits):
            pd.Series(split).to_hdf(h5file.filename, h5file.create_group(group, str(i))._v_pathname, format='table')
            # h5file.create_array(group, str(i), np.array(split))

    @staticmethod
    def load_splits(group: tbl.Group) -> _SplitsType:
        h5file = group._v_file
        return tuple(tuple(pd.read_hdf(h5file.filename, group._f_get_child(str(i))._v_pathname)) for i in
                     range(group._v_nchildren))

    def save_subjects(self, group: tbl.Group):
        if self.subjects is not None:
            h5file = group._v_file
            for subject_id, subject in self.subjects.items():
                subject.to_hdf_group(h5file.create_group(group, subject_id))

    @classmethod
    def load_subjects(cls, group: tbl.Group) -> Optional[Dict[str, Patient]]:
        patients = {k: cls.patient_class.from_hdf_group(group._f_get_child(k)) for k in group._v_groups}
        return None if len(patients) == 0 else patients

    def save(self, store: Union[str, Path, tbl.Group], overwrite: bool = False,
             complib: Literal['blosc', 'zlib', 'lzo', 'bzip2'] = 'blosc', complevel: int = 9):
        """Save the Patients object to disk.

        Args:
            store (Union[str, Path]): path to save the Patients object.
            overwrite (bool, optional): whether to overwrite existing files. Defaults to False.
        """
        if not isinstance(store, tbl.Group):
            self.save_config(store, key='tvx_ehr', overwrite=overwrite)
            filters = tbl.Filters(complib=complib, complevel=complevel)
            with tbl.open_file(str(Path(store).with_suffix('.h5')), mode='w', filters=filters,
                               max_numexpr_threads=None, max_blosc_threads=None) as store:
                return self.save(store.root, overwrite=overwrite)
        else:
            self.save_config(store._v_file.filename, key='tvx_ehr', overwrite=True)
        h5file = store._v_file
        self.dataset.save(h5file.create_group(store, 'dataset'), overwrite=overwrite)
        if self.splits is not None:
            self.save_splits(h5file.create_group(store, 'splits'))
        self.numerical_processors.save(h5file.create_group(store, 'numerical_processors'))
        self.save_subjects(h5file.create_group(store, 'tvx'))
        if len(self.pipeline_report) > 0:
            report_key = h5file.create_group(store, 'report')._v_pathname
            self.pipeline_report.to_hdf(h5file.filename, key=report_key, format='table')

    @classmethod
    def load(cls, store: Union[str, Path, tbl.Group]) -> TVxEHR:
        """Load the Patients object from disk.

        Args:
            path (Union[str, Path]): path to load the Patients object from.
            dataset_path (Optional[Union[str, Path]], optional): dataset path to load the dataset from.
            Defaults to None. if None, the dataset representation will be loaded from a path generated by
            the function dataset_path_prefix (same directory with different .h5 file name).
        Returns:
            TVxEHR: loaded Patients object.
        """
        if not isinstance(store, tbl.Group):
            with tbl.open_file(str(Path(store).with_suffix('.h5')), 'r') as store:
                return cls.load(store.root)
        h5file = store._v_file

        dataset = Dataset.load(store.dataset)
        config, classname = cls.load_config(Path(h5file.filename), key='tvx_ehr')
        tvx_class = cls.module_class(classname)
        numerical_processors = DatasetNumericalProcessors.load(store.numerical_processors)
        if 'report' in store:
            pipeline_report = pd.read_hdf(h5file.filename, key=store.report._v_pathname, mode='r')
        else:
            pipeline_report = pd.DataFrame()
        return TVxEHR.import_module(config=config,
                                    classname=classname,
                                    splits=cls.load_splits(store.splits) if hasattr(store, 'splits') else None,
                                    numerical_processors=numerical_processors,
                                    pipeline_report=pipeline_report,
                                    dataset=dataset,
                                    subjects=tvx_class.load_subjects(store.tvx))

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
                     discount_first_admission: bool = False):
        """Generate epoch splits for training.

        Args:
            subject_ids (Optional[List[str]]): list of subject IDs to split.
            batch_n_admissions (int): number of admissions per batch.
            discount_first_admission (bool, optional): whether to ignore the first admission from the counts. Defaults to False.

        Returns:
            List[List[str]]: List of lists containing the split subject IDs.
        """
        if subject_ids is None:
            subject_ids = list(self.subjects.keys())

        n_splits = self.n_admissions(
            subject_ids, discount_first_admission) // batch_n_admissions
        if n_splits == 0:
            n_splits = 1
        p_splits = np.linspace(0, 1, n_splits + 1)[1:-1]

        subject_ids = np.array(subject_ids,
                               dtype=type(list(self.subjects.keys())[0]))

        n_adms = self.dataset.subjects_n_admissions
        if discount_first_admission:
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

    def iter_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        """Iterate over the observables for the given subject IDs.

        Args:
            subject_ids: list of subject IDs.

        Yields:
            InpatientObservables: InpatientObservables object.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                yield adm.observables

    def iter_lead_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        """Iterate over the leading observables for the given subject IDs.

        Args:
            subject_ids: list of subject IDs.

        Yields:
            InpatientObservables: InpatientObservables object.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                yield adm.leading_observable

    def n_obs_times(self, subject_ids=None):
        """Get the total number of observation times.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            int: total number of observation times.
        """
        return sum(len(obs) for obs in self.iter_obs(subject_ids))

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

        return sum(a.interval_days for s in subject_ids for a in self.subjects[s].admissions)

    def interval_hours(self, subject_ids=None):
        """Get the total number of hours in-hospital.

        Args:
            subject_ids: List of subject IDs.

        Returns:
            int: Total number of hours in-hospital.
        """
        if subject_ids is None:
            subject_ids = self.subjects.keys()

        return sum(a.interval_hours for s in subject_ids for a in self.subjects[s].admissions)

    def p_obs(self, subject_ids=None):
        """Get the proportion of present observation over all unique timestamps for the given subject IDs.

        Args:
            subject_ids: list of subject IDs.

        Returns:
            float: proportion of observables presence per unique timestamp.
        """
        return sum(obs.mask.sum() for obs in self.iter_obs(subject_ids)) / self.n_obs_times() / len(self.scheme.obs)

    def obs_coocurrence_matrix(self, subject_ids=None):
        """Compute the co-occurrence (or co-presence) matrix of observables.

        Returns:
            jnp.ndarray: co-occurrence (or co-presence) matrix of observables.
        """
        obs = [obs.mask for obs in self.iter_obs(subject_ids) if len(obs) > 0]
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

    def _unscaled_observation(self, obs: InpatientObservables) -> InpatientObservables:
        """Unscale the observation values, undo the preprocessing scaling.

        Args:
            obs (InpatientObservables): observation to unscale.

        Returns:
            InpatientObservables: unscaled observation.
        """
        obs_scaler = self.numerical_processors.scalers.obs
        value = obs_scaler.unscale(obs.value)
        return InpatientObservables(time=obs.time, value=value, mask=obs.mask)

    def _unscaled_leading_observable(self, lead: InpatientObservables):
        """Unscale the leading observable values, undo the preprocessing scaling.

        Args:
            lead (InpatientObservables): leading observable to unscale.

        Returns:
            InpatientObservables: unscaled leading observable.
        """
        lead_scaler = self.numerical_processors.scalers.obs
        obs_index = self.config.leading_observable.code_index
        value = lead_scaler.unscale_code(lead.value, obs_index)
        return InpatientObservables(time=lead.time, value=value, mask=lead.mask)

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
    patient_class: ClassVar[Type[SegmentedPatient]] = SegmentedPatient

    @classmethod
    def _setup_pipeline(cls, config: Config) -> AbstractDatasetPipeline:
        raise NotImplementedError("SegmentedPatient is a final representation. It cannot have a pipeline.")

    def execute_pipeline(self) -> AbstractDatasetRepresentation:
        raise NotImplementedError("SegmentedPatient is a final representation. It cannot have a pipeline.")

    def iter_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                for obs_segment in adm.observables:
                    yield obs_segment

    def iter_lead_obs(self, subject_ids=None) -> Iterable[InpatientObservables]:
        if subject_ids is None:
            subject_ids = self.subjects.keys()
        for s in subject_ids:
            for adm in self.subjects[s].admissions:
                yield adm.leading_observable

    @staticmethod
    def from_tvx_ehr(tvx_ehr: TVxEHR, maximum_padding: int = 100) -> SegmentedTVxEHR:
        hosp_procedures_size = len(tvx_ehr.scheme.hosp_procedures) if tvx_ehr.scheme.hosp_procedures else None
        icu_procedures_size = len(tvx_ehr.scheme.icu_procedures) if tvx_ehr.scheme.icu_procedures else None
        icu_inputs_size = len(tvx_ehr.dataset.scheme.icu_inputs) if tvx_ehr.dataset.scheme.icu_inputs else None
        subjects = {k: SegmentedPatient.from_patient(v, hosp_procedures_size=hosp_procedures_size,
                                                     icu_procedures_size=icu_procedures_size,
                                                     icu_inputs_size=icu_inputs_size,
                                                     maximum_padding=maximum_padding) for k, v in
                    tvx_ehr.subjects.items()}
        return SegmentedTVxEHR(config=tvx_ehr.config, dataset=tvx_ehr.dataset,
                               numerical_processors=tvx_ehr.numerical_processors,
                               subjects=subjects, splits=tvx_ehr.splits, pipeline_report=tvx_ehr.pipeline_report)

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
