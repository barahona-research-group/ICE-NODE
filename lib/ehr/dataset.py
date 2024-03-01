"""."""
from __future__ import annotations

import logging
import random
from abc import abstractmethod, ABCMeta
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List, Any, Literal

import equinox as eqx
import numpy as np
import pandas as pd

from .coding_scheme import (CodingScheme, OutcomeExtractor, FileBasedOutcomeExtractor)
from .concepts import (DemographicVectorConfig)
from ..base import Config, Module, Data
from ..utils import write_config, load_config


class TableConfig(Config):
    @property
    def alias_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.as_dict().items() if k.endswith('_alias')}

    @property
    def alias_id_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.alias_dict.items() if '_id_' in k}

    @property
    def index(self) -> Optional[str]:
        return None

    @property
    def time_cols(self) -> Tuple[str, ...]:
        return tuple(v for k, v in self.alias_dict.items() if 'time' in k or 'date' in k)

    @property
    def coded_cols(self) -> Tuple[str, ...]:
        return tuple(v for k, v in self.alias_dict.items() if 'code' in k)


class AdmissionLinkedTableConfig(TableConfig):
    admission_id_alias: str


class SubjectLinkedTableConfig(TableConfig):
    subject_id_alias: str


class TimestampedTableConfig(TableConfig):
    time_alias: str


class TimestampedMultiColumnTableConfig(TimestampedTableConfig):
    attributes: Tuple[str, ...] = field(kw_only=True)
    type_hint: Tuple[Literal['N', 'C', 'B', 'O'], ...] = field(kw_only=True, default=None)
    default_type_hint: Literal['N', 'C', 'B', 'O'] = 'N'

    def __post_init__(self):
        if self.type_hint is None:
            self.type_hint = (self.default_type_hint,) * len(self.attributes)
        assert len(self.attributes) == len(self.type_hint), \
            f"Length of attributes and type_hint must be the same. Got {len(self.attributes)} and {len(self.type_hint)}."
        assert all(t in ('N', 'C', 'B', 'O') for t in self.type_hint), \
            f"Type hint must be one of 'N', 'C', 'B', 'O'. Got {self.type_hint}."


class CodedTableConfig(TableConfig):
    code_alias: str
    description_alias: str


class TimestampedCodedTableConfig(CodedTableConfig, TimestampedTableConfig):
    pass


class TimestampedCodedValueTableConfig(TimestampedCodedTableConfig):
    value_alias: str


class AdmissionLinkedCodedValueTableConfig(CodedTableConfig, AdmissionLinkedTableConfig):
    pass


class IntervalBasedTableConfig(TableConfig):
    start_time_alias: str
    end_time_alias: str


class AdmissionTableConfig(AdmissionLinkedTableConfig, SubjectLinkedTableConfig):
    admission_time_alias: str
    discharge_time_alias: str

    @property
    def index(self):
        return self.admission_id_alias


class StaticTableConfig(SubjectLinkedTableConfig):
    gender_alias: str
    race_alias: str
    date_of_birth_alias: str

    @property
    def index(self):
        return self.subject_id_alias


class AdmissionTimestampedMultiColumnTableConfig(TimestampedMultiColumnTableConfig, AdmissionLinkedTableConfig):
    name: str = field(kw_only=True)


class AdmissionTimestampedCodedValueTableConfig(TimestampedCodedValueTableConfig, AdmissionLinkedTableConfig):
    pass


class AdmissionIntervalBasedCodedTableConfig(IntervalBasedTableConfig, CodedTableConfig,
                                             AdmissionLinkedTableConfig):
    pass


class RatedInputTableConfig(AdmissionIntervalBasedCodedTableConfig):
    amount_alias: str
    amount_unit_alias: str
    derived_unit_normalization_factor: str
    derived_universal_unit: str
    derived_normalized_amount: str
    derived_normalized_amount_per_hour: str


class DatasetTablesConfig(Config):
    static: StaticTableConfig
    admissions: AdmissionTableConfig
    dx_discharge: Optional[AdmissionLinkedCodedValueTableConfig]
    obs: Optional[AdmissionTimestampedCodedValueTableConfig]
    icu_procedures: Optional[AdmissionIntervalBasedCodedTableConfig]
    icu_inputs: Optional[RatedInputTableConfig]
    hosp_procedures: Optional[AdmissionIntervalBasedCodedTableConfig]

    def __post_init__(self):
        self._assert_consistent_aliases()

    def _assert_consistent_aliases(self):
        config_dict = self.table_config_dict

        for k, v in config_dict.items():
            if k == 'static' or not isinstance(v, SubjectLinkedTableConfig):
                continue
            assert v.subject_id_alias == self.static.subject_id_alias, \
                f"Subject id alias for {k} must be the same as the one in static table. Got {v.subject_id_alias}." \
                f"Expected {self.static.subject_id_alias}."

        for k, v in config_dict.items():
            if k == 'admissions' or not isinstance(v, AdmissionLinkedTableConfig):
                continue
            assert v.admission_id_alias == self.admissions.admission_id_alias, \
                f"Admission id alias for {k} must be the same as the one in admissions table. Got {v.admission_id_alias}." \
                f"Expected {self.admissions.admission_id_alias}."

    @property
    def admission_id_alias(self):
        return self.admissions.admission_id_alias

    @property
    def subject_id_alias(self):
        return self.static.subject_id_alias

    @property
    def table_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, TableConfig)}

    @property
    def timestamped_table_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, TimestampedTableConfig)}

    @property
    def interval_based_table_config_dict(self):
        return {k: v for k, v in self.__dict__.items() if
                isinstance(v, IntervalBasedTableConfig)}

    @property
    def indices(self) -> Dict[str, str]:
        return {
            k: v.index
            for k, v in self.__dict__.items()
            if isinstance(v, TableConfig) and v.index is not None
        }

    @property
    def time_cols(self) -> Dict[str, Tuple[str, ...]]:
        return {
            k: v.time_cols
            for k, v in self.__dict__.items()
            if isinstance(v, TableConfig) and len(v.time_cols) > 0
        }

    @property
    def code_column(self) -> Dict[str, str]:
        return {k: v.code_alias for k, v in self.__dict__.items() if isinstance(v, CodedTableConfig)}


class DatasetTables(Data):
    static: pd.DataFrame
    admissions: pd.DataFrame
    dx_discharge: Optional[pd.DataFrame] = None
    obs: Optional[pd.DataFrame] = None
    icu_procedures: Optional[pd.DataFrame] = None
    icu_inputs: Optional[pd.DataFrame] = None
    hosp_procedures: Optional[pd.DataFrame] = None

    @property
    def tables_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if isinstance(v, pd.DataFrame)
        }

    @staticmethod
    def table_meta(df: pd.DataFrame) -> pd.DataFrame:
        meta = {
            f'index_dtype': str(df.index.dtype),
            f'index_name': str(df.index.name)}
        meta.update({
            f'column_{i}': col for i, col in enumerate(df.columns)})
        meta.update({
            f'column_dtype_{df.columns[i]}': str(dtype) for i, dtype in enumerate(df.dtypes)})
        return pd.DataFrame(meta, index=[0])

    @staticmethod
    def empty_table(meta: pd.DataFrame) -> pd.DataFrame:
        meta = meta.iloc[0].to_dict()
        index_dtype = meta.pop('index_dtype')
        index_name = meta.pop('index_name')
        cols = []
        column_types = {}
        while len(meta) > 0:
            k, v = meta.popitem()
            if k.startswith('column_dtype_'):
                column_types[k.split('column_dtype_')[1]] = v
            elif k.startswith('column_'):
                order = int(k.split('column_')[1])
                cols.append((order, v))
        cols = [col for _, col in sorted(cols, key=lambda x: x[0])]
        df = pd.DataFrame(columns=cols).astype(column_types)
        df.index = pd.Index([], dtype=index_dtype)
        if index_name != 'None':
            df.index.name = index_name
        return df

    def save(self, path: Union[str, Path], overwrite: bool):
        if Path(path).exists():
            if overwrite:
                Path(path).unlink()
            else:
                raise RuntimeError(f'File {path} already exists.')

        for name, df in self.tables_dict.items():
            filepath = Path(path)

            # Empty dataframes are not saved normally, https://github.com/pandas-dev/pandas/issues/13016,
            # https://github.com/PyTables/PyTables/issues/592
            # So we keep the column names, dtypes, index name, index dtype, in a metadata object.
            if df.empty:
                self.table_meta(df).to_hdf(filepath, key=f'{name}_meta', format='table')
            else:
                df.to_hdf(filepath, key=name, format='table')

    @staticmethod
    def load(path: Union[str, Path]) -> DatasetTables:
        with pd.HDFStore(path, mode='r') as store:
            tables = {k.split('/')[1]: store[k] for k in store.keys() if not k.endswith('_meta')}
            meta = {k.split('/')[1].split('_meta')[0]: store[k] for k in store.keys() if k.endswith('_meta')}
            for table_name, meta_df in meta.items():
                assert table_name not in tables, f'Table {table_name} is both in the tables and meta dict.'
                tables[table_name] = DatasetTables.empty_table(meta_df)

            return DatasetTables(**tables)

    def equals(self, other: DatasetTables) -> bool:
        return all(
            self.__dict__[k].equals(other.__dict__[k])
            for k in self.__dict__.keys()
            if isinstance(self.__dict__[k], pd.DataFrame)
        )


class DatasetSchemeConfig(Config):
    ethnicity: str
    gender: str
    dx_discharge: Optional[str]
    obs: Optional[str] = None
    icu_procedures: Optional[str] = None
    hosp_procedures: Optional[str] = None
    icu_inputs: Optional[str] = None
    outcome: Optional[str] = None


class DatasetScheme(Module):
    """
    Represents a dataset scheme that defines the coding schemes and outcome extractor for a dataset.

    Attributes:
        config (DatasetSchemeConfig): the configuration for the dataset scheme.
        dx_discharge (CodingScheme): the coding scheme for the diagnosis.
        ethnicity (CodingScheme): the coding scheme for the ethnicity.
        gender (CodingScheme): the coding scheme for the gender.
        outcome (Optional[OutcomeExtractor]): the outcome extractor for the dataset, if specified.

    Methods:
        __init__(self, config: DatasetSchemeConfig, **kwargs): initializes a new instance of the DatasetScheme class.
        scheme_dict(self): returns a dictionary of the coding schemes in the dataset scheme.
        make_target_scheme_config(self, **kwargs): creates a new target scheme configuration based on the current scheme.
        make_target_scheme(self, config=None, **kwargs): creates a new target scheme based on the current scheme.
        demographic_vector_size(self, demographic_vector_config: DemographicVectorConfig): calculates the size of the demographic vector.
        dx_mapper(self, target_scheme: DatasetScheme): returns the mapper for the diagnosis coding scheme to the corresponding target scheme.
        ethnicity_mapper(self, target_scheme: DatasetScheme): returns the mapper for the ethnicity coding scheme to the corresponding target scheme.
        supported_target_scheme_options(self): returns the supported target scheme options for each coding scheme.
    """
    config: DatasetSchemeConfig

    @staticmethod
    def _scheme(name: str) -> Optional[CodingScheme]:
        try:
            return CodingScheme.from_name(name)
        except KeyError as e:
            return None

    @cached_property
    def ethnicity(self) -> CodingScheme:
        return self._scheme(self.config.ethnicity)

    @cached_property
    def gender(self) -> CodingScheme:
        return self._scheme(self.config.gender)

    @cached_property
    def dx_discharge(self) -> CodingScheme:
        return self._scheme(self.config.dx_discharge)

    @cached_property
    def obs(self) -> Optional[CodingScheme]:
        return self._scheme(self.config.obs)

    @cached_property
    def icu_procedures(self) -> Optional[CodingScheme]:
        return self._scheme(self.config.icu_procedures)

    @cached_property
    def hosp_procedures(self) -> Optional[CodingScheme]:
        return self._scheme(self.config.hosp_procedures)

    @cached_property
    def icu_inputs(self) -> Optional[CodingScheme]:
        return self._scheme(self.config.icu_inputs)

    @cached_property
    def outcome(self) -> Optional[OutcomeExtractor]:
        return OutcomeExtractor.from_name(self.config.outcome) if self.config.outcome else None

    @property
    def scheme_dict(self):
        return {
            k: self._scheme(v)
            for k, v in self.config.as_dict().items() if isinstance(self._scheme(v), CodingScheme)
        }

    @classmethod
    def _assert_valid_maps(cls, source, target):
        for attr in source.scheme_dict.keys():

            if attr == 'outcome':
                continue

            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme.name
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"

    def make_target_scheme_config(self, **kwargs):
        return self.config.update(**kwargs)

    def make_target_scheme(self, config=None, **kwargs):
        if config is None:
            config = self.make_target_scheme_config(**kwargs)
        t_scheme = type(self)(config)
        self._assert_valid_maps(self, t_scheme)
        return t_scheme

    def demographic_vector_size(
            self, demographic_vector_config: DemographicVectorConfig):
        size = 0
        if demographic_vector_config.gender:
            size += len(self.gender)
        if demographic_vector_config.age:
            size += 1
        if demographic_vector_config.ethnicity:
            size += len(self.ethnicity)
        return size

    def dx_mapper(self, target_scheme: DatasetScheme):
        return self.dx_discharge.mapper_to(target_scheme.dx_discharge.name)

    def ethnicity_mapper(self, target_scheme: DatasetScheme):
        return self.ethnicity.mapper_to(target_scheme.ethnicity.name)

    @property
    def supported_target_scheme_options(self):
        supported_attr_targets = {
            k: (v.name,) + v.supported_targets
            for k, v in self.scheme_dict.items() if not isinstance(v, OutcomeExtractor)
        }
        supported_outcomes = FileBasedOutcomeExtractor.supported_outcomes(self.dx_discharge.name)
        supported_attr_targets['outcome'] = supported_outcomes
        return supported_attr_targets


class AbstractDatasetTransformation(eqx.Module):
    name: str = None

    @abstractmethod
    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        pass


class AbstractDatasetPipelineConfig(Config):
    pass


class AbstractDatasetPipeline(Module, metaclass=ABCMeta):
    transformations: List[AbstractDatasetTransformation]

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dict[str, pd.DataFrame]]:
        pass


class DatasetConfig(Config):
    scheme: DatasetSchemeConfig
    tables: DatasetTablesConfig
    pipeline: AbstractDatasetPipelineConfig
    pipeline_executed: bool = False


class Dataset(Module):
    """
    A class representing a dataset.

    Attributes:
        df (Dict[str, pd.DataFrame]): a dictionary of dataframes, where the keys are the names of the dataframes.
        config (DatasetConfig): the configuration object for the dataset.
        scheme (DatasetScheme): the scheme object for the dataset.
        colname (Dict[str, ColumnNames]): a dictionary of column names, where the keys are the names of the dataframes.

    Methods:
        __init__(self, config: DatasetConfig = None, config_path: str = None, **kwargs): initializes the Dataset object.
        supported_target_scheme_options(self): returns the supported target scheme options.
        to_subjects(self, **kwargs): converts the dataset to subject objects.
        save(self, path: Union[str, Path], overwrite: bool = False): saves the dataset to disk.
        load(cls, path: Union[str, Path]): loads the dataset from disk.
    """
    config: DatasetConfig
    tables: DatasetTables
    scheme: DatasetScheme
    core_pipeline: AbstractDatasetPipeline
    core_pipeline_report: pd.DataFrame

    def __init__(self, config: DatasetConfig):
        super().__init__(config=config)
        self.scheme = self.load_dataset_scheme(config)
        self.tables = self.load_tables(config, self.scheme)
        self.core_pipeline = self._setup_core_pipeline(config)
        self.core_pipeline_report = pd.DataFrame()

    @classmethod
    def load_dataset_scheme(cls, config: DatasetConfig) -> DatasetScheme:
        return DatasetScheme(config=config.scheme)

    @classmethod
    @abstractmethod
    def load_tables(cls, config: DatasetConfig, scheme: DatasetScheme) -> DatasetTables:
        pass

    @classmethod
    @abstractmethod
    def _setup_core_pipeline(cls, config: DatasetConfig) -> AbstractDatasetPipeline:
        pass

    def execute_pipeline(self) -> Dataset:
        if len(self.core_pipeline_report) > 0:
            logging.warning("Pipeline has already been executed. Doing nothing.")
            return self
        dataset, aux = self.core_pipeline(self)

        dataset = eqx.tree_at(lambda x: x.config.pipeline_executed, dataset, True)
        dataset = eqx.tree_at(lambda x: x.core_pipeline_report, dataset, aux['report'])
        return dataset

    @cached_property
    def supported_target_scheme_options(self):
        return self.scheme.supported_target_scheme_options

    @cached_property
    def subject_ids(self):
        assert self.config.pipeline_executed, "Pipeline must be executed first."
        assert self.tables.static.index.name == self.config.tables.static.subject_id_alias, \
            f"Index name of static table must be {self.config.tables.static.subject_id_alias}."
        return self.tables.static.index.unique()

    @cached_property
    def subjects_intervals_sum(self) -> pd.Series:
        c_admittime = self.config.tables.admissions.admission_time_alias
        c_dischtime = self.config.tables.admissions.discharge_time_alias
        c_subject_id = self.config.tables.admissions.subject_id_alias
        admissions = self.tables.admissions
        interval = (admissions[c_dischtime] - admissions[c_admittime]).dt.total_seconds()
        admissions = admissions.assign(interval=interval)
        return admissions.groupby(c_subject_id)['interval'].sum()

    @cached_property
    def subjects_n_admissions(self) -> pd.Series:
        c_subject_id = self.config.tables.admissions.subject_id_alias
        admissions = self.tables.admissions
        return admissions.groupby(c_subject_id).size()

    def random_splits(self,
                      splits: List[float],
                      subject_ids: Optional[List[str]] = None,
                      random_seed: int = 42,
                      balance: str = 'subjects',
                      discount_first_admission: bool = False):
        assert list(splits) == sorted(splits), "Splits must be sorted."
        assert balance in ('subjects', 'admissions',
                           'admissions_intervals'), "Balanced must be'subjects', 'admissions', or 'admissions_intervals'."
        if subject_ids is None:
            subject_ids = self.subject_ids
        assert len(subject_ids) > 0, "No subjects in the dataset."

        subject_ids = sorted(subject_ids)

        random.Random(random_seed).shuffle(subject_ids)
        subject_ids = np.array(subject_ids)

        c_subject_id = self.config.tables.static.subject_id_alias

        admissions = self.tables.admissions[self.tables.admissions[c_subject_id].isin(subject_ids)]

        if balance == 'subjects':
            probs = (np.ones(len(subject_ids)) / len(subject_ids)).cumsum()

        elif balance == 'admissions':
            assert len(admissions) > 0, "No admissions in the dataset."
            n_admissions = self.subjects_n_admissions.loc[subject_ids]
            if discount_first_admission:
                n_admissions = n_admissions - 1
            p_admissions = n_admissions / n_admissions.sum()
            probs = p_admissions.values.cumsum()

        elif balance == 'admissions_intervals':
            assert len(admissions) > 0, "No admissions in the dataset."
            subjects_intervals_sum = self.subjects_intervals_sum.loc[subject_ids]
            p_subject_intervals = subjects_intervals_sum / subjects_intervals_sum.sum()
            probs = p_subject_intervals.values.cumsum()
        else:
            raise ValueError(f'Unknown balanced option: {balance}')

        # Deal with edge cases where the splits are exactly the same as the probabilities.
        for i in range(len(splits)):
            if any(abs(probs - splits[i]) < 1e-6):
                splits[i] = splits[i] + 1e-6

        splits = np.searchsorted(probs, splits)
        return [a.tolist() for a in np.split(subject_ids, splits)]

    def equals(self, other: 'Dataset'):
        return self.config == other.config and self.tables.equals(other.tables) and self.scheme == other.scheme and \
            self.core_pipeline_report.equals(other.core_pipeline_report)

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tables.save(path.with_suffix('.tables.h5'), overwrite)
        self.core_pipeline_report.to_hdf(path.with_suffix('.pipeline.h5'),
                                         key='report',
                                         format='table')

        config_path = path.with_suffix('.config.json')
        if config_path.exists():
            if overwrite:
                config_path.unlink()
            else:
                raise RuntimeError(f'File {config_path} already exists.')
        write_config(self.config.to_dict(), str(config_path))

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        tables = DatasetTables.load(path.with_suffix('.tables.h5'))
        config = DatasetConfig.from_dict(load_config(str(path.with_suffix('.config.json'))))
        dataset = eqx.tree_at(lambda x: x.tables, cls(config=config), tables,
                              is_leaf=lambda x: x is None)
        with pd.HDFStore(path.with_suffix('.pipeline.h5')) as store:
            report = store['report'] if 'report' in store else pd.DataFrame()
            return eqx.tree_at(lambda x: x.core_pipeline_report, dataset, report)

# TODO: 31 Jan 2024:
#  - [x] change from_fit to __init__().fit()
#  - [x] SQLTableConfig to inherit from DatasetTablesConfig
#  - [x] Assert functions to check the consistency of subject_id, admission_id in all tables.
#  - [x] List the three main test cases for merge_overlapping_admissions.
#  - [ ] Interface Structure: Controls (icu_inputs, icu_procedures, hosp_procedures), InitObs (dx_codes or dx_history), Obs (obs), Lead(lead(obs))
#  - [ ] Move Predictions/AdmissionPredictions to lib.ml.
#  - [ ] Plan a week of refactoring/testing/documentation/ship the lib.ehr separately.
#  - [ ] Publish Website for lib.ehr: decide on the lib name, decide on the website name.
