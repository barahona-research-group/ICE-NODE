"""."""
from __future__ import annotations

import logging
import random
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any, List

import equinox as eqx
import pandas as pd

from .coding_scheme import (CodingScheme, OutcomeExtractor)
from .concepts import (DemographicVectorConfig)
from ..base import Config, Module
from ..utils import write_config, load_config

SECONDS_TO_HOURS_SCALER: float = 1 / 3600.0  # convert seconds to hours


class TableConfig(Config):
    name: str

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
    admission_id_alias: str = 'hadm_id'


class SubjectLinkedTableConfig(TableConfig):
    subject_id_alias: str = 'subject_id'


class TimestampedTableConfig(TableConfig):
    time_alias: str = 'time_bin'


class TimestampedMultiColumnTableConfig(TimestampedTableConfig):
    attributes: Tuple[str] = tuple()


class CodedTableConfig(TableConfig):
    code_alias: str = 'code'
    description_alias: str = 'description'


class TimestampedCodedTableConfig(CodedTableConfig, TimestampedTableConfig):
    pass


class TimestampedCodedValueTableConfig(TimestampedCodedTableConfig):
    value_alias: str = 'value'


class IntervalBasedTableConfig(TableConfig):
    start_time_alias: str = 'start_time'
    end_time_alias: str = 'end_time'


class AdmissionTableConfig(AdmissionLinkedTableConfig, SubjectLinkedTableConfig):
    admission_time_alias: str = 'admittime'
    discharge_time_alias: str = 'dischtime'

    @property
    def index(self):
        return self.admission_id_alias


class StaticTableConfig(SubjectLinkedTableConfig):
    gender_alias: str = 'gender'
    race_alias: str = 'race'
    date_of_birth_alias: str = 'dob'

    @property
    def index(self):
        return self.subject_id_alias


class AdmissionMixedICDTableConfig(CodedTableConfig, AdmissionLinkedTableConfig):
    icd_code_alias: str = 'icd_code'
    icd_version_alias: str = 'icd_version'


class AdmissionTimestampedMultiColumnTableConfig(TimestampedMultiColumnTableConfig, AdmissionLinkedTableConfig):
    pass


class AdmissionTimestampedCodedValueTableConfig(TimestampedCodedValueTableConfig, AdmissionLinkedTableConfig):
    pass


class AdmissionIntervalBasedCodedTableConfig(IntervalBasedTableConfig, CodedTableConfig,
                                             AdmissionTableConfig):
    pass


class AdmissionIntervalBasedMixedICDTableConfig(IntervalBasedTableConfig, AdmissionMixedICDTableConfig):
    pass


class RatedInputTableConfig(AdmissionIntervalBasedCodedTableConfig):
    amount_alias: str = 'amount'
    amount_unit_alias: str = 'amountuom'
    derived_amount_per_hour: str = 'derived_amount_per_hour'
    derived_normalized_amount_per_hour: str = 'derived_normalized_amount_per_hour'
    derived_unit_normalization_factor: str = 'derived_unit_normalization_factor'
    derived_universal_unit: str = 'derived_universal_unit'


class DatasetTablesConfig(Config):
    static: StaticTableConfig
    admissions: AdmissionTableConfig
    dx_discharge: Optional[AdmissionMixedICDTableConfig] = None
    obs: Optional[List[TimestampedTableConfig]] = None
    icu_procedures: Optional[IntervalBasedTableConfig] = None
    icu_inputs: Optional[RatedInputTableConfig] = None
    hosp_procedures: Optional[IntervalBasedTableConfig] = None

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


class DatasetTables(Module):
    static: pd.DataFrame
    admissions: pd.DataFrame
    dx_discharge: Optional[pd.DataFrame] = None
    obs: Optional[List[pd.DataFrame]] = None
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

    def save(self, path: Union[str, Path], overwrite: bool):
        for name, df in self.tables_dict.items():
            filepath = Path(path)
            if filepath.exists():
                if overwrite:
                    filepath.unlink()
                else:
                    raise RuntimeError(f'File {path} already exists.')

            df.to_hdf(filepath, key=name, format='table')

    @staticmethod
    def load(path: Union[str, Path]) -> DatasetTables:
        with pd.HDFStore(path) as store:
            return DatasetTables(**{k: store[k] for k in store.keys()})


class DatasetSchemeConfig(Config):
    ethnicity: str
    gender: str
    dx_discharge: str
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
    ethnicity: CodingScheme
    gender: CodingScheme
    dx_discharge: CodingScheme
    icu_procedures: Optional[CodingScheme] = None
    hosp_procedures: Optional[CodingScheme] = None
    icu_inputs: Optional[CodingScheme] = None
    outcome: Optional[OutcomeExtractor] = None

    def __init__(self, config: DatasetSchemeConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        config = self.config.as_dict()

        if config.get('outcome'):
            self.outcome = OutcomeExtractor.from_name(config.pop('outcome'))

        for k, v in config.items():
            if isinstance(v, str):
                setattr(self, k, CodingScheme.from_name(v))

    @property
    def scheme_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items() if isinstance(v, CodingScheme)
        }

    @classmethod
    def _assert_valid_maps(cls, source, target):
        for attr in source.scheme_dict.keys():

            if attr == 'outcome':
                continue

            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"

    def make_target_scheme_config(self, **kwargs):
        assert 'outcome' in kwargs, "Outcome must be specified"
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
        supproted_attr_targets = {
            k: (getattr(self, k).__class__.__name__,) +
               getattr(self, k).supported_targets
            for k in self.scheme_dict if k != 'outcome'
        }
        supported_outcomes = OutcomeExtractor.supported_outcomes(self.dx_discharge.name)
        supproted_attr_targets['outcome'] = supported_outcomes
        return supproted_attr_targets


class ReportAttributes(Config):
    transformation: str = None
    operation: str = None
    table: str = None
    column: str = None
    value_type: str = None
    before: Any = None
    after: Any = None
    additional_parameters: Dict[str, Any] = None


class DatasetTransformation(Module):
    name: str = None

    @property
    def additional_parameters(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != 'name' or not k.startswith('_')}

    def report(self, aux: Dict[str, Any], **kwargs):
        if aux.get('report') is None:
            aux['report'] = []
        additional_params_str = ', '.join([f"{k}={v}" for k, v in self.additional_parameters.items()])
        aux['report'].append(ReportAttributes(transformation=self.name or type(self).__name__,
                                              additional_parameters=additional_params_str,
                                              **kwargs))

    def synchronize_index(self, dataset: Dataset, indexed_table_name: str,
                          index_name: str, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        tables_dict = dataset.tables.tables_dict

        target_tables = {  # tables that have admission_id as column
            k: v for k, v in
            tables_dict.items()
            if k != indexed_table_name and index_name in v.columns
        }

        index = tables_dict[indexed_table_name].index
        tables = dataset.tables
        for table_name, table in target_tables.items():
            n1 = len(table)
            table = table[table[index_name].isin(index)]
            n2 = len(table)
            self.report(aux, table=table_name, column=index_name, before=n1, after=n2, value_type='count',
                        operation='sync_index')
            tables = eqx.tree_at(lambda x: getattr(tables, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), aux

    def filter_no_admission_subjects(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        static = dataset.tables.static
        admissions = dataset.tables.admissions
        c_subject = dataset.config.tables.static.subject_id_alias
        no_admission_subjects = static[~static.index.isin(admissions[c_subject].unique())].index
        n1 = len(static)
        static = static.drop(no_admission_subjects, axis='index')
        n2 = len(static)
        self.report(aux, table='static', column=c_subject, before=n1, after=n2, value_type='count',
                    operation='filter_no_admission_subjects')
        return eqx.tree_at(lambda x: x.tables.static, dataset, static), aux

    def synchronize_admissions(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        dataset, aux = self.synchronize_index(dataset, 'admissions',
                                              dataset.config.tables.admissions.admission_id_alias, aux)
        return self.filter_no_admission_subjects(dataset, aux)

    def synchronize_subjects(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        # Synchronizing subjects might entail synchronizing admissions, so we need to call it first
        dataset, aux = self.synchronize_index(dataset, 'static',
                                              dataset.config.tables.static.subject_id_alias, aux)
        return self.synchronize_admissions(dataset, aux)

    @abstractmethod
    def __call__(self, dataset: Dataset, auxiliary) -> Tuple[Dataset, Dict[str, Any]]:
        pass


class DatasetPipeline(Module):
    transformations: List[DatasetTransformation]

    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dict[str, Any]]:
        auxiliary = {'report': []}
        current_report_list = []
        for t in self.transformations:
            dataset, auxiliary_ = t(dataset, auxiliary)
            auxiliary.update(auxiliary_)
            if auxiliary.get('report'):
                new_report_list = auxiliary.get('report').copy()
                transformation_report = new_report_list[len(current_report_list):]
                current_report_list = new_report_list

                if len(transformation_report) > 0:
                    report_df = pd.DataFrame([x.as_dict() for x in transformation_report])
                    report_str = report_df.to_string().replace('\n', '\n\t')
                    logging.debug(f"Transformation Statistics: {t.name or type(t).__name__}:\n{report_str}")

        if auxiliary.get('report'):
            report = pd.DataFrame([x.as_dict() for x in auxiliary['report']])
            auxiliary['report'] = report
            logging.info(report.to_string().replace('\n', '\n\t'))
        return dataset, auxiliary


class DatasetPipelineConfig(Config):
    overlapping_admissions: str = 'merge'
    sample: Optional[int] = None
    offset: Optional[int] = 0


class DatasetConfig(Config):
    scheme: DatasetSchemeConfig
    tables: DatasetTablesConfig
    dataset_pipeline: DatasetPipelineConfig
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
    pipeline: DatasetPipeline
    pipeline_report: pd.DataFrame = pd.DataFrame()

    def __init__(self, config: DatasetConfig, tables: DatasetTables):
        super().__init__(config=config)
        self.tables = tables
        self.scheme = DatasetScheme(config=config.scheme)
        self.pipeline = self._setup_pipeline(config)

    @classmethod
    @abstractmethod
    def _setup_pipeline(cls, config: DatasetConfig) -> DatasetPipeline:
        pass

    def execute_pipeline(self) -> Dataset:
        if len(self.pipeline_report) > 0:
            logging.warning("Pipeline has already been executed. Doing nothing.")
            return self
        dataset, aux = self.pipeline(self)

        dataset = eqx.tree_at(lambda x: x.config.pipeline_executed, dataset, True)
        dataset = eqx.tree_at(lambda x: x.pipeline_report, dataset, aux['report'])
        return dataset

    @property
    def supported_target_scheme_options(self):
        return self.scheme.supported_target_scheme_options

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tables.save(path.with_suffix('.tables.h5'), overwrite)
        self.pipeline_report.to_hdf(path.with_suffix('.pipeline.h5'), key='report', format='table')

        config_path = path.with_suffix('.config.json')
        if config_path.exists():
            if overwrite:
                config_path.unlink()
        else:
            raise RuntimeError(f'File {config_path} already exists.')
        write_config(self.config.to_dict(),
                     path.with_suffix('.config.json'))

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        tables = DatasetTables.load(path.with_suffix('.tables.h5'))
        with pd.HDFStore(path.with_suffix('.pipeline.h5')) as store:
            pipeline_report = store['report']

        config_path = path.with_suffix('.config.json')
        config = DatasetConfig.from_dict(load_config(config_path))
        dataset = cls(config=config, tables=tables)
        dataset = eqx.tree_at(lambda x: x.pipeline_report, dataset, pipeline_report)
        return dataset


class SampleSubjects(DatasetTransformation):
    n_subjects: int
    seed: Optional[int]
    offset: int

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        static = dataset.tables.static
        # assert index name is subject_id
        c_subject_id = dataset.config.tables.static.subject_id_alias
        assert c_subject_id in static.index.names, f'Index name must be {c_subject_id}'

        rng = random.Random(self.seed)
        subjects = static[c_subject_id].unique()
        subjects = subjects.tolist()
        rng.shuffle(subjects)
        subjects = subjects[self.offset:self.offset + self.n_subjects]
        n1 = len(static)
        static = static[static[c_subject_id].isin(subjects)]
        n2 = len(static)
        self.report(aux, table='static', column=c_subject_id, before=n1, after=n2, value_type='count',
                    operation='sample')
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, aux)


class CastTimestamps(DatasetTransformation):
    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        tables = dataset.tables
        tables_dict = tables.tables_dict
        for table_name, time_cols in dataset.config.tables.time_cols.items():
            if len(time_cols) == 0:
                continue
            table = tables_dict[table_name]

            for time_col in time_cols:
                assert time_col in table.columns, f'{time_col} not found in {table_name}'

                if table[time_col].dtype == 'datetime64[ns]':
                    logging.debug(f'{table_name}[{time_col}] already in datetime64[ns]')
                    continue
                dtype1 = table[time_col].dtype
                table[time_col] = pd.to_datetime(table[time_col], errors='raise')
                dtype2 = table[time_col].dtype
                self.report(aux, table=table_name, column=time_col, before=dtype1, after=dtype2, value_type='dtype',
                            operation='cast')

            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)
        return eqx.tree_at(lambda x: x.tables, dataset, tables), aux


class FilterCodes(DatasetTransformation):

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column in dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(dataset.scheme, table_name)
            n1 = len(table)
            table = table[table[code_column].isin(coding_scheme.codes)]
            n2 = len(table)
            self.report(aux, table=table_name, column=code_column, before=n1, after=n2, value_type='count',
                        operation='filter')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, aux


class SetAdmissionRelativeTimes(DatasetTransformation):
    @staticmethod
    def temporal_admission_linked_table(dataset: Dataset, table_name: str) -> bool:
        conf = getattr(dataset.config.tables, table_name)
        temporal = isinstance(conf, TimestampedTableConfig) or isinstance(conf, IntervalBasedTableConfig)
        admission_linked = isinstance(conf, AdmissionLinkedTableConfig)
        return temporal and admission_linked

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        time_cols = {k: v for k, v in dataset.config.tables.time_cols.items()
                     if self.temporal_admission_linked_table(dataset, k)}

        c_admittime = dataset.config.tables.admissions.admission_time_alias
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        admissions = dataset.tables.admissions[[c_admittime]]
        tables_dict = dataset.tables.tables_dict

        for table_name, time_cols in time_cols.items():
            table = tables_dict[table_name]
            df = pd.merge(table, admissions,
                          left_on=c_admission_id,
                          right_index=True,
                          how='left')
            for time_col in time_cols:
                df = df.assign(time_col=(df[time_col] - df[c_admittime]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER)
                self.report(aux, table=table_name, column=time_col, before=table[time_col].dtype,
                            after=df[time_col].dtype,
                            value_type='dtype', operation='set_admission_relative_times')

            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, df)
        return dataset, aux


class FilterSubjectsNegativeAdmissionLengths(DatasetTransformation):
    name: str = 'FilterSubjectsNegativeAdmissionLengths'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        c_subject = dataset.config.tables.static.subject_id_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias
        # assert dtypes are datetime64[ns]
        assert dataset.tables.static[c_admittime].dtype == 'datetime64[ns]' and \
               dataset.tables.static[c_dischtime].dtype == 'datetime64[ns]', \
            f'{c_admittime} and {c_dischtime} must be datetime64[ns]'

        admissions = dataset.tables.admissions
        static = dataset.tables.static
        neg_los_subjects = admissions[admissions[c_dischtime] < admissions[c_admittime]][c_subject].unique()
        n_before = len(static)
        static = static[~static.index.isin(neg_los_subjects)]
        n_after = len(static)
        self.report(aux, table='static', column=c_subject, value_type='count', operation='filter',
                    before=n_before, after=n_after)
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, aux)


class SetCodeIntegerIndices(DatasetTransformation):
    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column in dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(dataset.scheme, table_name)
            dtype1 = table[code_column].dtype
            table = table.assign(code_column=table[code_column].map(coding_scheme.index)).astype({code_column: int})
            dtype2 = table[code_column].dtype
            self.report(aux, table=table_name, column=code_column, before=dtype1, after=dtype2, value_type='dtype',
                        operation='set_index')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, aux


class SetIndex(DatasetTransformation):
    name: str = 'SetIndex'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        tables_dict = dataset.tables.tables_dict
        for indexed_table_name, index_name in dataset.config.tables.indices.items():
            table = tables_dict[indexed_table_name]
            index1 = table.index.name
            table = table.set_index(index_name)
            index2 = table.index.name
            self.report(aux, table=indexed_table_name, column=index_name, before=index1, after=index2,
                        value_type='index_name',
                        operation='set_index')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, indexed_table_name), dataset, table)
        return dataset, aux


class ProcessOverlappingAdmissions(DatasetTransformation):
    merge: bool  # if True, merge overlapping admissions. Otherwise, remove overlapping admissions.

    def map_admission_ids(self, dataset: Dataset, aux: Dict[str, Any], sub2sup: Dict[str, str]) -> Tuple[
        Dataset, Dict[str, Any]]:
        tables_dict = dataset.tables.tables_dict
        c_admission_id = dataset.config.tables.admissions.admission_id_alias

        target_tables = {  # tables that have admission_id as column
            k: v for k, v in
            tables_dict.items()
            if k != 'admissions' and c_admission_id in v.columns
        }

        tables = dataset.tables
        for table_name, table in target_tables.items():
            n1 = table[c_admission_id].nunique()
            table = table.assign(c_admission_id=table[c_admission_id].map(sub2sup))
            n2 = table[c_admission_id].nunique()
            self.report(aux, table=table_name, column=c_admission_id, before=n1, after=n2, value_type='nunique',
                        operation='map_admission_id')
            tables = eqx.tree_at(lambda x: getattr(tables, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), aux

    @staticmethod
    def _collect_overlaps(subject_admissions, c_admittime, c_dischtime):
        # Sort by admission time.
        subject_admissions = subject_admissions.sort_values(c_admittime)

        # Previous discharge time.
        index = subject_admissions.index
        subject_admissions.loc[index[1:], 'prev_dischtime'] = subject_admissions.loc[index[:-1], c_dischtime].values
        # Cumulative-max of previous discharge time.
        subject_admissions['prev_dischtime_cummax'] = subject_admissions['prev_dischtime'].cummax()

        # Get corresponding index of the maximum discharge time up to the current admission.
        lambda_fn = lambda x: subject_admissions[subject_admissions['prev_dischtime_cummax'] == x].first_valid_index()
        subject_admissions['prev_dischtime_cummax_idx'] = subject_admissions['prev_dischtime_cummax'].map(lambda_fn)

        # Drop admissions with admittime after the prev_max discharge time. No overlaps with preceding admissions.
        # Note: this line needs to come after adding 'prev_dischtime_cummax_idx' column.
        subject_admissions = subject_admissions[
            subject_admissions[c_admittime] > subject_admissions['prev_dischtime_cummax']]
        subject_admissions = subject_admissions[subject_admissions['prev_dischtime_cummax_idx'].notnull()]

        # New admissions mappings.
        return subject_admissions['prev_dischtime_cummax_idx'].to_dict()

    def _merge_overlapping_admissions(self,
                                      dataset: Dataset, aux: Dict[str, Any],
                                      sub2sup: Dict[str, str]) -> Tuple[Dataset, Dict[str, Any]]:
        admissions = dataset.tables.admissions
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias

        # Map from super-admissions to its sub-admissions.
        sup2sub = defaultdict(list)
        for sub, sup in sub2sup.items():
            sup2sub[sup].append(sub)

        # Step 2: Merge overlapping admissions by extending discharge time to the maximum discharge
        # time of its sub-admissions.
        for super_idx, sub_indices in sup2sub.items():
            current_dischtime = admissions.loc[super_idx, c_dischtime]
            new_dischtime = max(admissions.loc[sub_indices, c_dischtime].max(), current_dischtime)
            admissions.loc[super_idx, c_dischtime] = new_dischtime

        # Step 3: Remove sub-admissions.
        n1 = len(admissions)
        admissions = admissions.drop(list(sub2sup.keys()), axis='index')
        n2 = len(admissions)
        dataset = eqx.tree_at(lambda x: x.tables.admissions, dataset, admissions)
        self.report(aux, table='admissions', column=c_admission_id, value_type='count',
                    operation='merge_overlapping_admissions',
                    before=n1, after=n2)

        # Step 4: update admission ids in other tables.
        return self.map_admission_ids(dataset, aux, sub2sup)

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        admissions = dataset.tables.admissions
        c_subject_id = dataset.config.tables.admissions.subject_id_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias

        # Step 1: Collect overlapping admissions
        # Map from sub-admissions to the new super-admissions.
        sub2sup = {adm_id: super_adm_id for _, subject_adms in admissions.groupby(c_subject_id)
                   for adm_id, super_adm_id in self._collect_overlaps(subject_adms, c_admittime, c_dischtime).items()}

        # Step 2: Apply action.
        if self.merge:
            # Step 3: Extend discharge time of super admissions, remove sub-admissions,
            # and update admission ids in other tables.
            return self._merge_overlapping_admissions(dataset, aux, sub2sup)
        else:
            # Step 3: Collect subjects with at least one overlapping admission and remove them entirely.
            subject_ids = admissions.loc[sub2sup.keys(), c_subject_id].unique()
            static = dataset.tables.static
            n1 = len(static)
            static = static.drop(subject_ids, axis='index')
            n2 = len(static)
            self.report(aux, table='static', column=c_subject_id, value_type='count',
                        operation='filter_problematic_subjects',
                        before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
            # Step 4: synchronize subjects
            return self.synchronize_subjects(dataset, aux)


class FilterClampTimestampsToAdmissionInterval(DatasetTransformation):

    def _filter_timestamped_tables(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        timestamped_tables_conf = dataset.config.tables.timestamped_table_config_dict
        timestamped_tables = {name: getattr(dataset.tables, name) for name in
                              timestamped_tables_conf.keys()}
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        admissions = dataset.tables.admissions[[c_admittime, c_dischtime]]

        for name, table in timestamped_tables.items():
            c_time = timestamped_tables_conf[name].time_alias
            df = pd.merge(table, admissions, how='left', left_on=c_admission_id, right_index=True)
            index = df[df[c_time].between(df[c_admittime], df[c_dischtime])].index
            n1 = len(table)
            table = table.loc[index]
            n2 = len(table)
            self.report(aux, table=name, column=c_time, value_type='count', operation='filter',
                        before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, table)

        return dataset, aux

    def _filter_interval_based_tables(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        interval_based_tables_conf = dataset.config.tables.interval_based_table_config_dict
        interval_based_tables = {name: getattr(dataset.tables, name) for name in
                                 interval_based_tables_conf.keys()}
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        admissions = dataset.tables.admissions[[c_admittime, c_dischtime]]

        for name, table in interval_based_tables.items():
            c_start_time = interval_based_tables_conf[name].start_time_alias
            c_end_time = interval_based_tables_conf[name].end_time_alias
            df = pd.merge(table, admissions, how='left', left_on=c_admission_id, right_index=True)
            index = df[df[c_start_time].between(df[c_admittime], df[c_dischtime]) |
                       df[c_end_time].between(df[c_admittime], df[c_dischtime])].index
            # Step 1: Filter out intervals that are entirely outside admission interval.
            n1 = len(df)
            df = df.loc[index]
            n2 = len(df)
            self.report(aux, table=name, column=(c_start_time, c_end_time),
                        value_type='count', operation='filter',
                        before=n1, after=n2)

            # Step 2: Clamp intervals to admission interval.
            n_to_clamp = (df[c_start_time] < df[c_admittime] | df[c_end_time] > df[c_dischtime]).sum()
            self.report(aux, table=name, column=(c_start_time, c_end_time),
                        value_type='count', operation='clamp',
                        before=None, after=n_to_clamp)
            df[c_start_time] = df[c_start_time].clip(lower=df[c_admittime], upper=df[c_dischtime])
            df[c_end_time] = df[c_end_time].clip(lower=df[c_admittime], upper=df[c_dischtime])
            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, df)

        return dataset, aux

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        dataset, aux = self._filter_timestamped_tables(dataset, aux)
        return self._filter_interval_based_tables(dataset, aux)


class FilterInvalidInputRatesSubjects(DatasetTransformation):

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        c_normalized_amount_per_hour = dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_subject_id = dataset.config.tables.admissions.subject_id_alias

        icu_inputs = dataset.tables.icu_inputs
        static = dataset.tables.static
        admissions = dataset.tables.admissions

        nan_input_rates = icu_inputs[icu_inputs[c_normalized_amount_per_hour].isnull()]
        n_nan_inputs = len(nan_input_rates)
        nan_adm_ids = nan_input_rates[c_admission_id].unique()
        n_nan_adms = len(nan_adm_ids)

        nan_subject_ids = admissions[admissions.index.isin(nan_adm_ids)][c_subject_id].unique()
        n_nan_subjects = len(nan_subject_ids)

        self.report(aux, table=('icu_inputs', 'admissions', 'static'),
                    column=(c_normalized_amount_per_hour, c_admission_id, c_subject_id),
                    value_type='nan_counts',
                    before=(n_nan_inputs, n_nan_adms, n_nan_subjects),
                    after=None,
                    operation='filter_invalid_input_rates_subjects')

        n1 = len(static)
        static = static[~static[c_subject_id].isin(nan_subject_ids)]
        n2 = len(static)
        self.report(aux, table='static', column=c_subject_id, value_type='count',
                    before=n1, after=n2,
                    operation='filter_invalid_input_rates_subjects')

        return self.synchronize_subjects(dataset, aux)


class ICUInputRateUnitConversion(DatasetTransformation):
    conversion_table: pd.DataFrame

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        c_code = dataset.config.tables.icu_inputs.code_alias
        c_amount = dataset.config.tables.icu_inputs.amount_alias
        c_start_time = dataset.config.tables.icu_inputs.start_time_alias
        c_end_time = dataset.config.tables.icu_inputs.end_time_alias
        c_amount_unit = dataset.config.tables.icu_inputs.amount_unit_alias
        c_amount_per_hour = dataset.config.tables.icu_inputs.derived_amount_per_hour
        c_normalized_amount_per_hour = dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
        c_universal_unit = dataset.config.tables.icu_inputs.derived_universal_unit
        c_normalization_factor = dataset.config.tables.icu_inputs.derived_unit_normalization_factor
        icu_inputs = dataset.tables.icu_inputs
        assert (c in icu_inputs.columns for c in [c_code, c_amount, c_amount_unit]), \
            f"Some columns in: {c_code}, {c_amount}, {c_amount_unit}, not found in icu_inputs table"
        assert c_amount_per_hour not in icu_inputs.columns and c_normalized_amount_per_hour not in icu_inputs.columns, \
            f"Column {c_amount_per_hour} or {c_normalized_amount_per_hour} already exists in icu_inputs table"
        assert (c in self.conversion_table for c in [c_code, c_amount_unit, c_universal_unit,
                                                     c_normalization_factor]), \
            f"Some columns in: {', '.join([c_code, c_amount_unit, c_universal_unit, c_normalization_factor])}, not " \
            "found in the conversion table"

        df = pd.merge(icu_inputs, self.conversion_table, how='left',
                      on=[c_code, c_amount_unit])
        delta_hours = ((df[c_end_time] - df[c_start_time]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER)
        df[c_amount_per_hour] = df[c_amount] / delta_hours
        df[c_normalized_amount_per_hour] = df[c_amount_per_hour] * df[c_normalization_factor]
        df = df[icu_inputs.columns + [c_amount_per_hour, c_normalized_amount_per_hour,
                                      c_universal_unit, c_normalization_factor]]
        dataset = eqx.tree_at(lambda x: x.tables.icu_inputs, dataset, df)
        self.report(aux, table='icu_inputs', column=None,
                    value_type='columns', operation='new_columns',
                    before=icu_inputs.columns, after=df.columns)

        return dataset, aux


class OutlierRemovel(DatasetTransformation):
    outlier_q1: float = 0.25
    outlier_q2: float = 0.75
    outlier_iqr_scale: float = 1.5
    outlier_z1: float = -2.5
    outlier_z2: float = 2.5
    table_name: str = None

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        pass


class ObsScaler(DatasetTransformation):
    scaler: str = 'adaptive'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        pass


class InputScaler(DatasetTransformation):
    scaler: str = 'minmax'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, Any]]:
        pass
