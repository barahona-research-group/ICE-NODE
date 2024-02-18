from __future__ import annotations

import logging
import random
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from dataclasses import field
from typing import Dict, Optional, Tuple, List, Any, Callable, Union, ClassVar, Type

import equinox as eqx
import numpy as np
import pandas as pd

from .dataset import TimestampedTableConfig, IntervalBasedTableConfig, Dataset, AdmissionLinkedTableConfig, \
    AbstractDatasetTransformation, AbstractDatasetPipelineConfig, AbstractDatasetPipeline
from ..base import Config, Module

SECONDS_TO_HOURS_SCALER: float = 1 / 3600.0  # convert seconds to hours


class ReportAttributes(Config):
    transformation: str = None
    operation: str = None
    table: str = None
    column: str = None
    value_type: str = None
    before: Any = None
    after: Any = None
    additional_parameters: Dict[str, Any] = None


class DatasetTransformation(AbstractDatasetTransformation):
    name: str = None
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = tuple()
    blockers: ClassVar[Tuple[Type[DatasetTransformation], ...]] = tuple()

    @property
    def additional_parameters(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != 'name' and not k.startswith('_')}

    def report(self, aux: Dict[str, Any], **kwargs) -> None:
        if aux.get('report') is None:
            aux['report'] = []
        additional_params_str = ', '.join([f"{k}={v}" for k, v in self.additional_parameters.items()])
        aux['report'].append(ReportAttributes(transformation=self.name or type(self).__name__,
                                              additional_parameters=additional_params_str,
                                              **kwargs))

    def reporter(self) -> Callable:
        return self.report

    @staticmethod
    def synchronize_index(dataset: Dataset, indexed_table_name: str,
                          index_name: str, aux: Dict[str, Any],
                          report: Callable) -> Tuple[Dataset, Dict[str, Any]]:
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
            report(aux, table=table_name, column=index_name, before=n1, after=n2, value_type='count',
                   operation='sync_index')
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), aux

    @staticmethod
    def filter_no_admission_subjects(dataset: Dataset, aux: Dict[str, Any],
                                     report: Callable) -> Tuple[Dataset, Dict[str, Any]]:
        static = dataset.tables.static
        admissions = dataset.tables.admissions
        c_subject = dataset.config.tables.static.subject_id_alias
        no_admission_subjects = static[~static.index.isin(admissions[c_subject].unique())].index
        n1 = len(static)
        static = static.drop(no_admission_subjects, axis='index')
        n2 = len(static)
        report(aux, table='static', column=c_subject, before=n1, after=n2, value_type='count',
               operation='filter_no_admission_subjects')
        return eqx.tree_at(lambda x: x.tables.static, dataset, static), aux

    @classmethod
    def synchronize_admissions(cls, dataset: Dataset, aux: Dict[str, Any],
                               reporter: Callable) -> Tuple[Dataset, Dict[str, Any]]:
        dataset, aux = cls.synchronize_index(dataset, 'admissions',
                                             dataset.config.tables.admissions.admission_id_alias, aux,
                                             reporter)
        return cls.filter_no_admission_subjects(dataset, aux, reporter)

    @classmethod
    def synchronize_subjects(cls, dataset: Dataset, aux: Dict[str, Any],
                             reporter: Callable) -> Tuple[Dataset, Dict[str, Any]]:
        # Synchronizing subjects might entail synchronizing admissions, so we need to call it first
        dataset, aux = cls.synchronize_index(dataset, 'static',
                                             dataset.config.tables.static.subject_id_alias, aux, reporter)
        return cls.synchronize_admissions(dataset, aux, reporter)

    @abstractmethod
    def __call__(self, dataset: Dataset, auxiliary) -> Tuple[Dataset, Dict[str, Any]]:
        pass


class DatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlapping_admissions: str = 'merge'
    sample: Optional[int] = None
    offset: Optional[int] = 0


class DatasetPipeline(AbstractDatasetPipeline):
    transformations: List[DatasetTransformation]

    def __init__(self, transformations: List[DatasetTransformation]):
        super().__init__(config=Config())
        self.transformations = transformations
        self._validate_transformation_sequence()

    def _validate_transformation_sequence(self):
        applied_set = set()
        for t in self.transformations:
            if t.name in applied_set:
                raise ValueError(f"Transformation {t.name} applied more than once.")
            applied_set.add(t.name)
            for d in t.dependencies:
                assert any(isinstance(x, d) for x in applied_set), \
                    f"Transformation {t.name} depends on {d.__name__} which was not applied before."
            for b in t.blockers:
                assert all(not isinstance(x, b) for x in applied_set), \
                    f"Transformation {t.name} is blocked by {b.__name__} which was applied before."

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


class SampleSubjects(DatasetTransformation):
    n_subjects: int = field(kw_only=True)
    seed: Optional[int] = None
    offset: int = 0
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetIndex,)

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        static = dataset.tables.static
        # assert index name is subject_id
        c_subject_id = dataset.config.tables.static.subject_id_alias
        assert c_subject_id in static.index.names, f'Index name must be {c_subject_id}'

        rng = random.Random(self.seed)
        subjects = static.index.unique().tolist()
        rng.shuffle(subjects)
        subjects = subjects[self.offset:self.offset + self.n_subjects]
        n1 = len(static)
        static = static.loc[subjects]
        n2 = len(static)
        self.report(aux, table='static', column='index', before=n1, after=n2, value_type='count',
                    operation='sample')
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, aux, self.reporter())


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


class SetAdmissionRelativeTimes(DatasetTransformation):
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (CastTimestamps, SetIndex)

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
                df = df.assign(
                    **{time_col: (df[time_col] - df[c_admittime]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER})

                self.report(aux, table=table_name, column=time_col, before=table[time_col].dtype,
                            after=df[time_col].dtype,
                            value_type='dtype', operation='set_admission_relative_times')

            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, df)

        return dataset, aux


class FilterSubjectsNegativeAdmissionLengths(DatasetTransformation):
    name: str = 'FilterSubjectsNegativeAdmissionLengths'
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (CastTimestamps, SetIndex)

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
        return self.synchronize_subjects(dataset, aux, self.reporter())


class SetCodeIntegerIndices(DatasetTransformation):
    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column in dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(dataset.scheme, table_name)
            dtype1 = table[code_column].dtype
            n1 = len(table)
            table = table.assign(code_column=table[code_column].map(coding_scheme.index))
            table = table[table[code_column].notnull()].astype({code_column: int})
            dtype2 = table[code_column].dtype
            n2 = len(table)
            self.report(aux, table=table_name, column=code_column, before=n1, after=n2, value_type='count',
                        operation='filter_unsupported_codes')
            self.report(aux, table=table_name, column=code_column, before=dtype1, after=dtype2, value_type='dtype',
                        operation='code_integer_index')

            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, aux


class FilterUnsupportedCodes(DatasetTransformation):
    blockers: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetCodeIntegerIndices,)

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


class ProcessOverlappingAdmissions(DatasetTransformation):
    merge: bool = field(
        kw_only=True)  # if True, merge overlapping admissions. Otherwise, remove overlapping admissions.

    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetIndex, CastTimestamps,)

    @staticmethod
    def map_admission_ids(dataset: Dataset, aux: Dict[str, Any], sub2sup: Dict[str, str],
                          reporter: Callable) -> Tuple[Dataset, Dict[str, Any]]:
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
            table[c_admission_id] = table[c_admission_id].map(lambda i: sub2sup.get(i, i))
            n2 = table[c_admission_id].nunique()
            reporter(aux, table=table_name, column=c_admission_id, before=n1, after=n2, value_type='nunique',
                     operation='map_admission_id')
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), aux

    @staticmethod
    def _collect_overlaps(subject_admissions, c_admittime, c_dischtime):
        """
        Collect overlapping admissions for a subject.
        Tested in test.unit.ehr.test_pipeline.TestProcessOverlappingAdmissions.test_overlapping_cases
        """
        # Sort by admission time.
        subject_admissions = subject_admissions.sort_values(c_admittime)

        # Previous discharge time.
        index = subject_admissions.index
        subject_admissions.loc[index[1:], 'prev_dischtime'] = subject_admissions.loc[index[:-1], c_dischtime].values
        # Cumulative-max of previous discharge time.
        subject_admissions['prev_dischtime_cummax'] = subject_admissions['prev_dischtime'].cummax()

        # Get corresponding index of the maximum discharge time up to the current admission.
        lambda_fn = lambda x: subject_admissions[subject_admissions[c_dischtime] == x].first_valid_index()
        subject_admissions['prev_dischtime_cummax_idx'] = subject_admissions['prev_dischtime_cummax'].map(lambda_fn)

        # Drop admissions with admittime after the prev_max discharge time. No overlaps with preceding admissions.
        # Note: this line needs to come after adding 'prev_dischtime_cummax_idx' column.
        subject_admissions = subject_admissions[
            subject_admissions[c_admittime] <= subject_admissions['prev_dischtime_cummax']]
        subject_admissions = subject_admissions[subject_admissions['prev_dischtime_cummax_idx'].notnull()]

        # New admissions mappings.
        child2parent = subject_admissions['prev_dischtime_cummax_idx'].to_dict()
        # Recursively map parents to further ancestors until the root admission.
        while len(set(child2parent.values()).intersection(child2parent.keys())) > 0:
            child2parent = {k: child2parent.get(v, v) for k, v in child2parent.items()}

        return child2parent

    @classmethod
    def _merge_overlapping_admissions(cls,
                                      dataset: Dataset, aux: Dict[str, Any],
                                      sub2sup: Dict[str, str], reporter: Callable) -> Tuple[Dataset, Dict[str, Any]]:
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
        reporter(aux, table='admissions', column=c_admission_id, value_type='count',
                 operation='merge_overlapping_admissions',
                 before=n1, after=n2)

        # Step 4: update admission ids in other tables.
        return cls.map_admission_ids(dataset, aux, sub2sup, reporter)

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
            return self._merge_overlapping_admissions(dataset, aux, sub2sup, self.reporter())
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
            return self.synchronize_subjects(dataset, aux, self.reporter())


class FilterClampTimestampsToAdmissionInterval(DatasetTransformation):
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetIndex, CastTimestamps,)
    blockers: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetAdmissionRelativeTimes,)

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
        interval_based_tables: Dict[str, pd.DataFrame] = {name: getattr(dataset.tables, name) for name in
                                                          interval_based_tables_conf.keys()}
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        admissions = dataset.tables.admissions[[c_admittime, c_dischtime]]

        for name, table in interval_based_tables.items():
            c_start_time = interval_based_tables_conf[name].start_time_alias
            c_end_time = interval_based_tables_conf[name].end_time_alias
            df = pd.merge(table, admissions, how='left', left_on=c_admission_id, right_index=True)
            # Step 1: Filter out intervals that are entirely outside admission interval.
            index = df[df[c_start_time].between(df[c_admittime], df[c_dischtime]) |
                       df[c_end_time].between(df[c_admittime], df[c_dischtime])].index
            n1 = len(df)
            df = df.loc[index]
            n2 = len(df)
            self.report(aux, table=name, column=(c_start_time, c_end_time),
                        value_type='count', operation='filter',
                        before=n1, after=n2)

            # Step 2: Clamp intervals to admission interval if either side is outside.
            n_to_clamp = np.sum((df[c_start_time] < df[c_admittime]) | (df[c_end_time] > df[c_dischtime]))
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


class ICUInputRateUnitConversion(DatasetTransformation):
    conversion_table: pd.DataFrame = field(kw_only=True)
    blockers: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetCodeIntegerIndices,)

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


class FilterInvalidInputRatesSubjects(DatasetTransformation):
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetIndex, ICUInputRateUnitConversion)

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
        static = static[~static.index.isin(nan_subject_ids)]
        n2 = len(static)
        self.report(aux, table='static', column=c_subject_id, value_type='count',
                    before=n1, after=n2,
                    operation='filter_invalid_input_rates_subjects')
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, aux, self.reporter())


class RandomSplits(DatasetTransformation):
    splits: List[float] = field(kw_only=True)
    splits_key: str = field(kw_only=True)
    seed: Optional[int] = None
    balance: str = 'subjects'
    discount_first_admission: bool = False

    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (SetIndex, CastTimestamps)

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        aux[self.splits_key] = dataset.random_splits(splits=self.splits,
                                                     random_seed=self.seed,
                                                     balance=self.balance,
                                                     discount_first_admission=self.discount_first_admission)
        self.report(aux, table='static', column=None, value_type='splits',
                    operation=f'aux["{self.splits_key}"]<-dataset.random_splits',
                    before=(len(dataset.tables.static),),
                    after=tuple(len(x) for x in aux[self.splits_key]))
        return dataset, aux


class TrainableTransformation(DatasetTransformation, metaclass=ABCMeta):
    splits_key: str = field(kw_only=True)
    training_split_index: Union[int, List[str]] = 0
    fitted_processor: str = 'obs_outlier_remover'
    fit_only: bool = False
    transformer_key: str = ''  # to be retrieved from aux.
    dependencies: ClassVar[Tuple[Type[DatasetTransformation], ...]] = (RandomSplits, SetIndex)

    def get_training_split(self, aux: Dict[str, Any]) -> List[str]:
        assert self.splits_key in aux, "Training subject ids cannot be retrieved." \
                                       "{self.splits_key} not found in aux."
        return aux[self.splits_key][self.training_split_index]

    def get_admission_ids(self, dataset: Dataset, aux: Dict[str, Any]) -> List[str]:
        c_subject_id = dataset.config.tables.static.subject_id_alias
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        admissions = dataset.tables.admissions[[c_subject_id]]
        assert c_admission_id in admissions.index.names, f"Column {c_admission_id} not found in admissions table index."
        training_subject_ids = self.get_training_split(aux)
        return admissions[admissions[c_subject_id].isin(training_subject_ids)].index.unique()


class ObsIQROutlierRemover(TrainableTransformation):
    outlier_q1: float = 0.25
    outlier_q2: float = 0.75
    outlier_iqr_scale: float = 1.5
    outlier_z1: float = -2.5
    outlier_z2: float = 2.5
    transformer_key: str = 'obs_outlier_remover'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        remover = IQROutlierRemover(table=lambda x: x.config.tables.obs,
                                    code_column=lambda x: x.config.tables.obs.code_alias,
                                    value_column=lambda x: x.config.tables.obs.value_alias,
                                    outlier_q1=self.outlier_q1,
                                    outlier_q2=self.outlier_q2,
                                    outlier_iqr_scale=self.outlier_iqr_scale,
                                    outlier_z1=self.outlier_z1,
                                    outlier_z2=self.outlier_z2).fit(dataset, self.get_admission_ids(dataset, aux))
        aux[self.transformer_key] = remover

        if self.fit_only:
            return dataset, aux

        n1 = len(dataset.tables.obs)
        # TODO: report specific removals stats for each code.
        dataset = remover(dataset)
        n2 = len(dataset.tables.obs)
        self.report(aux, table='obs', column=None, value_type='count',
                    operation='filter', before=n1, after=n2)
        return dataset, aux


class ObsAdaptiveScaler(TrainableTransformation):
    transformer_key = 'obs_scaler'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        scaler = AdaptiveScaler(table=lambda x: x.config.tables.obs,
                                code_column=lambda x: x.config.tables.obs.code_alias,
                                value_column=lambda x: x.config.tables.obs.value_alias).fit(dataset,
                                                                                            self.get_admission_ids(
                                                                                                dataset, aux))

        aux[self.transformer_key] = scaler

        if self.fit_only:
            return dataset, aux

        dtype1 = dataset.tables.obs[dataset.config.tables.obs.value_alias].dtype
        dataset = scaler(dataset)
        dtype2 = dataset.tables.obs[dataset.config.tables.obs.value_alias].dtype
        self.report(aux, table='obs', column=dataset.config.tables.obs.value_alias,
                    value_type='dtype',
                    operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                    before=dtype1, after=dtype2)
        return dataset, aux


class InputScaler(TrainableTransformation):
    transformer_key: str = 'icu_inputs_scaler'

    def __call__(self, dataset: Dataset, aux: Dict[str, Any]) -> Tuple[Dataset, Dict[str, str]]:
        code_column = lambda x: x.config.tables.icu_inputs.code_alias
        value_column = lambda x: x.config.tables.icu_inputs.derived_normalized_amount_per_hour
        scaler = MaxScaler(table=lambda x: x.config.tables.icu_inputs,
                           code_column=code_column,
                           value_column=value_column).fit(dataset, self.get_admission_ids(dataset, aux))
        aux[self.transformer_key] = scaler

        if self.fit_only:
            return dataset, aux

        dtype1 = dataset.tables.icu_inputs[value_column(dataset)].dtype
        dataset = scaler(dataset)
        dtype2 = dataset.tables.icu_inputs[value_column(dataset)].dtype
        self.report(aux, table='icu_inputs', column=dataset.config.tables.obs.value_alias,
                    value_type='dtype',
                    operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                    before=dtype1, after=dtype2)
        return dataset, aux


class CodedValueProcessor(Module):
    code_column: Callable[[Dataset], str]
    value_column: Callable[[Dataset], str]
    table: Callable[[Dataset], pd.DataFrame]

    def fit(self, dataset: Dataset, admission_ids: List[str]) -> 'CodedValueProcessor':
        df = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)
        c_adm_id = dataset.config.tables.obs.admission_id_alias
        df = df[[c_code, c_value, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]

        fitted = self
        for k, v in self._extract_stats(df, c_code, c_value).items():
            fitted = eqx.tree_at(lambda x: getattr(x, k), fitted, v)
        return fitted

    @abstractmethod
    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        pass

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        pass


class CodedValueScaler(CodedValueProcessor):
    use_float16: bool  # = field(default=True, init=False)

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


class ZScoreScaler(CodedValueScaler):
    mean: pd.Series
    std: pd.Series

    @property
    def original_dtype(self) -> np.dtype:
        return self.mean.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        mean = table[c_code].map(self.mean)
        std = table[c_code].map(self.std)
        table.loc[:, c_value] = (table[c_value] - mean) / std
        if self.use_float16:
            table = table.astype({c_value: np.float16})

        return eqx.tree_at(lambda x: self.table(dataset), dataset, table)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        return array * self.std.loc[index].values + self.mean.loc[index].values

    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        array = array.astype(self.original_dtype)
        return array * self.std.loc[code_index] + self.mean.loc[code_index]

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std()
            }))
        return dict(mean=stat['mu'], std=stat['sigma'])


class MaxScaler(CodedValueScaler):
    max_val: pd.Series

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        max_val = df[c_code].map(self.max_val)
        df.loc[:, c_value] = (df[c_value] / max_val)
        if self.use_float16:
            df = df.astype({c_value: np.float16})
        return eqx.tree_at(self.table, dataset, df)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        if array.shape[-1] == len(self.max_val):
            index = np.arange(array.shape[-1])
            return array * self.max_val.loc[index].values
        index = self.max_val.index.values
        array = array.copy()
        if array.ndim == 1:
            array[index] *= self.max_val.values
        else:
            array[:, index] *= self.max_val.values
        return array

    def unscale_code(self, array: np.ndarray, code_index: int) -> np.ndarray:
        array = array.astype(self.original_dtype)
        return array * self.max_val.loc[code_index]

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'min': x[c_value].min(),
                'max': x[c_value].max()
            }))
        return dict(min_val=stat['min'], max_val=stat['max'])


class AdaptiveScaler(CodedValueScaler):
    max_val: pd.Series
    min_val: pd.Series
    mean: pd.Series
    std: pd.Series

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        min_val = df[c_code].map(self.min_val)
        max_val = df[c_code].map(self.max_val)
        mean = df[c_code].map(self.mean)
        std = df[c_code].map(self.std)

        minmax_scaled = (df[c_value] - min_val) / max_val
        z_scaled = ((df[c_value] - mean) / std)

        df.loc[:, c_value] = np.where(min_val >= 0.0, minmax_scaled, z_scaled)
        if self.use_float16:
            df = df.astype({c_value: np.float16})
        return eqx.tree_at(self.table, dataset, df)

    def unscale(self, array: np.ndarray) -> np.ndarray:
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        mu = self.mean.loc[index].values
        sigma = self.std.loc[index].values
        min_val = self.min_val.loc[index].values
        max_val = self.max_val.loc[index].values
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def unscale_code(self, array: np.ndarray, code_index: str) -> np.ndarray:
        array = array.astype(self.original_dtype)
        mu = self.mean.loc[code_index]
        sigma = self.std.loc[code_index]
        min_val = self.min_val.loc[code_index]
        max_val = self.max_val.loc[code_index]
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std(),
                'min': x[c_value].min(),
                'max': x[c_value].max()
            }))
        return dict(mean=stat['mu'],
                    std=stat['sigma'],
                    min_val=stat['min'],
                    max_val=stat['max'])


class IQROutlierRemover(CodedValueProcessor):
    outlier_q1: float
    outlier_q2: float
    outlier_iqr_scale: float
    outlier_z1: float
    outlier_z2: float
    min_val: pd.Series
    max_val: pd.Series

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table(dataset)
        c_value = self.value_column(dataset)
        c_code = self.code_column(dataset)

        min_val = table[c_code].map(self.min_val)
        max_val = table[c_code].map(self.max_val)
        table = table[table[c_value].between(min_val, max_val)]

        return eqx.tree_at(self.table, dataset, table)

    def _extract_stats(self, df: pd.DataFrame, c_code: str, c_value: str) -> Dict[str, pd.Series]:
        outlier_q = np.array([self.outlier_q1, self.outlier_q2])
        q = df.groupby(c_code).apply(lambda x: x[c_value].quantile(outlier_q))

        q.columns = ['q1', 'q2']
        q['iqr'] = q['q2'] - q['q1']
        q['out_q1'] = q['q1'] - self.outlier_iqr_scale * q['iqr']
        q['out_q2'] = q['q2'] + self.outlier_iqr_scale * q['iqr']

        stat = df.groupby(c_code).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std()
            }))

        stat['out_z1'] = stat['mu'] - self.outlier_z1 * stat['sigma']
        stat['out_z2'] = stat['mu'] + self.outlier_z2 * stat['sigma']
        return dict(min_val=np.minimum(q['out_q1'], stat['out_z1']),
                    max_val=np.maximum(q['out_q2'], stat['out_z2']))
