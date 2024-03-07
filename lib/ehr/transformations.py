from __future__ import annotations

import logging
import random
from abc import abstractmethod
from collections import defaultdict
from dataclasses import field
from typing import Dict, Optional, Tuple, List, Any, Callable, ClassVar, Type, Final, Set

import equinox as eqx
import numpy as np
import pandas as pd

from . import TVxEHR
from .dataset import Dataset, AbstractDatasetTransformation, AbstractDatasetPipelineConfig, AbstractDatasetPipeline, \
    ReportAttributes, TransformationsDependency
from .tvx_ehr import TVxReportAttributes, TrainableTransformation, AbstractTVxTransformation

SECONDS_TO_HOURS_SCALER: float = 1 / 3600.0  # convert seconds to hours


class DatasetTransformation(AbstractDatasetTransformation):

    @staticmethod
    def synchronize_index(dataset: Dataset, indexed_table_name: str,
                          index_name: str, report: Tuple[ReportAttributes, ...],
                          reporter: Callable) -> Tuple[Dataset, Tuple[ReportAttributes, ...]]:
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
            report = reporter(report, table=table_name, column=index_name, before=n1, after=n2, value_type='count',
                              operation='sync_index')
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), report

    @staticmethod
    def filter_no_admission_subjects(dataset: Dataset, report: Tuple[ReportAttributes, ...],
                                     reporter: Callable) -> Tuple[Dataset, Tuple[ReportAttributes, ...]]:
        static = dataset.tables.static
        admissions = dataset.tables.admissions
        c_subject = dataset.config.tables.static.subject_id_alias
        no_admission_subjects = static[~static.index.isin(admissions[c_subject].unique())].index
        n1 = len(static)
        static = static.drop(no_admission_subjects, axis='index')
        n2 = len(static)
        report = reporter(report, table='static', column=c_subject, before=n1, after=n2, value_type='count',
                          operation='filter_no_admission_subjects')
        return eqx.tree_at(lambda x: x.tables.static, dataset, static), report

    @classmethod
    def synchronize_admissions(cls, dataset: Dataset, report: Tuple[ReportAttributes, ...],
                               reporter: Callable) -> Tuple[Dataset, Tuple[ReportAttributes, ...]]:
        dataset, report = cls.synchronize_index(dataset, 'admissions',
                                                dataset.config.tables.admissions.admission_id_alias, report,
                                                reporter)
        return cls.filter_no_admission_subjects(dataset, report, reporter)

    @classmethod
    def synchronize_subjects(cls, dataset: Dataset, report: Tuple[ReportAttributes, ...],
                             reporter: Callable) -> Tuple[Dataset, Tuple[ReportAttributes, ...]]:
        # Synchronizing subjects might entail synchronizing admissions, so we need to call it first
        dataset, report = cls.synchronize_index(dataset, 'static',
                                                dataset.config.tables.static.subject_id_alias, report, reporter)
        return cls.synchronize_admissions(dataset, report, reporter)

    @abstractmethod
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        pass


class DatasetPipelineConfig(AbstractDatasetPipelineConfig):
    overlapping_admissions: str = 'merge'
    sample: Optional[int] = None
    offset: Optional[int] = 0


class ValidatedDatasetPipeline(AbstractDatasetPipeline):
    config: DatasetPipelineConfig
    transformations: List[DatasetTransformation]
    validator: ClassVar[TransformationsDependency] = field(default_factory=lambda: PIPELINE_VALIDATOR)


class SetIndex(DatasetTransformation):
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        tables_dict = dataset.tables.tables_dict
        for indexed_table_name, index_name in dataset.config.tables.indices.items():
            table = tables_dict[indexed_table_name]
            index1 = table.index.name
            table = table.set_index(index_name)
            index2 = table.index.name
            report = self.report(report, table=indexed_table_name, column=index_name, before=index1, after=index2,
                                 value_type='index_name',
                                 operation='set_index')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, indexed_table_name), dataset, table)
        return dataset, report


class CastTimestamps(DatasetTransformation):
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
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
                report = self.report(report, table=table_name, column=time_col, before=dtype1, after=dtype2,
                                     value_type='dtype',
                                     operation='cast')

            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)
        return eqx.tree_at(lambda x: x.tables, dataset, tables), report


class SetAdmissionRelativeTimes(DatasetTransformation):
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        time_cols = {k: v for k, v in dataset.config.tables.time_cols.items()
                     if dataset.config.tables.temporal_admission_linked_table(k)}

        c_admittime = dataset.config.tables.admissions.admission_time_alias
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        admissions = dataset.tables.admissions[[c_admittime]]
        tables_dict = dataset.tables.tables_dict

        for table_name, time_cols in time_cols.items():
            table = tables_dict[table_name]
            df = pd.merge(table, admissions,
                          left_on=c_admission_id,
                          right_index=True,
                          suffixes=(None, '_admissions'),
                          how='left')
            for time_col in time_cols:
                df = df.assign(
                    **{time_col: (df[time_col] - df[c_admittime]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER})

                report = self.report(report, table=table_name, column=time_col, before=table[time_col].dtype,
                                     after=df[time_col].dtype,
                                     value_type='dtype', operation='set_admission_relative_times')

            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, df)

        return dataset, report


class FilterSubjectsNegativeAdmissionLengths(DatasetTransformation):

    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        c_subject = dataset.config.tables.static.subject_id_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias
        admissions = dataset.tables.admissions

        # assert dtypes are datetime64[ns]
        assert admissions[c_admittime].dtype == 'datetime64[ns]' and \
               admissions[c_dischtime].dtype == 'datetime64[ns]', \
            f'{c_admittime} and {c_dischtime} must be datetime64[ns]'

        static = dataset.tables.static
        neg_los_subjects = admissions[admissions[c_dischtime] < admissions[c_admittime]][c_subject].unique()
        n_before = len(static)
        static = static[~static.index.isin(neg_los_subjects)]
        n_after = len(static)
        report = self.report(report, table='static', column=c_subject, value_type='count', operation='filter',
                             before=n_before, after=n_after)
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, report, self.reporter())


class SetCodeIntegerIndices(DatasetTransformation):
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column in dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(dataset.scheme, table_name)
            dtype1 = table[code_column].dtype
            n1 = len(table)
            table = table.assign(**{code_column: table[code_column].map(coding_scheme.index)})
            table = table[table[code_column].notnull()].astype({code_column: int})
            dtype2 = table[code_column].dtype
            n2 = len(table)
            report = self.report(report, table=table_name, column=code_column, before=n1, after=n2, value_type='count',
                                 operation='filter_unsupported_codes')
            report = self.report(report, table=table_name, column=code_column, before=dtype1, after=dtype2,
                                 value_type='dtype',
                                 operation='code_integer_index')

            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, report


class FilterUnsupportedCodes(DatasetTransformation):
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column in dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(dataset.scheme, table_name)
            n1 = len(table)
            table = table[table[code_column].isin(coding_scheme.codes)]
            n2 = len(table)
            report = self.report(report, table=table_name, column=code_column, before=n1, after=n2, value_type='count',
                                 operation='filter')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, report


class ProcessOverlappingAdmissions(DatasetTransformation):

    @staticmethod
    def map_admission_ids(dataset: Dataset, report: Tuple[ReportAttributes, ...], sub2sup: Dict[str, Any],
                          reporter: Callable) -> Tuple[Dataset, Tuple[ReportAttributes, ...]]:
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
            table.loc[:, c_admission_id] = table.loc[:, c_admission_id].map(lambda i: sub2sup.get(i, i))
            n2 = table[c_admission_id].nunique()
            report = reporter(report, table=table_name, column=c_admission_id, before=n1, after=n2,
                              value_type='nunique',
                              operation='map_admission_id')
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), report

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
                                      dataset: Dataset, report: Tuple[ReportAttributes, ...],
                                      sub2sup: Dict[str, str], reporter: Callable) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
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
        report = reporter(report, table='admissions', column=c_admission_id, value_type='count',
                          operation='merge_overlapping_admissions',
                          before=n1, after=n2)

        # Step 4: update admission ids in other tables.
        return cls.map_admission_ids(dataset, report, sub2sup, reporter)

    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
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
            return self._merge_overlapping_admissions(dataset, report, sub2sup, self.reporter())
        else:
            # Step 3: Collect subjects with at least one overlapping admission and remove them entirely.
            subject_ids = admissions.loc[sub2sup.keys(), c_subject_id].unique()
            static = dataset.tables.static
            n1 = len(static)
            static = static.drop(subject_ids, axis='index')
            n2 = len(static)
            report = self.report(report, table='static', column=c_subject_id, value_type='count',
                                 operation='filter_problematic_subjects',
                                 before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
            # Step 4: synchronize subjects
            return self.synchronize_subjects(dataset, report, self.reporter())


class FilterClampTimestampsToAdmissionInterval(DatasetTransformation):

    def _filter_timestamped_tables(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        timestamped_tables_conf = dataset.config.tables.timestamped_table_config_dict
        timestamped_tables = {name: getattr(dataset.tables, name) for name in
                              timestamped_tables_conf.keys()}
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_dischtime = dataset.config.tables.admissions.discharge_time_alias
        c_admittime = dataset.config.tables.admissions.admission_time_alias
        admissions = dataset.tables.admissions[[c_admittime, c_dischtime]]

        for name, table in timestamped_tables.items():
            c_time = timestamped_tables_conf[name].time_alias
            df = pd.merge(table, admissions, how='left',
                          left_on=c_admission_id, right_index=True,
                          suffixes=(None, '_y'))
            index = df[df[c_time].between(df[c_admittime], df[c_dischtime])].index
            n1 = len(table)
            table = table.loc[index]
            n2 = len(table)
            report = self.report(report, table=name, column=c_time, value_type='count', operation='filter',
                                 before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, table)

        return dataset, report

    def _filter_interval_based_tables(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
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
            df = pd.merge(table, admissions, how='left',
                          left_on=c_admission_id, right_index=True,
                          suffixes=(None, '_y'))
            # Step 1: Filter out intervals that are entirely outside admission interval.
            index = df[df[c_start_time].between(df[c_admittime], df[c_dischtime]) |
                       df[c_end_time].between(df[c_admittime], df[c_dischtime])].index
            n1 = len(df)
            df = df.loc[index]
            n2 = len(df)
            report = self.report(report, table=name, column=(c_start_time, c_end_time),
                                 value_type='count', operation='filter',
                                 before=n1, after=n2)

            # Step 2: Clamp intervals to admission interval if either side is outside.
            n_to_clamp = np.sum((df[c_start_time] < df[c_admittime]) | (df[c_end_time] > df[c_dischtime]))
            report = self.report(report, table=name, column=(c_start_time, c_end_time),
                                 value_type='count', operation='clamp',
                                 before=None, after=n_to_clamp)
            df[c_start_time] = df[c_start_time].clip(lower=df[c_admittime], upper=df[c_dischtime])
            df[c_end_time] = df[c_end_time].clip(lower=df[c_admittime], upper=df[c_dischtime])
            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, df)

        return dataset, report

    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        dataset, report = self._filter_timestamped_tables(dataset, report)
        return self._filter_interval_based_tables(dataset, report)


class SelectSubjectsWithObservation(DatasetTransformation):

    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        c_code = dataset.config.tables.obs.code_alias
        c_admission_id = dataset.config.tables.obs.admission_id_alias
        c_subject = dataset.config.tables.static.subject_id_alias
        obs = dataset.tables.obs

        admission_ids = obs[obs[c_code] == self.code][c_admission_id].unique()
        assert len(admission_ids) > 0, f'No observations for code {self.code}'

        subjects = dataset.tables.admissions.loc[admission_ids, c_subject].unique()
        static = dataset.tables.static
        n1 = len(static)
        static = static[static.index.isin(subjects)]
        n2 = len(static)
        report = self.report(report, table='static', column=c_subject, value_type='count',
                             operation=f'select_subjects(has({self.code}))',
                             before=n1, after=n2)
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, report, self.reporter())


class ICUInputRateUnitConversion(DatasetTransformation):

    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        c_code = dataset.config.tables.icu_inputs.code_alias
        c_amount = dataset.config.tables.icu_inputs.amount_alias
        c_start_time = dataset.config.tables.icu_inputs.start_time_alias
        c_end_time = dataset.config.tables.icu_inputs.end_time_alias
        c_amount_unit = dataset.config.tables.icu_inputs.amount_unit_alias
        c_normalized_amount = dataset.config.tables.icu_inputs.derived_normalized_amount
        c_normalized_amount_per_hour = dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
        c_universal_unit = dataset.config.tables.icu_inputs.derived_universal_unit
        c_normalization_factor = dataset.config.tables.icu_inputs.derived_unit_normalization_factor
        icu_inputs = dataset.tables.icu_inputs

        _derived_columns = [c_normalized_amount, c_normalized_amount_per_hour, c_universal_unit, c_normalization_factor]
        assert (c in icu_inputs.columns for c in [c_code, c_amount, c_amount_unit]), \
            f"Some columns in: {c_code}, {c_amount}, {c_amount_unit}, not found in icu_inputs table"
        assert all(c not in icu_inputs.columns for c in _derived_columns), \
            f"Some of these columns [{', '.join(_derived_columns)}] already exists in icu_inputs table"
        assert (c in self.conversion_table for c in _derived_columns[2:]), \
            f"Some columns in: {', '.join(_derived_columns[2:])}, not " \
            "found in the conversion table"

        df = pd.merge(icu_inputs, self.conversion_table, how='left',
                      on=[c_code, c_amount_unit],
                      suffixes=(None, '_y'))

        delta_hours = ((df[c_end_time] - df[c_start_time]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER)
        df[c_normalized_amount] = df[c_amount] * df[c_normalization_factor]
        df[c_normalized_amount_per_hour] = df[c_normalized_amount] / delta_hours
        df = df[icu_inputs.columns.tolist() + _derived_columns]
        dataset = eqx.tree_at(lambda x: x.tables.icu_inputs, dataset, df)
        report = self.report(report, table='icu_inputs', column=None,
                             value_type='columns', operation='new_columns',
                             before=icu_inputs.columns.tolist(), after=df.columns.tolist())

        return dataset, report


class FilterInvalidInputRatesSubjects(DatasetTransformation):
    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
        c_rate = dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
        c_admission_id = dataset.config.tables.admissions.admission_id_alias
        c_subject_id = dataset.config.tables.admissions.subject_id_alias

        icu_inputs = dataset.tables.icu_inputs
        static = dataset.tables.static
        admissions = dataset.tables.admissions

        nan_input_rates = icu_inputs[icu_inputs[c_rate].isnull()]
        n_nan_inputs = len(nan_input_rates)
        nan_adm_ids = nan_input_rates[c_admission_id].unique()
        n_nan_adms = len(nan_adm_ids)

        nan_subject_ids = admissions[admissions.index.isin(nan_adm_ids)][c_subject_id].unique()
        n_nan_subjects = len(nan_subject_ids)

        report = self.report(report, table=('icu_inputs', 'admissions', 'static'),
                             column=(c_rate, c_admission_id, c_subject_id),
                             value_type='nan_counts',
                             before=(n_nan_inputs, n_nan_adms, n_nan_subjects),
                             after=None,
                             operation='filter_invalid_input_rates_subjects')

        n1 = len(static)
        static = static[~static.index.isin(nan_subject_ids)]
        n2 = len(static)
        report = self.report(report, table='static', column=c_subject_id, value_type='count',
                             before=n1, after=n2,
                             operation='filter_invalid_input_rates_subjects')
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, report, self.reporter())


class SampleSubjects(AbstractTVxTransformation):

    def __call__(self, dataset: Dataset, report: Tuple[ReportAttributes, ...]) -> Tuple[
        Dataset, Tuple[ReportAttributes, ...]]:
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
        report = self.report(report, table='static', column='index', before=n1, after=n2, value_type='count',
                             operation='sample')
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return self.synchronize_subjects(dataset, report, self.reporter())


class RandomSplits(AbstractTVxTransformation):

    def __call__(self, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        config = tv_ehr.config.splits
        splits = tv_ehr.dataset.random_splits(splits=config.split_quantiles,
                                              random_seed=config.seed,
                                              balance=config.balance,
                                              discount_first_admission=config.discount_first_admission)

        report = self.report(report, table='static', column=None, value_type='splits',
                             operation=f'TVxEHR.splits<-TVxEHR.dataset.random_splits(TVxEHR.config.splits)',
                             before=(len(tv_ehr.dataset.tables.static),),
                             after=tuple(len(x) for x in splits))
        tv_ehr = eqx.tree_at(lambda x: x.splits, tv_ehr, splits)
        return tv_ehr, report


class CodedValueProcessor(eqx.Module):
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
    use_float16: bool

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
    mean: pd.Series = field(default_factory=lambda: pd.Series())
    std: pd.Series = field(default_factory=lambda: pd.Series())

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
    max_val: pd.Series = field(default_factory=lambda: pd.Series())

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table(dataset).copy()
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
                'max': x[c_value].max()
            }))
        return dict(max_val=stat['max'])


class AdaptiveScaler(CodedValueScaler):
    max_val: pd.Series = field(default_factory=lambda: pd.Series())
    min_val: pd.Series = field(default_factory=lambda: pd.Series())
    mean: pd.Series = field(default_factory=lambda: pd.Series())
    std: pd.Series = field(default_factory=lambda: pd.Series())

    @property
    def original_dtype(self) -> np.dtype:
        return self.max_val.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        df = self.table(dataset).copy()
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
    min_val: pd.Series = field(default_factory=lambda: pd.Series())
    max_val: pd.Series = field(default_factory=lambda: pd.Series())

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


class ObsIQROutlierRemover(TrainableTransformation):
    def __call__(self, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        config = tv_ehr.config.numerical_processors.outlier_removers.obs
        remover = IQROutlierRemover(table=lambda x: x.dataset.tables.obs,
                                    code_column=lambda x: x.dataset.config.tables.obs.code_alias,
                                    value_column=lambda x: x.dataset.config.tables.obs.value_alias,
                                    outlier_q1=config.outlier_q1,
                                    outlier_q2=config.outlier_q2,
                                    outlier_iqr_scale=config.outlier_iqr_scale,
                                    outlier_z1=config.outlier_z1,
                                    outlier_z2=config.outlier_z2).fit(tv_ehr.dataset, self.get_admission_ids(tv_ehr))
        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.outlier_removers.obs, tv_ehr, remover)
        report = self.report(report,
                             table='obs', column=None, value_type='type',
                             operation='TVxEHR.numerical_processors.outlier_removers.obs <- IQROutlierRemover',
                             after=type(remover))

        n1 = len(tv_ehr.dataset.tables.obs)
        # TODO: report specific removals stats for each code.
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, remover(tv_ehr.dataset))
        n2 = len(tv_ehr.dataset.tables.obs)
        report = self.report(report, table='obs', column=None, value_type='count',
                             operation='filter', before=n1, after=n2)
        return tv_ehr, report


class ObsAdaptiveScaler(TrainableTransformation):
    def __call__(self, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        config = tv_ehr.config.numerical_processors.scalers.obs
        value_column = lambda x: x.config.tables.obs.value_alias
        scaler = AdaptiveScaler(table=lambda x: x.tables.obs,
                                code_column=lambda x: x.config.tables.obs.code_alias,
                                value_column=value_column,
                                use_float16=config.use_float16).fit(tv_ehr.dataset,
                                                                    self.get_admission_ids(tv_ehr))
        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.scalers.obs, tv_ehr, scaler)
        report = self.report(report,
                             table='obs', column=None, value_type='type',
                             operation='TVxEHR.numerical_processors.scalers.obs <- AdaptiveScaler',
                             after=type(scaler))

        dtype1 = tv_ehr.dataset.tables.obs[value_column(tv_ehr.dataset)].dtype
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, scaler(tv_ehr.dataset))
        dtype2 = tv_ehr.dataset.tables.obs[value_column(tv_ehr.dataset)].dtype
        report = self.report(report, table='obs', column=value_column(tv_ehr.dataset),
                             value_type='dtype',
                             operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                             before=dtype1, after=dtype2)
        return tv_ehr, report


class InputScaler(TrainableTransformation):
    def __call__(self, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        code_column = lambda x: x.config.tables.icu_inputs.code_alias
        value_column = lambda x: x.config.tables.icu_inputs.derived_normalized_amount_per_hour
        scaler = MaxScaler(table=lambda x: x.tables.icu_inputs,
                           code_column=code_column,
                           value_column=value_column,
                           use_float16=self.use_float16).fit(tv_ehr.dataset, self.get_admission_ids(tv_ehr))

        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.scalers.icu_inputs, tv_ehr, scaler)
        report = self.report(report,
                             table='icu_inputs', column=None, value_type='type',
                             operation='TVxEHR.numerical_processors.scalers.icu_inputs <- MaxScaler',
                             after=type(scaler))

        dtype1 = tv_ehr.dataset.tables.icu_inputs[value_column(tv_ehr.dataset)].dtype
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, scaler(tv_ehr.dataset))
        dtype2 = tv_ehr.dataset.tables.icu_inputs[value_column(tv_ehr.dataset)].dtype
        report = self.report(report, table='icu_inputs', column=value_column(tv_ehr.dataset),
                             value_type='dtype',
                             operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                             before=dtype1, after=dtype2)
        return tv_ehr, report


DEPENDS_RELATIONS: Final[Dict[Type[DatasetTransformation], Set[Type[DatasetTransformation]]]] = {
    SetAdmissionRelativeTimes: {CastTimestamps, SetIndex},
    FilterSubjectsNegativeAdmissionLengths: {CastTimestamps, SetIndex},
    SetCodeIntegerIndices: {FilterUnsupportedCodes},
    ProcessOverlappingAdmissions: {SetIndex, CastTimestamps},
    FilterClampTimestampsToAdmissionInterval: {SetIndex, CastTimestamps},
    SelectSubjectsWithObservation: {SetIndex},
    ICUInputRateUnitConversion: {SetCodeIntegerIndices},
    FilterInvalidInputRatesSubjects: {SetIndex, ICUInputRateUnitConversion},
    RandomSplits: {SetIndex, CastTimestamps},
    TrainableTransformation: {RandomSplits, SetIndex, SetCodeIntegerIndices},
    ObsAdaptiveScaler: {ObsIQROutlierRemover}
    # <- inherits also from TrainableTransformation (TODO: test the inheritance of dependencies).
}

BLOCKED_BY_RELATIONS: Final[Dict[Type[DatasetTransformation], Set[Type[DatasetTransformation]]]] = {
    FilterClampTimestampsToAdmissionInterval: {SetAdmissionRelativeTimes},
    SelectSubjectsWithObservation: {SetCodeIntegerIndices},
    ICUInputRateUnitConversion: {SetCodeIntegerIndices},
}
PIPELINE_VALIDATOR = TransformationsDependency(
    depends=DEPENDS_RELATIONS,
    blocked_by=BLOCKED_BY_RELATIONS,
)

#     def subject_info_extractor(self, subject_ids, target_scheme):
#
#         static_df = self.df['static']
#         c_gender = self.colname["static"].gender
#         c_anchor_year = self.colname["static"].anchor_year
#         c_anchor_age = self.colname["static"].anchor_age
#         c_eth = self.colname["static"].ethnicity
#
#         static_df = static_df.loc[subject_ids]
#         gender = static_df[c_gender].map(self.scheme.gender.codeset2vec)
#         subject_gender = gender.to_dict()
#
#         anchor_date = pd.to_datetime(static_df[c_anchor_year],
#                                      format='%Y').dt.normalize()
#         anchor_age = static_df[c_anchor_age].map(
#             lambda y: pd.DateOffset(years=-y))
#         dob = anchor_date + anchor_age
#         subject_dob = dict(zip(static_df.index.values, dob))
#         subject_eth = dict()
#         eth_mapper = self.scheme.ethnicity_mapper(target_scheme)
#         for subject_id in static_df.index.values:
#             eth_code = eth_mapper.map_codeset(
#                 [static_df.loc[subject_id, c_eth]])
#             subject_eth[subject_id] = eth_mapper.codeset2vec(eth_code)
#
#         return subject_dob, subject_gender, subject_eth
#
#     def dx_codes_extractor(self, admission_ids_list, target_scheme):
#         c_adm_id = self.colname["dx_discharge"].admission_id
#         c_code = self.colname["dx_discharge"].code
#         c_version = self.colname["dx_discharge"].version
#
#         df = self.df["dx_discharge"]
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#         codes_df = {
#             adm_id: codes_df
#             for adm_id, codes_df in df.groupby(c_adm_id)
#         }
#         empty_vector = target_scheme.dx_discharge.empty_vector()
#
#         dx_mapper = self.scheme.dx_mapper(target_scheme)
#
#         def _extract_codes(adm_id):
#             _codes_df = codes_df.get(adm_id)
#             if _codes_df is None:
#                 return (adm_id, empty_vector)
#
#             vec = empty_vector
#             for version, version_df in _codes_df.groupby(c_version):
#                 mapper = dx_mapper[str(version)]
#                 codeset = mapper.map_codeset(version_df[c_code])
#                 vec = vec.union(mapper.codeset2vec(codeset))
#             return (adm_id, vec)
#
#         return map(_extract_codes, admission_ids_list)
#
#
# class MIMIC4ICUDataset(Dataset):
#
#     @classmethod
#     def _setup_core_pipeline(cls, config: DatasetConfig) -> ValidatedDatasetPipeline:
#         raise NotImplementedError("Not implemented")
#
#     def to_subjects(self,
#                     subject_ids: List[int],
#                     num_workers: int,
#                     demographic_vector_config: DemographicVectorConfig,
#                     leading_observable_config: LeadingObservableExtractorConfig,
#                     target_scheme: MIMIC4ICUDatasetScheme,
#                     time_binning: Optional[int] = None,
#                     **kwargs):
#
#         subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
#             subject_ids, target_scheme)
#         admission_ids = self.adm_extractor(subject_ids)
#         adm_ids_list = sum(map(list, admission_ids.values()), [])
#         logging.debug('Extracting dx_discharge codes...')
#         dx_codes = dict(self.dx_codes_extractor(adm_ids_list, target_scheme))
#         logging.debug('[DONE] Extracting dx_discharge codes')
#         logging.debug('Extracting dx_discharge codes history...')
#         dx_codes_history = dict(
#             self.dx_codes_history_extractor(dx_codes, admission_ids,
#                                             target_scheme))
#         logging.debug('[DONE] Extracting dx_discharge codes history')
#         logging.debug('Extracting outcome...')
#         outcome = dict(self.outcome_extractor(dx_codes, target_scheme))
#         logging.debug('[DONE] Extracting outcome')
#         logging.debug('Extracting procedures...')
#         procedures = dict(self.procedure_extractor(adm_ids_list))
#         logging.debug('[DONE] Extracting procedures')
#         logging.debug('Extracting inputs...')
#         inputs = dict(self.inputs_extractor(adm_ids_list))
#         logging.debug('[DONE] Extracting inputs')
#         logging.debug('Extracting observables...')
#         observables = dict(
#             self.observables_extractor(adm_ids_list, num_workers))
#
#         if time_binning is not None:
#             observables = dict((k, v.time_binning(time_binning))
#                                for k, v in observables.items())
#
#         logging.debug('[DONE] Extracting observables')
#
#         logging.debug('Compiling admissions...')
#         c_admittime = self.colname['adm'].admittime
#         c_dischtime = self.colname['adm'].dischtime
#         c_adm_interval = self.colname['adm'].adm_interval
#         adf = self.df['adm']
#         adm_dates = dict(
#             zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))
#         adm_interval = dict(zip(adf.index, adf[c_adm_interval]))
#         proc_repr = AggregateRepresentation(self.scheme.int_proc,
#                                             target_scheme.int_proc)
#
#         leading_obs_extractor = LeadingObservableExtractor(leading_observable_config)
#
#         def gen_admission(i):
#             interventions = InpatientInterventions(
#                 proc=procedures[i],
#                 input_=inputs[i],
#                 adm_interval=adm_interval[i])
#
#             obs = observables[i]
#             lead_obs = leading_obs_extractor(obs)
#
#             if time_binning is None:
#                 interventions = interventions.segment_proc(proc_repr)
#                 interventions = interventions.segment_input()
#                 lead_obs = lead_obs.segment(interventions.t_sep)
#                 obs = obs.segment(interventions.t_sep)
#
#             return Admission(admission_id=i,
#                              admission_dates=adm_dates[i],
#                              dx_codes=dx_codes[i],
#                              dx_codes_history=dx_codes_history[i],
#                              outcome=outcome[i],
#                              observables=obs,
#                              leading_observable=lead_obs,
#                              interventions=interventions)
#
#         def _gen_subject(subject_id):
#
#             _admission_ids = admission_ids[subject_id]
#             # for subject_id, subject_admission_ids in admission_ids.items():
#             _admission_ids = sorted(_admission_ids,
#                                     key=lambda aid: adm_dates[aid][0])
#
#             static_info = StaticInfo(
#                 date_of_birth=subject_dob[subject_id],
#                 gender=subject_gender[subject_id],
#                 ethnicity=subject_eth[subject_id],
#                 demographic_vector_config=demographic_vector_config)
#
#             with ThreadPoolExecutor(max_workers=num_workers) as executor:
#                 admissions = list(executor.map(gen_admission, _admission_ids))
#             return Patient(subject_id=subject_id,
#                            admissions=admissions,
#                            static_info=static_info)
#
#         return list(map(_gen_subject, subject_ids))
#
#     def procedure_extractor(self, admission_ids_list):
#         c_adm_id = self.colname["int_proc"].admission_id
#         c_code_index = self.colname["int_proc"].code_source_index
#         c_start_time = self.colname["int_proc"].start_time
#         c_end_time = self.colname["int_proc"].end_time
#         df = self.df["int_proc"]
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#
#         def group_fun(x):
#             return pd.Series({
#                 0: x[c_code_index].to_numpy(),
#                 1: x[c_start_time].to_numpy(),
#                 2: x[c_end_time].to_numpy()
#             })
#
#         grouped = df.groupby(c_adm_id).apply(group_fun)
#         adm_arr = grouped.index.tolist()
#         input_size = len(self.scheme.int_proc)
#         for i in adm_arr:
#             yield (i,
#                    InpatientInput(index=grouped.loc[i, 0],
#                                   rate=np.ones_like(grouped.loc[i, 0],
#                                                     dtype=bool),
#                                   starttime=grouped.loc[i, 1],
#                                   endtime=grouped.loc[i, 2],
#                                   size=input_size))
#
#         for adm_id in set(admission_ids_list) - set(adm_arr):
#             yield (adm_id, InpatientInput.empty(input_size))
#
#     def inputs_extractor(self, admission_ids_list):
#         c_adm_id = self.colname["int_input"].admission_id
#         c_start_time = self.colname["int_input"].start_time
#         c_end_time = self.colname["int_input"].end_time
#         c_rate = self.colname["int_input"].rate
#         c_code_index = self.colname["int_input"].code_source_index
#
#         df = self.df["int_input"]
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#
#         def group_fun(x):
#             return pd.Series({
#                 0: x[c_code_index].to_numpy(),
#                 1: x[c_rate].to_numpy(),
#                 2: x[c_start_time].to_numpy(),
#                 3: x[c_end_time].to_numpy()
#             })
#
#         grouped = df.groupby(c_adm_id).apply(group_fun)
#         adm_arr = grouped.index.tolist()
#         input_size = len(self.scheme.int_input)
#         for i in adm_arr:
#             yield (i,
#                    InpatientInput(index=grouped.loc[i, 0],
#                                   rate=grouped.loc[i, 1],
#                                   starttime=grouped.loc[i, 2],
#                                   endtime=grouped.loc[i, 3],
#                                   size=input_size))
#         for adm_id in set(admission_ids_list) - set(adm_arr):
#             yield (adm_id, InpatientInput.empty(input_size))
#
#     def observables_extractor(self, admission_ids_list, num_workers):
#         c_adm_id = self.colname["obs"].admission_id
#         c_time = self.colname["obs"].timestamp
#         c_value = self.colname["obs"].value
#         c_code_index = self.colname["obs"].code_source_index
#
#         df = self.df["obs"][[c_adm_id, c_time, c_value, c_code_index]]
#         logging.debug("obs: filter adms")
#         df = df[df[c_adm_id].isin(admission_ids_list)]
#
#         obs_dim = len(self.scheme.obs)
#
#         def ret_put(a, *args):
#             np.put(a, *args)
#             return a
#
#         def val_mask(x):
#             idx = x[c_code_index]
#             val = ret_put(np.zeros(obs_dim, dtype=np.float16), idx, x[c_value])
#             mask = ret_put(np.zeros(obs_dim, dtype=bool), idx, 1.0)
#             adm_id = x.index[0]
#             time = x[c_time].iloc[0]
#             return pd.Series({0: adm_id, 1: time, 2: val, 3: mask})
#
#         def gen_observation(val_mask):
#             time = val_mask[1].to_numpy()
#             value = val_mask[2]
#             mask = val_mask[3]
#             mask = np.vstack(mask.values).reshape((len(time), obs_dim))
#             value = np.vstack(value.values).reshape((len(time), obs_dim))
#             return InpatientObservables(time=time, value=value, mask=mask)
#
#         def partition_fun(part_df):
#             g = part_df.groupby([c_adm_id, c_time], sort=True, as_index=False)
#             return g.apply(val_mask).groupby(0).apply(gen_observation)
#
#         logging.debug("obs: dasking")
#         df = df.set_index(c_adm_id)
#         df = dd.from_pandas(df, npartitions=12, sort=True)
#         logging.debug("obs: groupby")
#         obs_obj_df = df.map_partitions(partition_fun, meta=(None, object))
#         logging.debug("obs: undasking")
#         obs_obj_df = obs_obj_df.compute()
#         logging.debug("obs: extract")
#
#         collected_adm_ids = obs_obj_df.index.tolist()
#         assert len(collected_adm_ids) == len(set(collected_adm_ids)), \
#             "Duplicate admission ids in obs"
#
#         for adm_id, obs in obs_obj_df.items():
#             yield (adm_id, obs)
#
#         logging.debug("obs: empty")
#         for adm_id in set(admission_ids_list) - set(obs_obj_df.index):
#             yield (adm_id, InpatientObservables.empty(obs_dim))
#
