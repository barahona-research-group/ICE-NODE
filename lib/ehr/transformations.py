from __future__ import annotations

import logging
from abc import ABCMeta
from collections import defaultdict
from dataclasses import field
from typing import Dict, Tuple, List, Any, ClassVar, Final, Type, Set

import equinox as eqx
import numpy as np
import pandas as pd

from .dataset import (Dataset, AbstractTransformation, AbstractDatasetPipeline,
                      TransformationsDependency, Report, SECONDS_TO_HOURS_SCALER)
from .example_datasets.mimiciv import MIMICIVDataset


class DatasetTransformation(AbstractTransformation, metaclass=ABCMeta):

    @staticmethod
    def synchronize_index(dataset: Dataset, indexed_table_name: str,
                          index_name: str, report: Report) -> Tuple[Dataset, Report]:
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
            report = report.add(table=table_name, column=index_name, before=n1, after=n2, value_type='count',
                                operation='sync_index')
            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)

        return eqx.tree_at(lambda x: x.tables, dataset, tables), report

    @staticmethod
    def filter_no_admission_subjects(dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        static = dataset.tables.static
        admissions = dataset.tables.admissions
        c_subject = dataset.config.tables.static.subject_id_alias
        no_admission_subjects = static[~static.index.isin(admissions[c_subject].unique())].index
        n1 = len(static)
        static = static.drop(no_admission_subjects, axis='index')
        n2 = len(static)
        report = report.add(table='static', column=c_subject, before=n1, after=n2, value_type='count',
                            operation='filter_no_admission_subjects')
        return eqx.tree_at(lambda x: x.tables.static, dataset, static), report

    @classmethod
    def synchronize_admissions(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        dataset, report = cls.synchronize_index(dataset, 'admissions',
                                                dataset.config.tables.admissions.admission_id_alias, report)
        return cls.filter_no_admission_subjects(dataset, report)

    @classmethod
    def synchronize_subjects(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        # Synchronizing subjects might entail synchronizing admissions, so we need to call it first
        dataset, report = cls.synchronize_index(dataset, 'static',
                                                dataset.config.tables.static.subject_id_alias, report)
        return cls.synchronize_admissions(dataset, report)


class SetIndex(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        for indexed_table_name, index_name in dataset.config.tables.indices.items():
            table = tables_dict[indexed_table_name]
            index1 = table.index.name
            table = table.set_index(index_name)
            index2 = table.index.name
            report = report.add(table=indexed_table_name, column=index_name, before=index1, after=index2,
                                value_type='index_name',
                                operation='set_index')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, indexed_table_name), dataset, table)
        return dataset, report


class CastTimestamps(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        tables = dataset.tables
        tables_dict = tables.tables_dict
        for table_name, time_cols in dataset.config.tables.time_cols.items():

            table = tables_dict[table_name].iloc[:, :]
            for time_col in time_cols:
                assert time_col in table.columns, f'{time_col} not found in {table_name}'

                if table[time_col].dtype == 'datetime64[ns]':
                    logging.debug(f'{table_name}[{time_col}] already in datetime64[ns]')
                    continue
                dtype1 = table[time_col].dtype
                table[time_col] = pd.to_datetime(table[time_col], errors='raise')
                dtype2 = table[time_col].dtype
                report = report.add(table=table_name, column=time_col, before=dtype1, after=dtype2,
                                    value_type='dtype',
                                    operation='cast')

            tables = eqx.tree_at(lambda x: getattr(x, table_name), tables, table)
        return eqx.tree_at(lambda x: x.tables, dataset, tables), report


class SetAdmissionRelativeTimes(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
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

                report = report.add(table=table_name, column=time_col, before=table[time_col].dtype,
                                    after=df[time_col].dtype,
                                    value_type='dtype', operation='set_admission_relative_times')

            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, df)

        return dataset, report


class FilterSubjectsNegativeAdmissionLengths(DatasetTransformation):

    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        table_config = dataset.config.tables.admissions
        c_subject_id = table_config.subject_id_alias
        c_dischtime = table_config.discharge_time_alias
        c_admittime = table_config.admission_time_alias
        admissions = dataset.tables.admissions

        # assert dtypes are datetime64[ns]
        assert admissions[c_admittime].dtype == 'datetime64[ns]' and \
               admissions[c_dischtime].dtype == 'datetime64[ns]', \
            f'{c_admittime} and {c_dischtime} must be datetime64[ns]'

        static = dataset.tables.static
        neg_los_subjects = admissions[admissions[c_dischtime] < admissions[c_admittime]][c_subject_id].unique()
        n_before = len(static)
        static = static[~static.index.isin(neg_los_subjects)]
        n_after = len(static)
        report = report.add(table='static', column=c_subject_id, value_type='count', operation='filter',
                            before=n_before, after=n_after)
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class FilterUnsupportedCodes(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        tables_dict = dataset.tables.tables_dict
        for table_name, code_column in dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(dataset.scheme, table_name)
            n1 = len(table)
            table = table[table[code_column].isin(coding_scheme.codes)]
            n2 = len(table)
            report = report.add(table=table_name, column=code_column, before=n1, after=n2, value_type='count',
                                operation='filter')
            dataset = eqx.tree_at(lambda x: getattr(x.tables, table_name), dataset, table)
        return dataset, report


class ProcessOverlappingAdmissions(DatasetTransformation):

    @staticmethod
    def map_admission_ids(dataset: Dataset, sub2sup: Dict[str, Any], report: Report) -> Tuple[Dataset, Report]:
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
            report = report.add(table=table_name, column=c_admission_id, before=n1, after=n2,
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
                                      dataset: Dataset,
                                      sub2sup: Dict[str, str], report: Report) -> Tuple[Dataset, Report]:
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
        report = report.add(table='admissions', column=c_admission_id, value_type='count',
                            operation='merge_overlapping_admissions',
                            before=n1, after=n2)

        # Step 4: update admission ids in other tables.
        return cls.map_admission_ids(dataset, sub2sup, report)

    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        admissions = dataset.tables.admissions
        table_config = dataset.config.tables.admissions
        c_subject_id = table_config.subject_id_alias
        c_dischtime = table_config.discharge_time_alias
        c_admittime = table_config.admission_time_alias

        # Step 1: Collect overlapping admissions
        # Map from sub-admissions to the new super-admissions.
        sub2sup = {adm_id: super_adm_id for _, subject_adms in admissions.groupby(c_subject_id)
                   for adm_id, super_adm_id in cls._collect_overlaps(subject_adms, c_admittime, c_dischtime).items()}

        # Step 2: Apply action.
        if dataset.config.overlapping_admissions == "merge":
            # Step 3: Extend discharge time of super admissions, remove sub-admissions,
            # and update admission ids in other tables.
            return cls._merge_overlapping_admissions(dataset, sub2sup, report)
        elif dataset.config.overlapping_admissions == "remove":
            # Step 3: Collect subjects with at least one overlapping admission and remove them entirely.
            subject_ids = admissions.loc[sub2sup.keys(), c_subject_id].unique()
            static = dataset.tables.static
            n1 = len(static)
            static = static.drop(subject_ids, axis='index')
            n2 = len(static)
            report = report.add(table='static', column=c_subject_id, value_type='count',
                                operation='filter_problematic_subjects',
                                before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
            # Step 4: synchronize subjects
            return cls.synchronize_subjects(dataset, report)
        else:
            raise ValueError(f'Unsupported action: {dataset.config.overlapping_admissions}')


class FilterClampTimestampsToAdmissionInterval(DatasetTransformation):

    @classmethod
    def _filter_timestamped_tables(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        timestamped_tables_conf = dataset.config.tables.timestamped_table_config_dict
        timestamped_tables = {name: getattr(dataset.tables, name) for name in
                              timestamped_tables_conf.keys()}
        table_config = dataset.config.tables.admissions
        c_admission_id = table_config.admission_id_alias
        c_dischtime = table_config.discharge_time_alias
        c_admittime = table_config.admission_time_alias

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
            report = report.add(table=name, column=c_time, value_type='count', operation='filter',
                                before=n1, after=n2)
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, table)

        return dataset, report

    @classmethod
    def _filter_interval_based_tables(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        interval_based_tables_conf = dataset.config.tables.interval_based_table_config_dict
        interval_based_tables: Dict[str, pd.DataFrame] = {name: getattr(dataset.tables, name) for name in
                                                          interval_based_tables_conf.keys()}
        table_config = dataset.config.tables.admissions
        c_admission_id = table_config.admission_id_alias
        c_dischtime = table_config.discharge_time_alias
        c_admittime = table_config.admission_time_alias

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
            report = report.add(table=name, column=(c_start_time, c_end_time),
                                value_type='count', operation='filter',
                                before=n1, after=n2)

            # Step 2: Clamp intervals to admission interval if either side is outside.
            n_to_clamp = np.sum((df[c_start_time] < df[c_admittime]) | (df[c_end_time] > df[c_dischtime]))
            report = report.add(table=name, column=(c_start_time, c_end_time),
                                value_type='count', operation='clamp',
                                before=None, after=n_to_clamp)
            df[c_start_time] = df[c_start_time].clip(lower=df[c_admittime], upper=df[c_dischtime])
            df[c_end_time] = df[c_end_time].clip(lower=df[c_admittime], upper=df[c_dischtime])
            df = df[table.columns]
            dataset = eqx.tree_at(lambda x: getattr(x.tables, name), dataset, df)

        return dataset, report

    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        dataset, report = cls._filter_timestamped_tables(dataset, report)
        return cls._filter_interval_based_tables(dataset, report)


class SelectSubjectsWithObservation(DatasetTransformation):

    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
        c_code = dataset.config.tables.obs.code_alias
        c_admission_id = dataset.config.tables.obs.admission_id_alias
        c_subject = dataset.config.tables.static.subject_id_alias
        obs = dataset.tables.obs

        code: Final[str] = dataset.config.filter_subjects_with_observation
        assert code is not None, 'No code provided for filtering subjects'

        admission_ids = obs[obs[c_code] == code][c_admission_id].unique()
        assert len(admission_ids) > 0, f'No observations for code {code}'

        subjects = dataset.tables.admissions.loc[admission_ids, c_subject].unique()
        static = dataset.tables.static
        n1 = len(static)
        static = static[static.index.isin(subjects)]
        n2 = len(static)
        report = report.add(table='static', column=c_subject, value_type='count',
                            operation=f'select_subjects(has({code}))',
                            before=n1, after=n2)
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


class ICUInputRateUnitConversion(DatasetTransformation):

    @classmethod
    def apply(cls, dataset: MIMICIVDataset, report: Report) -> Tuple[Dataset, Report]:
        ds_config = dataset.config
        tables_config = ds_config.tables
        table_config = tables_config.icu_inputs
        c_code = table_config.code_alias
        c_amount = table_config.amount_alias
        c_start_time = table_config.start_time_alias
        c_end_time = table_config.end_time_alias
        c_amount_unit = table_config.amount_unit_alias
        c_normalized_amount = table_config.derived_normalized_amount
        c_normalized_amount_per_hour = table_config.derived_normalized_amount_per_hour
        c_universal_unit = table_config.derived_universal_unit
        c_normalization_factor = table_config.derived_unit_normalization_factor
        icu_inputs = dataset.tables.icu_inputs

        _derived_columns = [c_normalized_amount, c_normalized_amount_per_hour, c_universal_unit, c_normalization_factor]

        conversion_table = dataset.icu_inputs_uom_normalization(
            dataset.config.tables.icu_inputs, dataset.config.scheme.icu_inputs_uom_normalization_table)

        assert (c in icu_inputs.columns for c in [c_code, c_amount, c_amount_unit]), \
            f"Some columns in: {c_code}, {c_amount}, {c_amount_unit}, not found in icu_inputs table"
        assert all(c not in icu_inputs.columns for c in _derived_columns), \
            f"Some of these columns [{', '.join(_derived_columns)}] already exists in icu_inputs table"
        assert (c in conversion_table for c in _derived_columns[2:]), \
            f"Some columns in: {', '.join(_derived_columns[2:])}, not " \
            "found in the conversion table"

        df = pd.merge(icu_inputs, conversion_table, how='left',
                      on=[c_code, c_amount_unit],
                      suffixes=(None, '_y'))

        delta_hours = ((df[c_end_time] - df[c_start_time]).dt.total_seconds() * SECONDS_TO_HOURS_SCALER)
        df[c_normalized_amount] = df[c_amount] * df[c_normalization_factor]
        df[c_normalized_amount_per_hour] = df[c_normalized_amount] / delta_hours
        df = df[icu_inputs.columns.tolist() + _derived_columns]
        dataset = eqx.tree_at(lambda x: x.tables.icu_inputs, dataset, df)
        report = report.add(table='icu_inputs', column=None,
                            value_type='columns', operation='new_columns',
                            before=icu_inputs.columns.tolist(), after=df.columns.tolist())

        return dataset, report


class FilterInvalidInputRatesSubjects(DatasetTransformation):
    @classmethod
    def apply(cls, dataset: Dataset, report: Report) -> Tuple[Dataset, Report]:
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

        report = report.add(table=('icu_inputs', 'admissions', 'static'),
                            column=(c_rate, c_admission_id, c_subject_id),
                            value_type='nan_counts',
                            before=(n_nan_inputs, n_nan_adms, n_nan_subjects),
                            after=None,
                            operation='filter_invalid_input_rates_subjects')

        n1 = len(static)
        static = static[~static.index.isin(nan_subject_ids)]
        n2 = len(static)
        report = report.add(table='static', column=c_subject_id, value_type='count',
                            before=n1, after=n2,
                            operation='filter_invalid_input_rates_subjects')
        dataset = eqx.tree_at(lambda x: x.tables.static, dataset, static)
        return cls.synchronize_subjects(dataset, report)


DS_DEPENDS_RELATIONS: Final[Dict[Type[DatasetTransformation], Set[Type[DatasetTransformation]]]] = {
    SetAdmissionRelativeTimes: {CastTimestamps, SetIndex},
    FilterSubjectsNegativeAdmissionLengths: {CastTimestamps, SetIndex},
    ProcessOverlappingAdmissions: {SetIndex, CastTimestamps},
    FilterClampTimestampsToAdmissionInterval: {SetIndex, CastTimestamps},
    SelectSubjectsWithObservation: {SetIndex},
    FilterInvalidInputRatesSubjects: {SetIndex, ICUInputRateUnitConversion},
}

DS_BLOCKED_BY_RELATIONS: Final[Dict[Type[DatasetTransformation], Set[Type[DatasetTransformation]]]] = {
    FilterClampTimestampsToAdmissionInterval: {SetAdmissionRelativeTimes},
    ICUInputRateUnitConversion: {SetAdmissionRelativeTimes}
}
DS_PIPELINE_VALIDATOR: Final[TransformationsDependency] = TransformationsDependency( {}, {}
    # depends=DS_DEPENDS_RELATIONS,
    # blocked_by=DS_BLOCKED_BY_RELATIONS,
)


class ValidatedDatasetPipeline(AbstractDatasetPipeline):
    transformations: List[DatasetTransformation] = field(kw_only=True)
    validator: ClassVar[TransformationsDependency] = DS_PIPELINE_VALIDATOR
