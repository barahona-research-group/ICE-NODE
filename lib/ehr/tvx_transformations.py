from __future__ import annotations

import logging
import random
from dataclasses import field
from typing import Dict, Tuple, List, Final, Type, Set

import dask.dataframe as dd
import equinox as eqx
import numpy as np
import pandas as pd

from . import TVxEHR, StaticInfo, CodesVector, InpatientInput, InpatientInterventions, InpatientObservables, \
    LeadingObservableExtractor, Admission, Patient
from .coding_scheme import CodeMap
from .dataset import Dataset, TransformationsDependency, AbstractTransformation, AdmissionIntervalBasedCodedTableConfig
from .transformations import CastTimestamps, SetIndex, DatasetTransformation
from .tvx_ehr import TVxReportAttributes, TrainableTransformation, AbstractTVxTransformation, TVxReport, \
    CodedValueScaler, CodedValueProcessor


class SampleSubjects(AbstractTVxTransformation):

    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        static = tv_ehr.dataset.tables.static
        # assert index name is subject_id
        c_subject_id = tv_ehr.dataset.config.tables.static.subject_id_alias
        assert c_subject_id in static.index.names, f'Index name must be {c_subject_id}'
        config = tv_ehr.config.sample
        rng = random.Random(config.seed)
        subjects = static.index.unique().tolist()
        rng.shuffle(subjects)
        subjects = subjects[config.offset:config.offset + config.n_subjects]
        n1 = len(static)
        static = static.loc[subjects]
        n2 = len(static)
        report = report.add(table='static', column='index', before=n1, after=n2, value_type='count',
                            transformation=cls,
                            operation='sample')
        dataset = eqx.tree_at(lambda x: x.tables.static, tv_ehr.dataset, static)
        dataset, report = DatasetTransformation.synchronize_subjects(dataset, report)
        return eqx.tree_at(lambda x: x.dataset, tv_ehr, dataset), report


class RandomSplits(AbstractTVxTransformation):

    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[
        TVxEHR, TVxReport]:
        config = tv_ehr.config.splits
        splits = tv_ehr.dataset.random_splits(splits=config.split_quantiles,
                                              random_seed=config.seed,
                                              balance=config.balance,
                                              discount_first_admission=config.discount_first_admission)

        report = report.add(table='static', column=None, value_type='splits',
                            transformation=cls,
                            operation=f'TVxEHR.splits<-TVxEHR.dataset.random_splits(TVxEHR.config.splits)',
                            before=(len(tv_ehr.dataset.tables.static),),
                            after=tuple(len(x) for x in splits))
        tv_ehr = eqx.tree_at(lambda x: x.splits, tv_ehr, splits)
        return tv_ehr, report


class ZScoreScaler(CodedValueScaler):
    mean: pd.Series = field(default_factory=lambda: pd.Series())
    std: pd.Series = field(default_factory=lambda: pd.Series())

    @property
    def original_dtype(self) -> np.dtype:
        return self.mean.dtype

    def __call__(self, dataset: Dataset) -> Dataset:
        table = self.table(dataset)

        mean = table[self.code_column].map(self.mean)
        std = table[self.code_column].map(self.std)
        table.loc[:, self.value_column] = (table[self.value_column] - mean) / std
        if self.use_float16:
            table = table.astype({self.value_column: np.float16})

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

        max_val = df[self.code_column].map(self.max_val)
        df.loc[:, self.value_column] = (df[self.value_column] / max_val)
        if self.use_float16:
            df = df.astype({self.value_column: np.float16})
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

        min_val = df[self.code_column].map(self.min_val)
        max_val = df[self.code_column].map(self.max_val)
        mean = df[self.code_column].map(self.mean)
        std = df[self.code_column].map(self.std)

        minmax_scaled = (df[self.value_column] - min_val) / max_val
        z_scaled = ((df[self.value_column] - mean) / std)

        df.loc[:, self.value_column] = np.where(min_val >= 0.0, minmax_scaled, z_scaled)
        if self.use_float16:
            df = df.astype({self.value_column: np.float16})
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

        min_val = table[self.code_column].map(self.min_val)
        max_val = table[self.code_column].map(self.max_val)
        table = table[table[self.value_column].between(min_val, max_val)]

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
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        config = tv_ehr.config.numerical_processors.outlier_removers.obs
        remover = IQROutlierRemover(table_name='obs',
                                    code_column=tv_ehr.dataset.config.tables.obs.code_alias,
                                    value_column=tv_ehr.dataset.config.tables.obs.value_alias,
                                    outlier_q1=config.outlier_q1,
                                    outlier_q2=config.outlier_q2,
                                    outlier_iqr_scale=config.outlier_iqr_scale,
                                    outlier_z1=config.outlier_z1,
                                    outlier_z2=config.outlier_z2).fit(tv_ehr.dataset, cls.get_admission_ids(tv_ehr))
        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.outlier_removers.obs, tv_ehr, remover,
                             is_leaf=lambda x: x is None)
        report = report.add(
            table='obs', column=None, value_type='type',
            transformation=cls,
            operation='TVxEHR.numerical_processors.outlier_removers.obs <- IQROutlierRemover',
            after=type(remover))

        n1 = len(tv_ehr.dataset.tables.obs)
        # TODO: report specific removals stats for each code.
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, remover(tv_ehr.dataset))
        n2 = len(tv_ehr.dataset.tables.obs)
        report = report.add(table='obs', column=None, value_type='count',
                            transformation=cls,
                            operation='filter', before=n1, after=n2)
        return tv_ehr, report


class ObsAdaptiveScaler(TrainableTransformation):
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        config = tv_ehr.config.numerical_processors.scalers.obs
        value_column = tv_ehr.dataset.config.tables.obs.value_alias
        scaler = AdaptiveScaler(table_name='obs',
                                code_column=tv_ehr.dataset.config.tables.obs.code_alias,
                                value_column=value_column,
                                use_float16=config.use_float16).fit(tv_ehr.dataset,
                                                                    cls.get_admission_ids(tv_ehr))
        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.scalers.obs, tv_ehr, scaler,
                             is_leaf=lambda x: x is None)
        report = report.add(
            table='obs', column=None, value_type='type',
            transformation=cls,
            operation='TVxEHR.numerical_processors.scalers.obs <- AdaptiveScaler',
            after=type(scaler))

        dtype1 = tv_ehr.dataset.tables.obs[value_column].dtype
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, scaler(tv_ehr.dataset))
        dtype2 = tv_ehr.dataset.tables.obs[value_column].dtype
        report = report.add(table='obs', column=value_column,
                            value_type='dtype',
                            transformation=cls,
                            operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                            before=dtype1, after=dtype2)
        return tv_ehr, report


class InputScaler(TrainableTransformation):
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        code_column = tv_ehr.dataset.config.tables.icu_inputs.code_alias
        value_column = tv_ehr.dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
        config = tv_ehr.config.numerical_processors.scalers.icu_inputs
        scaler = MaxScaler(table_name='icu_inputs',
                           code_column=code_column,
                           value_column=value_column,
                           use_float16=config.use_float16).fit(tv_ehr.dataset, cls.get_admission_ids(tv_ehr))

        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.scalers.icu_inputs, tv_ehr, scaler,
                             is_leaf=lambda x: x is None)
        report = report.add(
            table='icu_inputs', column=None, value_type='type',
            operation='TVxEHR.numerical_processors.scalers.icu_inputs <- MaxScaler',
            after=type(scaler))

        dtype1 = tv_ehr.dataset.tables.icu_inputs[value_column].dtype
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, scaler(tv_ehr.dataset))
        dtype2 = tv_ehr.dataset.tables.icu_inputs[value_column].dtype
        report = report.add(table='icu_inputs', column=value_column,
                            value_type='dtype',
                            transformation=cls,
                            operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                            before=dtype1, after=dtype2)
        return tv_ehr, report


TVX_DEPENDS_RELATIONS: Final[Dict[Type[AbstractTransformation], Set[Type[AbstractTransformation]]]] = {
    RandomSplits: {SetIndex, CastTimestamps},
    TrainableTransformation: {RandomSplits, SetIndex},
    ObsAdaptiveScaler: {ObsIQROutlierRemover}
    # <- inherits also from TrainableTransformation (TODO: test the inheritance of dependencies).
}

TVX_BLOCKED_BY_RELATIONS: Final[Dict[Type[AbstractTransformation], Set[Type[AbstractTransformation]]]] = {
    # Any TVX Transformation blocks DS Transformation.
    DatasetTransformation: {AbstractTVxTransformation}
}
TVX_PIPELINE_VALIDATOR: Final[TransformationsDependency] = TransformationsDependency(
    depends=TVX_DEPENDS_RELATIONS,
    blocked_by=TVX_BLOCKED_BY_RELATIONS,
)


class InterventionSegmentation(AbstractTVxTransformation):
    pass


class ObsTimeBinning(AbstractTVxTransformation):
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        if tv_ehr.config.time_binning is None:
            return cls.skip(tv_ehr, report)

        interval = tv_ehr.config.time_binning
        obs_scheme = tv_ehr.scheme.obs
        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix('observables',
                                                                          InpatientObservables)
        original_obs = [admission.observables for subject in tv_ehr.subjects.values()
                        for admission in subject.admissions]
        tv_ehr = eqx.tree_at(lambda x: x.subjects, tv_ehr,
                             {subject_id: subject.observables_time_binning(interval, obs_scheme)
                              for subject_id, subject in tv_ehr.subjects.items()})
        binned_obs = [admission.observables for subject in tv_ehr.subjects.values()
                      for admission in subject.admissions]

        report = report.add(tvx_concept=tvx_concept_path,
                            value_type='concepts_count', operation='time_binning',
                            transformation=cls,
                            before=len(original_obs),
                            after=len(binned_obs))
        report = report.add(tvx_concept=tvx_concept_path,
                            value_type='timestamps_count', operation='time_binning',
                            transformation=cls,
                            before=sum(len(o) for o in original_obs),
                            after=sum(len(o) for o in binned_obs))
        report = report.add(tvx_concept=tvx_concept_path,
                            transformation=cls,
                            value_type='values_count', operation='time_binning',
                            before=sum(o.count() for o in original_obs),
                            after=sum(o.count() for o in binned_obs))
        return tv_ehr, report


class LeadingObservableExtraction(AbstractTVxTransformation):

    # TODO: blocks time binning.
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        extractor = LeadingObservableExtractor(tv_ehr.config.leading_observable)

        if tv_ehr.config.leading_observable is None:
            return cls.skip(tv_ehr, report)

        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix('leading_observables',
                                                                          InpatientObservables)
        tv_ehr = eqx.tree_at(lambda x: x.subjects, tv_ehr,
                             {subject_id: subject.extract_leading_observables(extractor)
                              for subject_id, subject in tv_ehr.subjects.items()})
        extracted = [admission.leading_observables for subject in tv_ehr.subjects.values()
                     for admission in subject.admissions]

        report = report.add(tvx_concept=tvx_concept_path,
                            value_type='concepts_count', operation='LeadingObservableExtractor',
                            transformation=cls,
                            after=len(extracted))
        report = report.add(tvx_concept=tvx_concept_path,
                            value_type='timestamps_count', operation='LeadingObservableExtractor',
                            transformation=cls,
                            after=sum(len(o) for o in extracted))
        report = report.add(tvx_concept=tvx_concept_path,
                            transformation=cls,
                            value_type='values_count', operation='LeadingObservableExtractor',
                            after=sum(o.count() for o in extracted))
        report = report.add(
            transformation=cls,
            tvx_concept_path=tvx_concept_path,
            value_type='type',
            operation='LeadingObservableExtractor',
            after=InpatientObservables)
        return tv_ehr, report


class TVxConcepts(AbstractTVxTransformation):

    @classmethod
    def _static_info(cls, tvx_ehr: TVxEHR, report: TVxReport) -> Tuple[Dict[str, StaticInfo], TVxReport]:

        static = tvx_ehr.dataset.tables.static
        config = tvx_ehr.config.demographic
        c_gender = tvx_ehr.dataset.config.tables.static
        c_date_of_birth = tvx_ehr.dataset.config.tables.static.date_of_birth_alias
        c_ethnicity = tvx_ehr.dataset.config.tables.static.race_alias

        report = report.add(
            transformation=cls,
            table='static', column=None, value_type='count', operation='extract_static_info',
            after=len(static))

        gender, ethnicity, dob = {}, {}, {}
        if c_date_of_birth in static.columns or config.age:
            dob = static[c_date_of_birth].to_dict()
        if tvx_ehr.scheme.gender is not None or config.gender:
            gender_m = tvx_ehr.gender_mapper
            gender = {k: gender_m.codeset2vec({c}) for k, c in static[c_gender].to_dict().items()}

        if tvx_ehr.scheme.ethnicity is not None or config.ethnicity:
            ethnicity_m = tvx_ehr.ethnicity_mapper
            ethnicity = {k: ethnicity_m.codeset2vec({c}) for k, c in static[c_ethnicity].to_dict().items()}

        static_info = {subject_id: StaticInfo(date_of_birth=dob.get(subject_id),
                                              ethnicity=ethnicity.get(subject_id),
                                              gender=gender.get(subject_id),
                                              demographic_vector_config=config) for subject_id in static.index}
        report = report.add(tvx_concept=TVxReportAttributes.static_info_prefix(),
                            transformation=cls,
                            column=None, value_type='count', operation='extract_static_info',
                            after=len(static_info))
        return static_info, report

    @staticmethod
    def _dx_discharge(tvx_ehr: TVxEHR) -> Dict[str, CodesVector]:
        c_adm_id = tvx_ehr.dataset.config.tables.dx_discharge.admission_id_alias
        c_code = tvx_ehr.dataset.config.tables.dx_discharge.code_alias
        dx_discharge = tvx_ehr.dataset.tables.dx_discharge
        dx_codes_set = dx_discharge.groupby(c_adm_id)[c_code].apply(set).to_dict()
        dx_mapper = tvx_ehr.dx_mapper
        return {adm_id: dx_mapper.codeset2vec(codeset) for adm_id, codeset in dx_codes_set.items()}

    @staticmethod
    def _dx_discharge_history(tvx_ehr: TVxEHR, dx_discharge: Dict[str, CodesVector]) -> Dict[str, CodesVector]:
        # TODO: test anti causality.
        dx_scheme = tvx_ehr.scheme.dx_discharge
        # For each subject accumulate previous dx_discharge codes.
        dx_discharge_history = dict()
        initial_history = dx_scheme.codeset2vec(set())
        # For each subject get the list of adm sorted by admission date.
        for subject_id, adm_ids in tvx_ehr.subjects_sorted_admission_ids.items():
            current_history = initial_history
            for adm_id in adm_ids:
                dx_discharge_history[adm_id] = current_history
                current_history = dx_discharge[adm_id].union(current_history)
        return dx_discharge_history

    @staticmethod
    def _outcome(tvx_ehr: TVxEHR, dx_discharge: Dict[str, CodesVector]) -> Dict[str, CodesVector]:
        outcome_extractor = tvx_ehr.scheme.outcome
        return {adm_id: outcome_extractor.map_vector(code_vec) for adm_id, code_vec in dx_discharge.items()}

    @staticmethod
    def _icu_inputs(tvx_ehr: TVxEHR) -> Dict[str, InpatientInput]:
        table_config = tvx_ehr.dataset.config.tables.icu_inputs
        c_admission_id = table_config.admission_id_alias
        c_code = table_config.code_alias
        c_rate = table_config.derived_normalized_amount_per_hour
        c_start_time = table_config.start_time_alias
        c_end_time = table_config.end_time_alias

        # Here we avoid deep copy, and we can still replace
        # a new column without affecting the original table.
        table = tvx_ehr.dataset.tables.icu_inputs.iloc[:, :]

        table[c_code] = table[c_code].map(tvx_ehr.scheme.icu_inputs.index)

        def group_fun(x):
            return pd.Series({
                0: x[c_code].to_numpy(),
                1: x[c_rate].to_numpy(),
                2: x[c_start_time].to_numpy(),
                3: x[c_end_time].to_numpy()
            })

        admission_icu_inputs = table.groupby(c_admission_id).apply(group_fun)
        input_size = len(tvx_ehr.scheme.icu_inputs)
        return {adm_id: InpatientInput(index=np.array(codes, dtype=np.int64),
                                       rate=rates,
                                       starttime=start,
                                       endtime=end,
                                       size=input_size)
                for adm_id, (codes, rates, start, end) in admission_icu_inputs.iterrows()}

    @staticmethod
    def _procedures(table: pd.DataFrame, config: AdmissionIntervalBasedCodedTableConfig,
                    code_map: CodeMap) -> Dict[str, InpatientInput]:
        c_admission_id = config.admission_id_alias
        c_code = config.code_alias
        c_start_time = config.start_time_alias
        c_end_time = config.end_time_alias

        # Here we avoid deep copy, and we can still replace
        # a new column without affecting the original table.
        table = table.iloc[:, :]
        table[c_code] = table[c_code].map(code_map.source_to_target_index)
        assert not table[c_code].isnull().any(), 'Some codes are not in the target scheme.'

        def group_fun(x):
            return pd.Series({
                0: x[c_code].to_numpy(),
                1: x[c_start_time].to_numpy(),
                2: x[c_end_time].to_numpy()
            })

        admission_procedures = table.groupby(c_admission_id).apply(group_fun)
        size = len(code_map.target_scheme)
        return {adm_id: InpatientInput(index=np.array(codes, dtype=np.int64),
                                       rate=np.ones_like(codes, dtype=bool),
                                       starttime=start,
                                       endtime=end,
                                       size=size)
                for adm_id, (codes, start, end) in admission_procedures.iterrows()}

    @staticmethod
    def _hosp_procedures(tvx_ehr: TVxEHR) -> Dict[str, InpatientInput]:
        return TVxConcepts._procedures(tvx_ehr.dataset.tables.hosp_procedures,
                                       tvx_ehr.dataset.config.tables.hosp_procedures,
                                       tvx_ehr.hosp_procedures_mapper)

    @staticmethod
    def _icu_procedures(tvx_ehr: TVxEHR) -> Dict[str, InpatientInput]:
        return TVxConcepts._procedures(tvx_ehr.dataset.tables.icu_procedures,
                                       tvx_ehr.dataset.config.tables.icu_procedures,
                                       tvx_ehr.icu_procedures_mapper)

    @classmethod
    def _interventions(cls, tvx_ehr: TVxEHR, report: TVxReport) -> Tuple[Dict[str, InpatientInterventions], TVxReport]:
        concept_path = TVxReportAttributes.inpatient_input_prefix
        hosp_procedures = cls._hosp_procedures(tvx_ehr)
        report = report.add(tvx_concept=concept_path('hosp_procedures'),
                            transformation=cls,
                            table='hosp_procedures',
                            column=None, value_type='count', operation='extract_hosp_procedures',
                            after=len(hosp_procedures))

        icu_procedures = cls._icu_procedures(tvx_ehr)
        report = report.add(tvx_concept=concept_path('icu_procedures'),
                            table='icu_procedures',
                            transformation=cls,
                            column=None, value_type='count', operation='extract_icu_procedures',
                            after=len(icu_procedures))

        icu_inputs = cls._icu_inputs(tvx_ehr)
        report = report.add(tvx_concept=concept_path('icu_inputs'), table='icu_inputs',
                            transformation=cls,

                            column=None, value_type='count', operation='extract_icu_inputs',
                            after=len(icu_inputs))
        interventions = {admission_id: InpatientInterventions(hosp_procedures=hosp_procedures.get(admission_id),
                                                              icu_procedures=icu_procedures.get(admission_id),
                                                              icu_inputs=icu_inputs.get(admission_id))
                         for admission_id in tvx_ehr.admission_ids}
        return interventions, report

    @classmethod
    def _observables(cls, tvx_ehr: TVxEHR, report: TVxReport) -> Tuple[Dict[str, InpatientObservables], TVxReport]:
        c_admission_id = tvx_ehr.dataset.config.tables.obs.admission_id_alias
        c_code = tvx_ehr.dataset.config.tables.obs.code_alias
        c_value = tvx_ehr.dataset.config.tables.obs.value_alias
        c_timestamp = tvx_ehr.dataset.config.tables.obs.time_alias

        # For dasking, we index by admission.
        table = tvx_ehr.dataset.tables.obs.set_index(c_admission_id)
        table[c_code] = table[c_code].map(tvx_ehr.scheme.obs.index)
        assert not table[c_code].isnull().any(), 'Some codes are not in the target scheme.'
        obs_dim = len(tvx_ehr.scheme.obs)
        tvx_concept_path = TVxReportAttributes.admission_attribute_prefix('observables',
                                                                          InpatientObservables)

        def ret_put(a, *args):
            np.put(a, *args)
            return a

        def val_mask(x):
            idx = x[c_code]
            val = ret_put(np.zeros(obs_dim, dtype=np.float16), idx, x[c_value])
            mask = ret_put(np.zeros(obs_dim, dtype=bool), idx, 1.0)
            adm_id = x.index[0]
            time = x[c_timestamp].iloc[0]
            return pd.Series({0: adm_id, 1: time, 2: val, 3: mask})

        def gen_observation(val_mask: pd.DataFrame) -> InpatientObservables:
            time = val_mask[1].to_numpy()
            value = val_mask[2]
            mask = val_mask[3]
            mask = np.vstack(mask.values).reshape((len(time), obs_dim))
            value = np.vstack(value.values).reshape((len(time), obs_dim))
            return InpatientObservables(time=time, value=value, mask=mask)

        def partition_fun(part_df):
            g = part_df.groupby([c_admission_id, c_timestamp], sort=True, as_index=False)
            return g.apply(val_mask).groupby(0).apply(gen_observation)

        logging.debug("obs: dasking")
        table = dd.from_pandas(table, npartitions=12, sort=True)
        logging.debug("obs: groupby")
        inpatient_observables_df = table.map_partitions(partition_fun, meta=(None, object))
        logging.debug("obs: undasking")
        inpatient_observables_df = inpatient_observables_df.compute()
        logging.debug("obs: extract")
        assert len(inpatient_observables_df.index.tolist()) == len(set(inpatient_observables_df.index.tolist())), \
            "Duplicate admission ids in obs"
        inpatient_observables = inpatient_observables_df.to_dict()
        report = report.add(
            table='obs', value_type='table_size', operation='extract_observables',
            after=len(tvx_ehr.dataset.tables.obs))
        report = report.add(tvx_concept=tvx_concept_path,
                            table='obs', value_type='concepts_count', operation='extract_observables',
                            transformation=cls,
                            after=len(inpatient_observables))
        report = report.add(tvx_concept=TVxReportAttributes.admission_attribute_prefix('observables',
                                                                                       InpatientObservables),
                            table='obs', value_type='timestamps_count', operation='extract_observables',
                            transformation=cls,
                            after=sum(len(o) for o in inpatient_observables.values()))
        report = report.add(tvx_concept=TVxReportAttributes.admission_attribute_prefix('observables',
                                                                                       InpatientObservables),
                            transformation=cls,
                            table='obs', value_type='values_count', operation='extract_observables',
                            after=sum(o.count() for o in inpatient_observables.values()))
        return inpatient_observables, report

    @classmethod
    def apply(cls, tvx_ehr: TVxEHR, report: TVxReport) -> Tuple[TVxEHR, TVxReport]:
        subject_admissions = tvx_ehr.subjects_sorted_admission_ids
        static_info, report = cls._static_info(tvx_ehr, report)

        dx_discharge = cls._dx_discharge(tvx_ehr)
        dx_discharge_history = cls._dx_discharge_history(tvx_ehr, dx_discharge)
        outcome = cls._outcome(tvx_ehr, dx_discharge)
        if tvx_ehr.config.interventions:
            interventions, report = cls._interventions(tvx_ehr, report)
        else:
            interventions = None
        if tvx_ehr.config.observables:
            observables, report = cls._observables(tvx_ehr, report)
        else:
            observables = None

        def _admissions(admission_ids: List[str]) -> List[Admission]:
            return [Admission(admission_id=i,
                              admission_dates=tvx_ehr.admission_dates[i],
                              dx_codes=dx_discharge[i],
                              dx_codes_history=dx_discharge_history[i],
                              outcome=outcome[i],
                              observables=observables[i],
                              interventions=interventions[i])
                    for i in admission_ids]

        subjects = {subject_id: Patient(subject_id=subject_id,
                                        admissions=_admissions(admission_ids),
                                        static_info=static_info[subject_id])
                    for subject_id, admission_ids in subject_admissions.items()}
        tv_ehr = eqx.tree_at(lambda x: x.subjects, tvx_ehr, subjects)
        report = report.add(tvx_concept=TVxReportAttributes.subjects_prefix(),
                            transformation=cls,
                            value_type='count', operation='extract_subjects',
                            after=len(subjects))
        return tv_ehr, report
