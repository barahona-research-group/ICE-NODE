from __future__ import annotations

import logging
import random
from abc import abstractmethod
from dataclasses import field
from typing import Dict, Tuple, List, Callable, Final, Type, Set, Hashable

import dask.dataframe as dd
import equinox as eqx
import numpy as np
import pandas as pd

from . import TVxEHR, StaticInfo, CodesVector, InpatientInput, InpatientInterventions, InpatientObservables
from .coding_scheme import CodeMap
from .dataset import Dataset, TransformationsDependency, AbstractTransformation, AdmissionIntervalBasedCodedTableConfig
from .transformations import CastTimestamps, SetIndex, DatasetTransformation
from .tvx_ehr import TVxReportAttributes, TrainableTransformation, AbstractTVxTransformation


class SampleSubjects(AbstractTVxTransformation):

    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
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
        report = cls.report(report, table='static', column='index', before=n1, after=n2, value_type='count',
                            operation='sample')
        dataset = eqx.tree_at(lambda x: x.tables.static, tv_ehr.dataset, static)
        dataset = DatasetTransformation.synchronize_subjects(dataset, report, cls.reporter())
        return eqx.tree_at(lambda x: x.dataset, tv_ehr, dataset), report


class RandomSplits(AbstractTVxTransformation):

    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        config = tv_ehr.config.splits
        splits = tv_ehr.dataset.random_splits(splits=config.split_quantiles,
                                              random_seed=config.seed,
                                              balance=config.balance,
                                              discount_first_admission=config.discount_first_admission)

        report = cls.report(report, table='static', column=None, value_type='splits',
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
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        config = tv_ehr.config.numerical_processors.outlier_removers.obs
        remover = IQROutlierRemover(table=lambda x: x.dataset.tables.obs,
                                    code_column=lambda x: x.dataset.config.tables.obs.code_alias,
                                    value_column=lambda x: x.dataset.config.tables.obs.value_alias,
                                    outlier_q1=config.outlier_q1,
                                    outlier_q2=config.outlier_q2,
                                    outlier_iqr_scale=config.outlier_iqr_scale,
                                    outlier_z1=config.outlier_z1,
                                    outlier_z2=config.outlier_z2).fit(tv_ehr.dataset, cls.get_admission_ids(tv_ehr))
        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.outlier_removers.obs, tv_ehr, remover)
        report = cls.report(report,
                            table='obs', column=None, value_type='type',
                            operation='TVxEHR.numerical_processors.outlier_removers.obs <- IQROutlierRemover',
                            after=type(remover))

        n1 = len(tv_ehr.dataset.tables.obs)
        # TODO: report specific removals stats for each code.
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, remover(tv_ehr.dataset))
        n2 = len(tv_ehr.dataset.tables.obs)
        report = cls.report(report, table='obs', column=None, value_type='count',
                            operation='filter', before=n1, after=n2)
        return tv_ehr, report


class ObsAdaptiveScaler(TrainableTransformation):
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        config = tv_ehr.config.numerical_processors.scalers.obs
        value_column = lambda x: x.config.tables.obs.value_alias
        scaler = AdaptiveScaler(table=lambda x: x.tables.obs,
                                code_column=lambda x: x.config.tables.obs.code_alias,
                                value_column=value_column,
                                use_float16=config.use_float16).fit(tv_ehr.dataset,
                                                                    cls.get_admission_ids(tv_ehr))
        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.scalers.obs, tv_ehr, scaler)
        report = cls.report(report,
                            table='obs', column=None, value_type='type',
                            operation='TVxEHR.numerical_processors.scalers.obs <- AdaptiveScaler',
                            after=type(scaler))

        dtype1 = tv_ehr.dataset.tables.obs[value_column(tv_ehr.dataset)].dtype
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, scaler(tv_ehr.dataset))
        dtype2 = tv_ehr.dataset.tables.obs[value_column(tv_ehr.dataset)].dtype
        report = cls.report(report, table='obs', column=value_column(tv_ehr.dataset),
                            value_type='dtype',
                            operation=f'scaled_and_maybe_cast_{scaler.use_float16}',
                            before=dtype1, after=dtype2)
        return tv_ehr, report


class InputScaler(TrainableTransformation):
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        code_column = lambda x: x.config.tables.icu_inputs.code_alias
        value_column = lambda x: x.config.tables.icu_inputs.derived_normalized_amount_per_hour
        config = tv_ehr.config.numerical_processors.scalers.icu_inputs
        scaler = MaxScaler(table=lambda x: x.tables.icu_inputs,
                           code_column=code_column,
                           value_column=value_column,
                           use_float16=config.use_float16).fit(tv_ehr.dataset, cls.get_admission_ids(tv_ehr))

        tv_ehr = eqx.tree_at(lambda x: x.numerical_processors.scalers.icu_inputs, tv_ehr, scaler)
        report = cls.report(report,
                            table='icu_inputs', column=None, value_type='type',
                            operation='TVxEHR.numerical_processors.scalers.icu_inputs <- MaxScaler',
                            after=type(scaler))

        dtype1 = tv_ehr.dataset.tables.icu_inputs[value_column(tv_ehr.dataset)].dtype
        tv_ehr = eqx.tree_at(lambda x: x.dataset, tv_ehr, scaler(tv_ehr.dataset))
        dtype2 = tv_ehr.dataset.tables.icu_inputs[value_column(tv_ehr.dataset)].dtype
        report = cls.report(report, table='icu_inputs', column=value_column(tv_ehr.dataset),
                            value_type='dtype',
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
    pass


class LeadingObservableExtraction(AbstractTVxTransformation):
    pass


class TVxConcepts(AbstractTVxTransformation):

    @staticmethod
    def _static_info(tvx_ehr: TVxEHR) -> Dict[str, StaticInfo]:
        static = tvx_ehr.dataset.tables.static
        config = tvx_ehr.config.demographic_vector
        c_gender = tvx_ehr.dataset.config.tables.static
        c_date_of_birth = tvx_ehr.dataset.config.tables.static.date_of_birth_alias
        c_ethnicity = tvx_ehr.dataset.config.tables.static.race_alias

        gender, ethnicity, dob = {}, {}, {}
        if c_date_of_birth in static.columns or config.age:
            dob = static[c_date_of_birth].to_dict()
        if tvx_ehr.scheme.gender is not None or config.gender:
            gender_m = tvx_ehr.gender_mapper
            gender = {k: gender_m.codeset2vec({c}) for k, c in static[c_gender].to_dict().items()}

        if tvx_ehr.scheme.ethnicity is not None or config.ethnicity:
            ethnicity_m = tvx_ehr.ethnicity_mapper
            ethnicity = {k: ethnicity_m.codeset2vec({c}) for k, c in static[c_ethnicity].to_dict().items()}

        return {subject_id: StaticInfo(date_of_birth=dob.get(subject_id),
                                       ethnicity=ethnicity.get(subject_id),
                                       gender=gender.get(subject_id),
                                       demographic_vector_config=config)
                for subject_id in static.index}

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
        c_admission_id = tvx_ehr.dataset.config.tables.icu_inputs.admission_id_alias
        c_code = tvx_ehr.dataset.config.tables.icu_inputs.code_alias
        c_rate = tvx_ehr.dataset.config.tables.icu_inputs.derived_normalized_amount_per_hour
        c_start_time = tvx_ehr.dataset.config.tables.icu_inputs.start_time_alias
        c_end_time = tvx_ehr.dataset.config.tables.icu_inputs.end_time_alias

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
    def _hosp_procedures(tvx_ehr: TVxEHR) -> Dict[str | Hashable, InpatientInput]:
        return TVxConcepts._procedures(tvx_ehr.dataset.tables.hosp_procedures,
                                       tvx_ehr.dataset.config.tables.hosp_procedures,
                                       tvx_ehr.hosp_procedures_mapper)

    @staticmethod
    def _icu_procedures(tvx_ehr: TVxEHR) -> Dict[str | Hashable, InpatientInput]:
        return TVxConcepts._procedures(tvx_ehr.dataset.tables.icu_procedures,
                                       tvx_ehr.dataset.config.tables.icu_procedures,
                                       tvx_ehr.icu_procedures_mapper)

    @classmethod
    def _interventions(cls, tvx_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        Dict[str, InpatientInterventions], Tuple[TVxReportAttributes, ...]]:
        concept_path = TVxReportAttributes.inpatient_input_prefix
        hosp_procedures = cls._hosp_procedures(tvx_ehr)
        report = cls.report(report, tvx_concept=concept_path('hosp_procedures'),
                            table='hosp_procedures',
                            column=None, value_type='count', operation='extract',
                            after=len(hosp_procedures))

        icu_procedures = cls._icu_procedures(tvx_ehr)
        report = cls.report(report, tvx_concept=concept_path('icu_procedures'),
                            table='icu_procedures',
                            column=None, value_type='count', operation='extract',
                            after=len(icu_procedures))

        icu_inputs = cls._icu_inputs(tvx_ehr)
        report = cls.report(report, tvx_concept=concept_path('icu_inputs'), table='icu_inputs',
                            column=None, value_type='count', operation='extract',
                            after=len(icu_inputs))
        interventions = {admission_id: InpatientInterventions(hosp_procedures=hosp_procedures.get(admission_id),
                                                              icu_procedures=icu_procedures.get(admission_id),
                                                              icu_inputs=icu_inputs.get(admission_id))
                         for admission_id in tvx_ehr.admission_ids}
        return interventions, report

    @staticmethod
    def _observables(tvx_ehr: TVxEHR) -> Dict[str, InpatientObservables]:
        c_admission_id = tvx_ehr.dataset.config.tables.obs.admission_id_alias
        c_code = tvx_ehr.dataset.config.tables.obs.code_alias
        c_value = tvx_ehr.dataset.config.tables.obs.value_alias
        c_timestamp = tvx_ehr.dataset.config.tables.obs.time_alias

        # For dasking, we index by admission.
        table = tvx_ehr.dataset.tables.obs.set_index(c_admission_id)
        table[c_code] = table[c_code].map(tvx_ehr.scheme.obs.index)
        assert not table[c_code].isnull().any(), 'Some codes are not in the target scheme.'
        obs_dim = len(tvx_ehr.scheme.obs)

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
        return inpatient_observables_df.to_dict()

    @classmethod
    def apply(cls, tvx_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        subject_admissions = tvx_ehr.subjects_sorted_admission_ids
        static_info = cls._static_info(tvx_ehr)
        dx_discharge = cls._dx_discharge(tvx_ehr)
        dx_discharge_history = cls._dx_discharge_history(tvx_ehr, dx_discharge)
        outcome = cls._outcome(tvx_ehr, dx_discharge)
        if tvx_ehr.config.interventions:
            interventions, report = cls._interventions(tvx_ehr, report)
        else:
            interventions = None
        if tvx_ehr.config.observables:
            observables = cls._observables(tvx_ehr)
        else:
            observables = None

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

#

# def adm_extractor(self, subject_ids):
#     c_subject_id = self.colname["adm"].subject_id
#     df = self.df["adm"]
#     df = df[df[c_subject_id].isin(subject_ids)]
#     return {
#         subject_id: subject_df.index.tolist()
#         for subject_id, subject_df in df.groupby(c_subject_id)
#     }
