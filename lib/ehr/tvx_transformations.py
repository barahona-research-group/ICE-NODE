from __future__ import annotations

import random
from abc import abstractmethod
from dataclasses import field
from typing import Dict, Tuple, List, Callable, Final, Type, Set

import equinox as eqx
import numpy as np
import pandas as pd

from . import TVxEHR, StaticInfo
from .dataset import Dataset, TransformationsDependency, AbstractTransformation
from .transformations import FilterUnsupportedCodes, CastTimestamps, SetIndex, DatasetTransformation
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


class SetCodeIntegerIndices(AbstractTVxTransformation):
    @classmethod
    def apply(cls, tv_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        TVxEHR, Tuple[TVxReportAttributes, ...]]:
        tables_dict = tv_ehr.dataset.tables.tables_dict
        for table_name, code_column in tv_ehr.dataset.config.tables.code_column.items():
            table = tables_dict[table_name]
            coding_scheme = getattr(tv_ehr.dataset.scheme, table_name)
            dtype1 = table[code_column].dtype
            n1 = len(table)
            table = table.assign(**{code_column: table[code_column].map(coding_scheme.index)})
            table = table[table[code_column].notnull()].astype({code_column: int})
            dtype2 = table[code_column].dtype
            n2 = len(table)
            report = cls.report(report, table=table_name, column=code_column, before=n1, after=n2, value_type='count',
                                operation='filter_unsupported_codes')
            report = cls.report(report, table=table_name, column=code_column, before=dtype1, after=dtype2,
                                value_type='dtype',
                                operation='code_integer_index')

            tv_ehr = eqx.tree_at(lambda x: getattr(x.dataset.tables, table_name), tv_ehr, table)
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
    SetCodeIntegerIndices: {FilterUnsupportedCodes},
    RandomSplits: {SetIndex, CastTimestamps},
    TrainableTransformation: {RandomSplits, SetIndex, SetCodeIntegerIndices},
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

class TVxConcepts(AbstractTVxTransformation):

    @classmethod
    def _static_info_extractor(cls, tvx_ehr: TVxEHR, report: Tuple[TVxReportAttributes, ...]) -> Tuple[
        Dict[str, StaticInfo], Tuple[TVxReportAttributes, ...]]:
        static = tvx_ehr.dataset.tables.static

        config = tvx_ehr.config.demographic_vector
        c_gender = tvx_ehr.dataset.config.tables.static
        c_date_of_birth = tvx_ehr.dataset.config.tables.static.date_of_birth_alias



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
    # def subject_info_extractor(self, subject_ids: List[int],
    #                            target_scheme: DatasetScheme):
    #     """
    #     Important comment from MIMIC-III documentation at \
    #         https://mimic.mit.edu/docs/iii/tables/patients/
    #     > DOB is the date of birth of the given patient. Patients who are \
    #         older than 89 years old at any time in the database have had their\
    #         date of birth shifted to obscure their age and comply with HIPAA.\
    #         The shift process was as follows: the patient’s age at their \
    #         first admission was determined. The date of birth was then set to\
    #         exactly 300 years before their first admission.
    #     """
    #     assert self.scheme.gender is target_scheme.gender, (
    #         "No conversion assumed for gender attribute")
    #
    #     c_gender = self.colname["static"].gender
    #     c_eth = self.colname["static"].ethnicity
    #     c_dob = self.colname["static"].date_of_birth
    #
    #     c_admittime = self.colname["adm"].admittime
    #     c_dischtime = self.colname["adm"].dischtime
    #     c_subject_id = self.colname["adm"].subject_id
    #
    #     adm_df = self.df['adm'][self.df['adm'][c_subject_id].isin(subject_ids)]
    #
    #     df = self.df['static'].copy()
    #     df = df.loc[subject_ids]
    #     gender = df[c_gender].map(self.scheme.gender.codeset2vec)
    #
    #     subject_gender = gender.to_dict()
    #
    #     df[c_dob] = pd.to_datetime(df[c_dob])
    #     last_disch_date = adm_df.groupby(c_subject_id)[c_dischtime].max()
    #     first_adm_date = adm_df.groupby(c_subject_id)[c_admittime].min()
    #
    #     last_disch_date = last_disch_date.loc[df.index]
    #     first_adm_date = first_adm_date.loc[df.index]
    #     uncertainty = (last_disch_date.dt.year - first_adm_date.dt.year) // 2
    #     shift = (uncertainty + 89).astype('timedelta64[Y]')
    #     df.loc[:, c_dob] = df[c_dob].mask(
    #         (last_disch_date.dt.year - df[c_dob].dt.year) > 150,
    #         first_adm_date - shift)
    #
    #     subject_dob = df[c_dob].dt.normalize().to_dict()
    #     # TODO: check https://mimic.mit.edu/docs/iii/about/time/
    #     eth_mapper = self.scheme.ethnicity_mapper(target_scheme)
    #
    #     def eth2vec(eth):
    #         code = eth_mapper.map_codeset(eth)
    #         return eth_mapper.codeset2vec(code)
    #
    #     subject_eth = df[c_eth].map(eth2vec).to_dict()
    #
    #     return subject_dob, subject_gender, subject_eth
    #
    # def adm_extractor(self, subject_ids):
    #     c_subject_id = self.colname["adm"].subject_id
    #     df = self.df["adm"]
    #     df = df[df[c_subject_id].isin(subject_ids)]
    #     return {
    #         subject_id: subject_df.index.tolist()
    #         for subject_id, subject_df in df.groupby(c_subject_id)
    #     }
    #
    # def dx_codes_extractor(self, admission_ids_list,
    #                        target_scheme: DatasetScheme):
    #     c_adm_id = self.colname["dx_discharge"].admission_id
    #     c_code = self.colname["dx_discharge"].code
    #
    #     df = self.df["dx_discharge"]
    #     df = df[df[c_adm_id].isin(admission_ids_list)]
    #
    #     codes_df = {
    #         adm_id: codes_df
    #         for adm_id, codes_df in df.groupby(c_adm_id)
    #     }
    #     empty_vector = target_scheme.dx_discharge.empty_vector()
    #     mapper = self.scheme.dx_mapper(target_scheme)
    #
    #     def _extract_codes(adm_id):
    #         _codes_df = codes_df.get(adm_id)
    #         if _codes_df is None:
    #             return (adm_id, empty_vector)
    #         codeset = mapper.map_codeset(_codes_df[c_code])
    #         return (adm_id, mapper.codeset2vec(codeset))
    #
    #     return dict(map(_extract_codes, admission_ids_list))
    #
    # def dx_codes_history_extractor(self, dx_codes, admission_ids,
    #                                target_scheme):
    #     for subject_id, subject_admission_ids in admission_ids.items():
    #         _adm_ids = sorted(subject_admission_ids)
    #         vec = target_scheme.dx_discharge.empty_vector()
    #         yield (_adm_ids[0], vec)
    #
    #         for prev_adm_id, adm_id in zip(_adm_ids[:-1], _adm_ids[1:]):
    #             if prev_adm_id in dx_codes:
    #                 vec = vec.union(dx_codes[prev_adm_id])
    #             yield (adm_id, vec)
    #
    # def outcome_extractor(self, dx_codes, target_scheme):
    #     return zip(dx_codes.keys(),
    #                map(target_scheme.outcome.mapcodevector, dx_codes.values()))
