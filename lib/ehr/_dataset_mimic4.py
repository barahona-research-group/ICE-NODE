"""."""
from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict, Callable, List, Optional

import dask.dataframe as dd
import equinox as eqx
import numpy as np
import pandas as pd

from .coding_scheme import OutcomeExtractor, CodingScheme
from .concepts import (InpatientInput, InpatientObservables, Patient,
                       Admission, DemographicVectorConfig,
                       AggregateRepresentation, InpatientInterventions,
                       LeadingObservableExtractorConfig, StaticInfo)
from .dataset import (DatasetScheme, ColumnNames, DatasetConfig,
                      DatasetSchemeConfig)
from ._dataset_mimic3 import MIMIC3Dataset, try_compute
from ..base import Config
from ..utils import load_config

warnings.filterwarnings('error',
                        category=RuntimeWarning,
                        message=r'overflow encountered in cast')


class OutlierRemover(eqx.Module):

    c_value: str
    c_code_index: str
    min_val: pd.Series
    max_val: pd.Series

    def __call__(self, df):
        min_val = df[self.c_code_index].map(self.min_val)
        max_val = df[self.c_code_index].map(self.max_val)
        df = df[df[self.c_value].between(min_val, max_val)]
        return df


class ZScoreScaler(eqx.Module):
    """Standardizes the values of a column in a DataFrame by subtracting the mean and scaling to unit variance.

    Attributes:
        c_value: Name of column to standardize.
        c_code_index: Name of column with codes to map stats to.
        mean: Mapping of codes to mean values for standardization.
        std: Mapping of codes to std dev values for standardization.
        use_float16: Whether to convert standardized values to float16.

    Methods:
        __call__: Performs standardization on the DataFrame passed.
        unscale: Reverts standardized arrays back to original scale.
        unscale_code: Reverts standardized arrays back to scale for a specific code.
    """

    c_value: str
    c_code_index: str
    mean: pd.Series
    std: pd.Series
    use_float16: bool = True

    @property
    def original_dtype(self):
        return self.mean.dtype

    def __call__(self, df):
        mean = df[self.c_code_index].map(self.mean)
        std = df[self.c_code_index].map(self.std)
        df.loc[:, self.c_value] = (df[self.c_value] - mean) / std
        if self.use_float16:
            df = df.astype({self.c_value: np.float16})
        return df

    def unscale(self, array):
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        return array * self.std.loc[index].values + self.mean.loc[index].values

    def unscale_code(self, array, code_index):
        array = array.astype(self.original_dtype)
        return array * self.std.loc[code_index] + self.mean.loc[code_index]


class MaxScaler(eqx.Module):
    c_value: str
    c_code_index: str
    max_val: pd.Series
    use_float16: bool = True

    @property
    def original_dtype(self):
        return self.max_val.dtype

    def __call__(self, df):
        max_val = df[self.c_code_index].map(self.max_val)
        df.loc[:, self.c_value] = (df[self.c_value] / max_val)
        if self.use_float16:
            df = df.astype({self.c_value: np.float16})
        return df

    def unscale(self, array):
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

    def unscale_code(self, array, code_index):
        array = array.astype(self.original_dtype)
        return array * self.max_val.loc[code_index]


class AdaptiveScaler(eqx.Module):
    c_value: str
    c_code_index: str
    max_val: pd.Series
    min_val: pd.Series
    mean: pd.Series
    std: pd.Series
    use_float16: bool = True

    @property
    def original_dtype(self):
        return self.max_val.dtype

    def __call__(self, df):
        min_val = df[self.c_code_index].map(self.min_val)
        max_val = df[self.c_code_index].map(self.max_val)
        mean = df[self.c_code_index].map(self.mean)
        std = df[self.c_code_index].map(self.std)

        minmax_scaled = (df[self.c_value] - min_val) / max_val
        z_scaled = ((df[self.c_value] - mean) / std)

        df.loc[:, self.c_value] = np.where(min_val >= 0.0, minmax_scaled,
                                           z_scaled)
        if self.use_float16:
            df = df.astype({self.c_value: np.float16})
        return df

    def unscale(self, array):
        array = array.astype(self.original_dtype)
        index = np.arange(array.shape[-1])
        mu = self.mean.loc[index].values
        sigma = self.std.loc[index].values
        min_val = self.min_val.loc[index].values
        max_val = self.max_val.loc[index].values
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)

    def unscale_code(self, array, code_index):
        array = array.astype(self.original_dtype)
        mu = self.mean.loc[code_index]
        sigma = self.std.loc[code_index]
        min_val = self.min_val.loc[code_index]
        max_val = self.max_val.loc[code_index]
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)


class MIMIC4DatasetScheme(DatasetScheme):
    dx: Union[Dict[str, CodingScheme], CodingScheme]

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        if isinstance(self.config.dx, dict):
            self.dx = {
                version: CodingScheme.from_name(scheme)
                for version, scheme in self.config.dx.items()
            }

    @classmethod
    def _assert_valid_maps(cls, source, target):
        attrs = list(k for k in source.scheme_dict if k != 'dx')
        for attr in attrs:
            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"
        for version, s_scheme in source.dx.items():
            t_scheme = target.dx
            assert s_scheme.mapper_to(
                t_scheme), f"Cannot map dx (version={version}) \
                from {s_scheme} to {t_scheme}"

    def dx_mapper(self, target_scheme: DatasetScheme):
        return {
            version: s_dx.mapper_to(target_scheme.dx.name)
            for version, s_dx in self.dx.items()
        }

    @property
    def supported_target_scheme_options(self):
        supproted_attr_targets = {
            k: (getattr(self, k).__class__.__name__,) +
               getattr(self, k).supported_targets
            for k in self.scheme_dict
        }
        supported_dx_targets = {
            version: (scheme.__class__.__name__,) + scheme.supported_targets
            for version, scheme in self.dx.items()
        }
        supproted_attr_targets['dx'] = list(
            set.intersection(*map(set, supported_dx_targets.values())))
        supported_outcomes = {
            version: OutcomeExtractor.supported_outcomes(scheme)
            for version, scheme in self.dx.items()
        }
        supproted_attr_targets['outcome'] = list(
            set.intersection(*map(set, supported_outcomes.values())))

        return supproted_attr_targets


class MIMIC4ICUDatasetSchemeConfig(DatasetSchemeConfig):
    int_proc: str = 'int_mimic4_proc' # -> 'int_mimic4_grouped_proc'
    int_input: str = 'int_mimic4_input' # -> 'int_mimic4_input_group'
    obs: str = 'mimic4_obs'


class MIMIC4ICUDatasetScheme(MIMIC4DatasetScheme):
    int_proc: CodingScheme
    int_input: CodingScheme
    obs: CodingScheme

    def make_target_scheme_config(self, **kwargs):
        assert 'outcome' in kwargs, "Outcome must be specified"
        return self.config.update(int_proc='int_mimic4_grouped_proc',
                                  int_input='int_mimic4_input_group',
                                  **kwargs)


class MIMIC4Dataset(MIMIC3Dataset):
    scheme: MIMIC4DatasetScheme

    def _dx_fix_icd_dots(self):
        c_code = self.colname["dx"].code
        c_version = self.colname["dx"].version
        df = self.df["dx"]
        df = df.assign(**{c_code: df[c_code].str.strip()})
        self.df['dx'] = df
        for version, scheme in self.scheme.dx.items():
            if isinstance(scheme, CodingScheme):
                ver_mask = df[c_version] == version
                code_col = df[c_code]
                df = df.assign(**{
                    c_code:
                        code_col.mask(ver_mask, code_col.map(scheme.add_dots))
                })
        self.df["dx"] = df

    def _dx_filter_unsupported_icd(self):
        c_code = self.colname["dx"].code
        c_version = self.colname["dx"].version
        self.df["dx"] = self._validate_dx_codes(self.df["dx"], c_code,
                                                c_version, self.scheme.dx)

    @staticmethod
    def _validate_dx_codes(df, code_col, version_col, version_map):
        filtered_df = []
        groups = df.groupby(version_col)
        for version in groups[version_col].unique():
            version_df = groups.get_group(version[0])
            codeset = set(version_df[code_col])
            scheme = version_map[str(version[0])]
            scheme_codes = set(scheme.codes)

            unrecognised = codeset - scheme_codes
            if len(unrecognised) > 0:
                logging.debug(
                    f'Unrecognised ICD v{version} codes: {len(unrecognised)} '
                    f'({len(unrecognised) / len(codeset):.2%})')
                logging.debug(f'Unrecognised {type(scheme)} codes '
                              f'({len(unrecognised)}) '
                              f'to be removed (first 30): '
                              f'{sorted(unrecognised)[:30]}')

            filtered_df.append(
                version_df[version_df[code_col].isin(scheme_codes)])

        return dd.concat(filtered_df).reset_index(drop=True)

    def subject_info_extractor(self, subject_ids, target_scheme):

        static_df = self.df['static']
        c_gender = self.colname["static"].gender
        c_anchor_year = self.colname["static"].anchor_year
        c_anchor_age = self.colname["static"].anchor_age
        c_eth = self.colname["static"].ethnicity

        static_df = static_df.loc[subject_ids]
        gender = static_df[c_gender].map(self.scheme.gender.codeset2vec)
        subject_gender = gender.to_dict()

        anchor_date = pd.to_datetime(static_df[c_anchor_year],
                                     format='%Y').dt.normalize()
        anchor_age = static_df[c_anchor_age].map(
            lambda y: pd.DateOffset(years=-y))
        dob = anchor_date + anchor_age
        subject_dob = dict(zip(static_df.index.values, dob))
        subject_eth = dict()
        eth_mapper = self.scheme.ethnicity_mapper(target_scheme)
        for subject_id in static_df.index.values:
            eth_code = eth_mapper.map_codeset(
                [static_df.loc[subject_id, c_eth]])
            subject_eth[subject_id] = eth_mapper.codeset2vec(eth_code)

        return subject_dob, subject_gender, subject_eth

    def dx_codes_extractor(self, admission_ids_list, target_scheme):
        c_adm_id = self.colname["dx"].admission_id
        c_code = self.colname["dx"].code
        c_version = self.colname["dx"].version

        df = self.df["dx"]
        df = df[df[c_adm_id].isin(admission_ids_list)]
        codes_df = {
            adm_id: codes_df
            for adm_id, codes_df in df.groupby(c_adm_id)
        }
        empty_vector = target_scheme.dx.empty_vector()

        dx_mapper = self.scheme.dx_mapper(target_scheme)

        def _extract_codes(adm_id):
            _codes_df = codes_df.get(adm_id)
            if _codes_df is None:
                return (adm_id, empty_vector)

            vec = empty_vector
            for version, version_df in _codes_df.groupby(c_version):
                mapper = dx_mapper[str(version)]
                codeset = mapper.map_codeset(version_df[c_code])
                vec = vec.union(mapper.codeset2vec(codeset))
            return (adm_id, vec)

        return map(_extract_codes, admission_ids_list)


class MIMIC4ICUDataset(MIMIC4Dataset):
    scheme: MIMIC4ICUDatasetScheme
    scalers_history: Dict[str, Callable]
    outlier_remover_history: Dict[str, Callable]

    def __init__(self, config: DatasetConfig = None, config_path: str = None):
        if config is None:
            config = Config.from_dict(load_config(config_path))

        eqx.Module.__init__(self)
        self.config = config
        self.scheme = DatasetScheme.import_module(config.scheme,
                                                  config.scheme_classname)
        self.colname = {
            f: ColumnNames.make(m)
            for f, m in config.colname.items()
        }
        self._load_dataframes()

        self.scalers_history = dict()
        self.outlier_remover_history = dict()

    def _int_input_remove_subjects_with_nans(self):
        c_subject = self.colname["adm"].subject_id
        c_adm_id = self.colname["int_input"].admission_id
        c_rate = self.colname["int_input"].rate
        adm_df = self.df["adm"]
        inp_df = self.df["int_input"]

        nan_input_rates = inp_df[c_rate].isnull()
        nan_adm_ids = inp_df[nan_input_rates][c_adm_id].unique()
        nan_subj_ids = adm_df[adm_df.index.isin(
            nan_adm_ids)][c_subject].unique()
        logging.debug("Removing subjects with at least "
                      f"one nan input rate: {len(nan_subj_ids)}")
        self.df["adm"] = adm_df[~adm_df[c_subject].isin(nan_subj_ids)]

    @staticmethod
    def _add_code_source_index(df, source_scheme, colname):
        c_code = colname.code
        colname = colname._replace(code_source_index="code_source_index")
        df = df.assign(
            code_source_index=df[c_code].map(source_scheme.index).astype(int))
        return df, colname

    @classmethod
    def _validate_time_betweenness(cls, df, df_name, colname, time_cols):
        c_admittime = colname["adm"].admittime
        c_dischtime = colname["adm"].dischtime
        c_adm_id = colname[df_name].admission_id

        adm_df = df['adm'][[c_admittime, c_dischtime]]
        df = df[df_name]
        df = df.merge(adm_df, left_on=c_adm_id, right_index=True, how='left')
        for time_col in time_cols:
            col = colname[df_name][time_col]
            assert df[col].between(df[c_admittime], df[c_dischtime]).all(), \
                f"Time not between admission and discharge in {df_name} " \
 \
    @staticmethod
    def _set_relative_times(df_dict, colname, df_name, time_cols,
                            seconds_scaler):
        c_admittime = colname["adm"].admittime
        c_adm_id = colname[df_name].admission_id

        target_df = df_dict[df_name]
        adm_df = df_dict["adm"][[c_admittime]]
        df = target_df.merge(adm_df,
                             left_on=c_adm_id,
                             right_index=True,
                             how='left')
        df_colname = colname[df_name]
        for time_col in time_cols:
            col = df_colname[time_col]
            delta = df[col] - df[c_admittime]

            target_df = target_df.assign(
                **{
                    col: (delta.dt.total_seconds() *
                          seconds_scaler).astype(np.float32)
                })

        df_dict[df_name] = target_df

    def _fit_obs_outlier_remover(self, admission_ids, outlier_q1, outlier_q2,
                                 outlier_iqr_scale, outlier_z1, outlier_z2):
        c_code_index = self.colname["obs"].code_source_index
        c_value = self.colname["obs"].value
        c_adm_id = self.colname["obs"].admission_id
        df = self.df['obs'][[c_code_index, c_value, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]
        outlier_q = np.array([outlier_q1, outlier_q2])
        q = df.groupby(c_code_index).apply(
            lambda x: x[c_value].quantile(outlier_q))
        q.columns = ['q1', 'q2']
        q['iqr'] = q['q2'] - q['q1']
        q['out_q1'] = q['q1'] - outlier_iqr_scale * q['iqr']
        q['out_q2'] = q['q2'] + outlier_iqr_scale * q['iqr']

        stat = df.groupby(c_code_index).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std()
            }))
        stat['out_z1'] = stat['mu'] - outlier_z1 * stat['sigma']
        stat['out_z2'] = stat['mu'] + outlier_z2 * stat['sigma']

        min_val = np.minimum(q['out_q1'], stat['out_z1'])
        max_val = np.maximum(q['out_q2'], stat['out_z2'])

        return OutlierRemover(c_value=c_value,
                              c_code_index=c_code_index,
                              min_val=min_val,
                              max_val=max_val)

    def _fit_obs_scaler(self, admission_ids):
        c_code_index = self.colname["obs"].code_source_index
        c_value = self.colname["obs"].value
        c_adm_id = self.colname["obs"].admission_id
        df = self.df['obs'][[c_code_index, c_value, c_adm_id]]
        df = df[df[c_adm_id].isin(admission_ids)]
        stat = df.groupby(c_code_index).apply(
            lambda x: pd.Series({
                'mu': x[c_value].mean(),
                'sigma': x[c_value].std(),
                'min': x[c_value].min(),
                'max': x[c_value].max()
            }))
        return AdaptiveScaler(c_value=c_value,
                              c_code_index=c_code_index,
                              mean=stat['mu'],
                              std=stat['sigma'],
                              min_val=stat['min'],
                              max_val=stat['max'])

    def _fit_int_input_scaler(self, admission_ids):
        c_adm_id = self.colname["int_input"].admission_id
        c_code_index = self.colname["int_input"].code_source_index
        c_rate = self.colname["int_input"].rate
        df = self.df["int_input"][[c_adm_id, c_code_index, c_rate]]
        df = df[df[c_adm_id].isin(admission_ids)]
        return MaxScaler(c_value=c_rate,
                         c_code_index=c_code_index,
                         max_val=df.groupby(c_code_index).max()[c_rate])

    def fit_outlier_remover(self,
                            subject_ids: Optional[List[int]] = None,
                            outlier_q1=0.25,
                            outlier_q2=0.75,
                            outlier_iqr_scale=1.5,
                            outlier_z1=-2.5,
                            outlier_z2=2.5):

        c_subject = self.colname["adm"].subject_id
        adm_df = self.df["adm"]
        if subject_ids is None:
            train_adms = adm_df.index
        else:
            train_adms = adm_df[adm_df[c_subject].isin(subject_ids)].index

        return {
            'obs':
                self._fit_obs_outlier_remover(admission_ids=train_adms,
                                              outlier_q1=outlier_q1,
                                              outlier_q2=outlier_q2,
                                              outlier_iqr_scale=outlier_iqr_scale,
                                              outlier_z1=outlier_z1,
                                              outlier_z2=outlier_z2)
        }

    def fit_scalers(self, subject_ids: Optional[List[int]] = None):
        c_subject = self.colname["adm"].subject_id
        adm_df = self.df["adm"]
        if subject_ids is None:
            train_adms = adm_df.index
        else:
            train_adms = adm_df[adm_df[c_subject].isin(subject_ids)].index

        return {
            'obs': self._fit_obs_scaler(admission_ids=train_adms),
            'int_input': self._fit_int_input_scaler(admission_ids=train_adms)
        }

    def remove_outliers(self, outlier_remover):
        assert len(self.outlier_remover_history) == 0, \
            "Outlier remover can only be applied once."
        df = self.df.copy()
        history = outlier_remover
        for df_name, remover in outlier_remover.items():
            n1 = len(df[df_name])
            df[df_name] = remover(df[df_name])
            n2 = len(df[df_name])
            logging.debug(f'Removed {n1 - n2} ({(n1 - n2) / n2 :0.3f}) '
                          f'outliers from {df_name}')

        updated = eqx.tree_at(lambda x: x.df, self, df)
        updated = eqx.tree_at(lambda x: x.outlier_remover_history, updated,
                              history)
        return updated

    def apply_scalers(self, scalers):
        assert len(self.scalers_history) == 0, \
            "Scalers can only be applied once."
        df = self.df.copy()
        history = scalers
        for df_name, scaler in scalers.items():
            df[df_name] = scaler(df[df_name])

        updated = eqx.tree_at(lambda x: x.df, self, df)
        updated = eqx.tree_at(lambda x: x.scalers_history, updated, history)
        return updated

    def _load_dataframes(self):

        MIMIC3Dataset._load_dataframes(self)

        colname = self.colname.copy()
        scheme = self.scheme
        df = self.df

        logging.debug("Time casting..")
        # Cast timestamps for intervensions
        for time_col in ("start_time", "end_time"):
            for file in ("int_proc", "int_input"):
                col = colname[file][time_col]
                df[file][col] = pd.to_datetime(df[file][col])

        # Cast timestamps for observables
        col = colname["obs"].timestamp
        df["obs"][col] = pd.to_datetime(df["obs"][col])

        self._validate_time_betweenness(df, "obs", colname, ["timestamp"])
        self._validate_time_betweenness(df, "int_proc", colname,
                                        ["start_time", "end_time"])
        self._validate_time_betweenness(df, "int_input", colname,
                                        ["start_time", "end_time"])

        logging.debug("[DONE] Time casting..")

        logging.debug("Dataframes validation and time conversion")

        def _filter_codes(df, c_code, source_scheme):
            mask = df[c_code].isin(source_scheme.codes)
            logging.debug(f'Removed codes: {df[~mask][c_code].unique()}')
            return df[mask]

        for name in ("int_proc", "int_input", "obs"):
            df[name] = _filter_codes(df[name], colname[name].code,
                                     getattr(scheme, name))

            df[name], colname[name] = self._add_code_source_index(
                df[name], getattr(scheme, name), colname[name])

        # self._adm_remove_subjects_with_overlapping_admissions(
        #     self.df["adm"], self.colname["adm"])
        self._int_input_remove_subjects_with_nans()
        self._set_relative_times(df, colname, "int_proc",
                                 ["start_time", "end_time"],
                                 self.seconds_scaler)
        self._set_relative_times(df, colname, "int_input",
                                 ["start_time", "end_time"],
                                 self.seconds_scaler)
        self._set_relative_times(df, colname, "obs", ["timestamp"],
                                 self.seconds_scaler)
        self.df = {k: try_compute(v) for k, v in df.items()}
        self._match_admissions_with_demographics(self.df, colname)
        self.colname = colname
        logging.debug("[DONE] Dataframes validation and time conversion")

    def to_subjects(self,
                    subject_ids: List[int],
                    num_workers: int,
                    demographic_vector_config: DemographicVectorConfig,
                    leading_observable_config: LeadingObservableConfig,
                    target_scheme: MIMIC4ICUDatasetScheme,
                    time_binning: Optional[int] = None,
                    **kwargs):

        subject_dob, subject_gender, subject_eth = self.subject_info_extractor(
            subject_ids, target_scheme)
        admission_ids = self.adm_extractor(subject_ids)
        adm_ids_list = sum(map(list, admission_ids.values()), [])
        logging.debug('Extracting dx codes...')
        dx_codes = dict(self.dx_codes_extractor(adm_ids_list, target_scheme))
        logging.debug('[DONE] Extracting dx codes')
        logging.debug('Extracting dx codes history...')
        dx_codes_history = dict(
            self.dx_codes_history_extractor(dx_codes, admission_ids,
                                            target_scheme))
        logging.debug('[DONE] Extracting dx codes history')
        logging.debug('Extracting outcome...')
        outcome = dict(self.outcome_extractor(dx_codes, target_scheme))
        logging.debug('[DONE] Extracting outcome')
        logging.debug('Extracting procedures...')
        procedures = dict(self.procedure_extractor(adm_ids_list))
        logging.debug('[DONE] Extracting procedures')
        logging.debug('Extracting inputs...')
        inputs = dict(self.inputs_extractor(adm_ids_list))
        logging.debug('[DONE] Extracting inputs')
        logging.debug('Extracting observables...')
        observables = dict(
            self.observables_extractor(adm_ids_list, num_workers))

        if time_binning is not None:
            observables = dict((k, v.time_binning(time_binning))
                               for k, v in observables.items())

        logging.debug('[DONE] Extracting observables')

        logging.debug('Compiling admissions...')
        c_admittime = self.colname['adm'].admittime
        c_dischtime = self.colname['adm'].dischtime
        c_adm_interval = self.colname['adm'].adm_interval
        adf = self.df['adm']
        adm_dates = dict(
            zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))
        adm_interval = dict(zip(adf.index, adf[c_adm_interval]))
        proc_repr = AggregateRepresentation(self.scheme.int_proc,
                                            target_scheme.int_proc)

        def gen_admission(i):
            interventions = InpatientInterventions(
                proc=procedures[i],
                input_=inputs[i],
                adm_interval=adm_interval[i])

            obs = observables[i]
            lead_obs = obs.make_leading_observable(leading_observable_config)

            if time_binning is None:
                interventions = interventions.segment_proc(proc_repr)
                interventions = interventions.segment_input()
                lead_obs = lead_obs.segment(interventions.t_sep)
                obs = obs.segment(interventions.t_sep)

            return Admission(admission_id=i,
                             admission_dates=adm_dates[i],
                             dx_codes=dx_codes[i],
                             dx_codes_history=dx_codes_history[i],
                             outcome=outcome[i],
                             observables=obs,
                             leading_observable=lead_obs,
                             interventions=interventions)

        def _gen_subject(subject_id):

            _admission_ids = admission_ids[subject_id]
            # for subject_id, subject_admission_ids in admission_ids.items():
            _admission_ids = sorted(_admission_ids,
                                    key=lambda aid: adm_dates[aid][0])

            static_info = StaticInfo(
                date_of_birth=subject_dob[subject_id],
                gender=subject_gender[subject_id],
                ethnicity=subject_eth[subject_id],
                demographic_vector_config=demographic_vector_config)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                admissions = list(executor.map(gen_admission, _admission_ids))
            return Patient(subject_id=subject_id,
                           admissions=admissions,
                           static_info=static_info)

        return list(map(_gen_subject, subject_ids))

    def procedure_extractor(self, admission_ids_list):
        c_adm_id = self.colname["int_proc"].admission_id
        c_code_index = self.colname["int_proc"].code_source_index
        c_start_time = self.colname["int_proc"].start_time
        c_end_time = self.colname["int_proc"].end_time
        df = self.df["int_proc"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        def group_fun(x):
            return pd.Series({
                0: x[c_code_index].to_numpy(),
                1: x[c_start_time].to_numpy(),
                2: x[c_end_time].to_numpy()
            })

        grouped = df.groupby(c_adm_id).apply(group_fun)
        adm_arr = grouped.index.tolist()
        input_size = len(self.scheme.int_proc)
        for i in adm_arr:
            yield (i,
                   InpatientInput(index=grouped.loc[i, 0],
                                  rate=np.ones_like(grouped.loc[i, 0],
                                                    dtype=bool),
                                  starttime=grouped.loc[i, 1],
                                  endtime=grouped.loc[i, 2],
                                  size=input_size))

        for adm_id in set(admission_ids_list) - set(adm_arr):
            yield (adm_id, InpatientInput.empty(input_size))

    def inputs_extractor(self, admission_ids_list):
        c_adm_id = self.colname["int_input"].admission_id
        c_start_time = self.colname["int_input"].start_time
        c_end_time = self.colname["int_input"].end_time
        c_rate = self.colname["int_input"].rate
        c_code_index = self.colname["int_input"].code_source_index

        df = self.df["int_input"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        def group_fun(x):
            return pd.Series({
                0: x[c_code_index].to_numpy(),
                1: x[c_rate].to_numpy(),
                2: x[c_start_time].to_numpy(),
                3: x[c_end_time].to_numpy()
            })

        grouped = df.groupby(c_adm_id).apply(group_fun)
        adm_arr = grouped.index.tolist()
        input_size = len(self.scheme.int_input)
        for i in adm_arr:
            yield (i,
                   InpatientInput(index=grouped.loc[i, 0],
                                  rate=grouped.loc[i, 1],
                                  starttime=grouped.loc[i, 2],
                                  endtime=grouped.loc[i, 3],
                                  size=input_size))
        for adm_id in set(admission_ids_list) - set(adm_arr):
            yield (adm_id, InpatientInput.empty(input_size))

    def observables_extractor(self, admission_ids_list, num_workers):
        c_adm_id = self.colname["obs"].admission_id
        c_time = self.colname["obs"].timestamp
        c_value = self.colname["obs"].value
        c_code_index = self.colname["obs"].code_source_index

        df = self.df["obs"][[c_adm_id, c_time, c_value, c_code_index]]
        logging.debug("obs: filter adms")
        df = df[df[c_adm_id].isin(admission_ids_list)]

        obs_dim = len(self.scheme.obs)

        def ret_put(a, *args):
            np.put(a, *args)
            return a

        def val_mask(x):
            idx = x[c_code_index]
            val = ret_put(np.zeros(obs_dim, dtype=np.float16), idx, x[c_value])
            mask = ret_put(np.zeros(obs_dim, dtype=bool), idx, 1.0)
            adm_id = x.index[0]
            time = x[c_time].iloc[0]
            return pd.Series({0: adm_id, 1: time, 2: val, 3: mask})

        def gen_observation(val_mask):
            time = val_mask[1].to_numpy()
            value = val_mask[2]
            mask = val_mask[3]
            mask = np.vstack(mask.values).reshape((len(time), obs_dim))
            value = np.vstack(value.values).reshape((len(time), obs_dim))
            return InpatientObservables(time=time, value=value, mask=mask)

        def partition_fun(part_df):
            g = part_df.groupby([c_adm_id, c_time], sort=True, as_index=False)
            return g.apply(val_mask).groupby(0).apply(gen_observation)

        logging.debug("obs: dasking")
        df = df.set_index(c_adm_id)
        df = dd.from_pandas(df, npartitions=12, sort=True)
        logging.debug("obs: groupby")
        obs_obj_df = df.map_partitions(partition_fun, meta=(None, object))
        logging.debug("obs: undasking")
        obs_obj_df = obs_obj_df.compute()
        logging.debug("obs: extract")

        collected_adm_ids = obs_obj_df.index.tolist()
        assert len(collected_adm_ids) == len(set(collected_adm_ids)), \
            "Duplicate admission ids in obs"

        for adm_id, obs in obs_obj_df.items():
            yield (adm_id, obs)

        logging.debug("obs: empty")
        for adm_id in set(admission_ids_list) - set(obs_obj_df.index):
            yield (adm_id, InpatientObservables.empty(obs_dim))
