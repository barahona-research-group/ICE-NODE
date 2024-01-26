"""."""
from __future__ import annotations

import logging
import os
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field
from functools import cached_property
from typing import Union, Dict, Callable, List, Optional, Tuple

import dask.dataframe as dd
import equinox as eqx
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import Engine

from ._coding_scheme_icd import (ICD)
from ._dataset_mimic3 import MIMIC3Dataset, try_compute
from .coding_scheme import (OutcomeExtractor, CodingScheme, CodingSchemeConfig, FlatScheme, CodeMap)
from .coding_scheme import _RSC_DIR
from .concepts import (InpatientInput, InpatientObservables, Patient,
                       Admission, DemographicVectorConfig,
                       AggregateRepresentation, InpatientInterventions,
                       LeadingObservableExtractorConfig, LeadingObservableExtractor,
                       StaticInfo)
from .dataset import (DatasetScheme, ColumnNames, DatasetConfig,
                      DatasetSchemeConfig)
from ..base import Config, Module
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
    int_proc: str = 'int_mimic4_proc'  # -> 'int_mimic4_grouped_proc'
    int_input: str = 'int_mimic4_input'  # -> 'int_mimic4_input_group'
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
                @ staticmethod

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
                    leading_observable_config: LeadingObservableExtractorConfig,
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

        leading_obs_extractor = LeadingObservableExtractor(leading_observable_config)

        def gen_admission(i):
            interventions = InpatientInterventions(
                proc=procedures[i],
                input_=inputs[i],
                adm_interval=adm_interval[i])

            obs = observables[i]
            lead_obs = leading_obs_extractor(obs)

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


class MIMICIVSQLTableConfig(Config):
    name: str
    query: str
    admission_id_alias: str = 'hadm_id'

    @property
    def alias_dict(self):
        return {k: v for k, v in self.as_dict().items() if k.endswith('_alias')}

    @property
    def alias_id_dict(self):
        return {k: v for k, v in self.alias_dict.items() if '_id_' in k}


class AdmissionLinkedMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    admission_id_alias: str = 'hadm_id'


class SubjectLinkedMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    subject_id_alias: str = 'subject_id'


class TimestampedMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    attributes: Tuple[str] = tuple()
    time_alias: str = 'time_bin'


class IntervalMIMICIVSQLTableConfig(MIMICIVSQLTableConfig):
    space_query: str = ''
    description_alias: str = 'description'
    start_time_alias: str = 'start_time'
    end_time_alias: str = 'end_time'


class AdmissionMIMICIVSQLTableConfig(AdmissionLinkedMIMICIVSQLTableConfig, SubjectLinkedMIMICIVSQLTableConfig):
    admission_time_alias: str = 'admittime'
    discharge_time_alias: str = 'dischtime'


class StaticMIMICIVSQLTableConfig(SubjectLinkedMIMICIVSQLTableConfig):
    gender_alias: str = 'gender'
    anchor_year_alias: str = 'anchor_year'
    anchor_age_alias: str = 'anchor_age'
    race_alias: str = 'race'


class DxDischargeMIMICIVSQLTableConfig(AdmissionLinkedMIMICIVSQLTableConfig):
    space_query: str = ''
    code_alias: str = 'icd_code'
    version_alias: str = 'icd_version'
    description_alias: str = 'description'


class IntervalICUProcedureMIMICIVSQLTableConfig(IntervalMIMICIVSQLTableConfig):
    item_id_alias: str = 'itemid'


class IntervalHospProcedureMIMICIVSQLTableConfig(IntervalMIMICIVSQLTableConfig):
    icd_code_alias: str = 'icd_code'
    icd_version_alias: str = 'icd_version'


class RatedInputMIMICIVSQLTableConfig(IntervalICUProcedureMIMICIVSQLTableConfig):
    rate_alias: str = 'rate'
    rate_unit_alias: str = 'rateuom'
    amount_alias: str = 'amount'
    amount_unit_alias: str = 'amountuom'


admissions_conf = AdmissionMIMICIVSQLTableConfig(name="admissions",
                                                 query=(r"""
select hadm_id as {admission_id_alias}, 
subject_id as {subject_id_alias},
admittime as {admission_time_alias},
dischtime as {discharge_time_alias}
from mimiciv_hosp.admissions 
"""))

static_conf = StaticMIMICIVSQLTableConfig(name="static",
                                          query=(r"""
select 
p.subject_id as {subject_id_alias},
p.gender as {gender_alias},
a.race as {race_alias},
p.anchor_age as {anchor_age_alias},
p.anchor_year as {anchor_year_alias}
from mimiciv_hosp.patients p
left join 
(select subject_id, max(race) as race
from mimiciv_hosp.admissions
group by subject_id) as a
on p.subject_id = a.subject_id
"""))

dx_discharge_conf = DxDischargeMIMICIVSQLTableConfig(name="dx",
                                                     query=(r"""
select hadm_id as {admission_id_alias}, 
        icd_code as {code_alias}, 
        icd_version as {version_alias}
from mimiciv_hosp.diagnoses_icd 
"""),
                                                     space_query=(r"""
select icd_code as {code_alias},
        icd_version as {version_alias},
        long_title as {description_alias}
from mimiciv_hosp.d_icd_diagnoses
"""))

renal_out_conf = TimestampedMIMICIVSQLTableConfig(name="renal_out",
                                                  attributes=['uo_rt_6hr', 'uo_rt_12hr', 'uo_rt_24hr'],
                                                  query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_derived.kdigo_uo as uo
inner join mimiciv_icu.icustays icu
on icu.stay_id = uo.stay_id
    """))

renal_creat_conf = TimestampedMIMICIVSQLTableConfig(name="renal_creat",
                                                    attributes=['creat'],
                                                    query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.kdigo_creatinine
    """))

renal_aki_conf = TimestampedMIMICIVSQLTableConfig(name="renal_aki",
                                                  attributes=['aki_stage_smoothed'],
                                                  query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.kdigo_stages
    """))

sofa_conf = TimestampedMIMICIVSQLTableConfig(name="renal_aki",
                                             attributes=["sofa_24hours"],
                                             query=(r"""
select hadm_id {admission_id_alias}, {attributes}, s.endtime {time_alias} 
from mimiciv_derived.sofa as s
inner join mimiciv_icu.icustays icu on s.stay_id = icu.stay_id    
"""))

blood_gas_conf = TimestampedMIMICIVSQLTableConfig(name="blood_gas",
                                                  attributes=['so2', 'po2', 'pco2', 'fio2', 'fio2_chartevents',
                                                              'aado2', 'aado2_calc',
                                                              'pao2fio2ratio', 'ph',
                                                              'baseexcess', 'bicarbonate', 'totalco2', 'hematocrit',
                                                              'hemoglobin',
                                                              'carboxyhemoglobin', 'methemoglobin',
                                                              'chloride', 'calcium', 'temperature', 'potassium',
                                                              'sodium', 'lactate', 'glucose'],
                                                  query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.bg as bg
where hadm_id is not null
"""))

blood_chemistry_conf = TimestampedMIMICIVSQLTableConfig(name="blood_chemistry",
                                                        attributes=['albumin', 'globulin', 'total_protein',
                                                                    'aniongap', 'bicarbonate', 'bun',
                                                                    'calcium', 'chloride',
                                                                    'creatinine', 'glucose', 'sodium', 'potassium'],
                                                        query=(r"""
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias} 
from mimiciv_derived.chemistry as ch
where hadm_id is not null
"""))

cardiac_marker_conf = TimestampedMIMICIVSQLTableConfig(name="cardiac_marker",
                                                       attributes=['troponin_t', 'ntprobnp', 'ck_mb'],
                                                       query=(r"""
with trop as
(
    select specimen_id, MAX(valuenum) as troponin_t
    from mimiciv_hosp.labevents
    where itemid = 51003
    group by specimen_id
)
select hadm_id {admission_id_alias}, {attributes}, charttime {time_alias}
from mimiciv_hosp.admissions a
left join mimiciv_derived.cardiac_marker c
  on a.hadm_id = c.hadm_id
left join trop
  on c.specimen_id = trop.specimen_id
where c.hadm_id is not null
"""))

weight_conf = TimestampedMIMICIVSQLTableConfig(name="weight",
                                               attributes=['weight'],
                                               query=(r"""
select icu.hadm_id {admission_id_alias}, {attributes}, w.time_bin {time_alias}
 from (
 (select stay_id, w.weight, w.starttime time_bin
  from mimiciv_derived.weight_durations as w)
 union all
 (select stay_id, w.weight, w.endtime time_bin
     from mimiciv_derived.weight_durations as w)
 ) w
inner join mimiciv_icu.icustays icu on w.stay_id = icu.stay_id
"""))

cbc_conf = TimestampedMIMICIVSQLTableConfig(name="cbc",
                                            attributes=['hematocrit', 'hemoglobin', 'mch', 'mchc', 'mcv', 'platelet',
                                                        'rbc', 'rdw', 'wbc'],
                                            query=(r"""
select hadm_id {admission_id_alias}, {attributes}, cbc.charttime {time_alias}
from mimiciv_derived.complete_blood_count as cbc
where hadm_id is not null
"""))

vital_conf = TimestampedMIMICIVSQLTableConfig(name="vital",
                                              attributes=['heart_rate', 'sbp', 'dbp', 'mbp', 'sbp_ni', 'dbp_ni',
                                                          'mbp_ni', 'resp_rate',
                                                          'temperature', 'spo2',
                                                          'glucose'],
                                              query=(r"""
select hadm_id {admission_id_alias}, {attributes}, v.charttime {time_alias}
from mimiciv_derived.vitalsign as v
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = v.stay_id
"""))

# Glasgow Coma Scale, a measure of neurological function
gcs_conf = TimestampedMIMICIVSQLTableConfig(name="gcs",
                                            attributes=['gcs', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gcs_unable'],
                                            query=(r"""
select hadm_id {admission_id_alias}, {attributes}, gcs.charttime {time_alias}
from mimiciv_derived.gcs as gcs
inner join mimiciv_icu.icustays as icu
 on icu.stay_id = gcs.stay_id
"""))

## Inputs - Canonicalise

icu_input_conf = RatedInputMIMICIVSQLTableConfig(name="int_input",
                                                 query=(r"""
select
    a.hadm_id as {admission_id_alias}
    , inp.itemid as {item_id_alias}
    , inp.starttime as {start_time_alias}
    , inp.endtime as {end_time_alias}
    , di.label as {description_alias}
    , inp.rate  as {rate_alias}
    , inp.amount as {amount_alias}
    , inp.rateuom as {rate_unit_alias}
    , inp.amountuom as {amount_unit_alias}
from mimiciv_hosp.admissions a
inner join mimiciv_icu.icustays i
    on a.hadm_id = i.hadm_id
left join mimiciv_icu.inputevents inp
    on i.stay_id = inp.stay_id
left join mimiciv_icu.d_items di
    on inp.itemid = di.itemid
"""),
                                                 space_query=(r"""
select di.itemid as {item_id_alias}
from mimiciv_icu.inputevents ie
left join mimiciv_icu.d_items di
on ie.itemid = di.itemid
where di.itemid is not null
"""))

## Procedures - Canonicalise and Refine
icuproc_conf = IntervalICUProcedureMIMICIVSQLTableConfig(name="int_proc_icu",
                                                         query=(r"""
select a.hadm_id as {admission_id_alias}
    , pe.itemid as {item_id_alias}
    , pe.starttime as {start_time_alias}
    , pe.endtime as {end_time_alias}
    , di.label as {description_alias}
from mimiciv_hosp.admissions a
inner join mimiciv_icu.icustays i
    on a.hadm_id = i.hadm_id
left join mimiciv_icu.procedureevents pe
    on i.stay_id = pe.stay_id
left join mimiciv_icu.d_items di
    on pe.itemid = di.itemid
"""),
                                                         space_query=(r"""
select di.itemid as {item_id_alias}
from mimiciv_icu.procedureevents pe
left join mimiciv_icu.d_items di
on pe.itemid = di.itemid
where di.itemid is not null
"""))

hospicdproc_conf = IntervalHospProcedureMIMICIVSQLTableConfig(name="int_proc_icd",
                                                              query=(r"""
select pi.hadm_id as {admission_id_alias}
, (pi.chartdate)::timestamp as {start_time_alias}
, (pi.chartdate + interval '1 hour')::timestamp as {end_time_alias}
, pi.icd_code as {icd_code_alias}
, pi.icd_version as {icd_version_alias}
, di.long_title as {description_alias}
FROM mimiciv_hosp.procedures_icd pi
INNER JOIN mimiciv_hosp.d_icd_procedures di
  ON pi.icd_version = di.icd_version
  AND pi.icd_code = di.icd_code
INNER JOIN mimiciv_hosp.admissions a
  ON pi.hadm_id = a.hadm_id
"""),
                                                              space_query=(r"""
                                                            
select di.icd_code  as {self.config.hosp_procedures_table.icd_code_alias},  
di.icd_version as {self.config.hosp_procedures_table.icd_version_alias},
di.long_title as {self.config.hosp_procedures_table.description_alias}
from mimiciv_hosp.procedures_icd pi
inner join mimiciv_hosp.d_icd_procedures di
  on pi.icd_version = di.icd_version
  and pi.icd_code = di.icd_code
where di.icd_code is not null and di.icd_version is not null
"""))


class MIMICIVSQTable(Module):
    config: MIMICIVSQLTableConfig

    def _coerce_id_to_str(self, df: pd.DataFrame):
        """
        Some of the integer ids in the database when downloaded are stored as floats.
        A fix is to coerce them to integers then fix as strings.
        """
        id_alias_dict = self.config.alias_id_dict
        # coerce to integers then fix as strings.
        int_dtypes = {k: int for k in id_alias_dict.values()}
        str_dtypes = {k: str for k in id_alias_dict.values()}
        return df.astype(int_dtypes).astype(str_dtypes)


class MIMICIVFixedSQTable(MIMICIVSQTable):
    def __call__(self, engine: Engine):
        query = self.config.query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine))


class TimestampedMIMICIVSQLTable(MIMICIVSQTable):
    config: TimestampedMIMICIVSQLTableConfig

    def __call__(self, engine: Engine,
                 attributes: List[str]):
        assert len(set(attributes)) == len(attributes), \
            f"Duplicate attributes {attributes}"
        assert all(a in self.config.attributes for a in attributes), \
            f"Some attributes {attributes} not in {self.config.attributes}"
        query = self.config.query.format(attributes=','.join(attributes),
                                         **self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine))


class IntervalMIMICIVSQLTable(MIMICIVFixedSQTable):
    config: IntervalMIMICIVSQLTableConfig

    def space(self, engine: Engine):
        query = self.config.space_query.format(**self.config.alias_dict)
        return self._coerce_id_to_str(pd.read_sql(query, engine))


class MIMICIVSQLConfig(Config):
    host: str
    port: int
    user: str
    password: str
    dbname: str

    static_table: StaticMIMICIVSQLTableConfig = static_conf
    admissions_table: AdmissionMIMICIVSQLTableConfig = admissions_conf
    dx_discharge_table: DxDischargeMIMICIVSQLTableConfig = dx_discharge_conf
    obs_tables: List[TimestampedMIMICIVSQLTableConfig] = field(default_factory=lambda: [
        renal_out_conf,
        renal_creat_conf,
        renal_aki_conf,
        sofa_conf,
        blood_gas_conf,
        blood_chemistry_conf,
        cardiac_marker_conf,
        weight_conf,
        cbc_conf,
        vital_conf,
        gcs_conf
    ])
    icu_procedures_table: IntervalICUProcedureMIMICIVSQLTableConfig = icuproc_conf
    icu_inputs_table: RatedInputMIMICIVSQLTableConfig = icu_input_conf
    hosp_procedures_table: IntervalHospProcedureMIMICIVSQLTableConfig = hospicdproc_conf


class ObservableMIMICScheme(FlatScheme):

    @classmethod
    def from_selection(cls, name: str, obs_variables: pd.DataFrame):
        """
        Create a scheme from a selection of observation variables.

        Args:
            name: Name of the scheme.
            obs_variables: A DataFrame containing the variables to include in the scheme.
                The DataFrame should have the following columns:
                    - table_name (index): The name of the table containing the variable.
                    - attribute: The name of the variable.

        Returns:
            (CodingScheme.FlatScheme) A new scheme containing the variables in obs_variables.
        """
        obs_variables = obs_variables.sort_values(['table_name', 'attribute'])
        # format codes to be of the form 'table_name.attribute'
        codes = obs_variables.index.map(lambda x: f"{x}.{obs_variables.loc[x, 'attribute']}").tolist()
        desc = dict(zip(codes, codes))
        index = dict(zip(codes, range(len(codes))))

        return cls(CodingSchemeConfig(name),
                   codes=codes,
                   desc=desc,
                   index=index)

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'table_name', 'attribute']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split('.')) for c in self.codes],
                            columns=columns)


class MixedICDMIMICIVScheme(FlatScheme):
    icd_schemes: Dict[str, ICD]

    @classmethod
    def from_selection(cls, name: str, icd_version_selection: pd.DataFrame,
                       version_alias: str, icd_code_alias: str, description_alias: str,
                       icd_schemes: Dict[str, str]):
        icd_version_selection = icd_version_selection.sort_values([version_alias, icd_code_alias])
        assert icd_version_selection[version_alias].isin(icd_schemes).all(), \
            f"Only {', '.join(map(lambda x: f'ICD-{x}', icd_schemes))} are expected."
        icd_schemes_loaded: Dict[str, ICD] = {k: ICD.from_name(v) for k, v in icd_schemes.items()}

        assert all(isinstance(s, ICD) for s in icd_schemes_loaded.values()), \
            "Only ICD schemes are expected."

        for version, icd_df in icd_version_selection.groupby(version_alias):
            scheme = icd_schemes_loaded[version]
            icd_version_selection.loc[icd_df.index, icd_code_alias] = \
                icd_df[icd_code_alias].str.replace(' ', '').str.replace('.', '').map(scheme.add_dots)

        codes = icd_version_selection.index.map(lambda x: f"{x}:{icd_version_selection.loc[x, 'icd_code']}").tolist()
        df = icd_version_selection.copy()
        df['code'] = codes
        index = dict(zip(codes, range(len(codes))))
        desc = df.set_index('code')[description_alias].to_dict()

        return cls(CodingSchemeConfig(name),
                   codes=codes,
                   desc=desc,
                   index=index,
                   icd_schemes=icd_schemes_loaded)

    def mixedcode_format_table(self, table: pd.DataFrame, code_alias: str, version_alias: str):
        """
        Format a table with mixed codes to the ICD version:icd_code format and filter out codes that are not in the scheme.
        """
        table = table.copy()
        assert version_alias in table.columns, f"Column {version_alias} not found."
        assert code_alias in table.columns, f"Column {code_alias} not found."
        assert table[version_alias].isin(self.icd_schemes).all(), \
            f"Only ICD version {list(self.icd_schemes.keys())} are expected."

        for version, icd_df in table.groupby(version_alias):
            scheme = self.icd_schemes[version]
            table.loc[icd_df.index, code_alias] = icd_df[code_alias].str.replace(' ', '').str.replace('.', '').map(
                scheme.add_dots)

        # the version:icd_code format.
        table['code'] = table.index.map(lambda x: f"{table.loc[x, version_alias]}:{table.loc[x, code_alias]}")

        # filter out codes that are not in the scheme.
        table = table[table['code'].isin(self.codes)].reset_index(drop=True)
        table = table.drop
        return table

    def generate_maps(self):
        """
        Register the mappings between the Mixed ICD scheme and the individual ICD scheme.
        For example, if the current `MixedICDMIMICIV` is mixing ICD-9 and ICD-10,
        then register the two mappings between this scheme and ICD-9 and ICD-10 separately.
        This assumes that the current runtime has already registered mappings
        between the individual ICD schemes.
        """

        # mixed2pure_maps has the form {icd_version: {mixed_code: {icd}}}.
        mixed2pure_maps = {}
        lost_codes = []
        dataframe = self.as_dataframe()
        for pure_version, pure_scheme in self.icd_schemes.items():
            # mixed2pure has the form {mixed_code: {icd}}.
            mixed2pure = defaultdict(set)
            for mixed_version, mixed_version_df in dataframe.groupby('icd_version'):
                icd2mixed = mixed_version_df.set_index('icd_code')['code'].to_dict()
                icd2mixed = {icd: mixed_code for icd, mixed_code in icd2mixed.items() if icd in pure_scheme.codes}
                assert len(icd2mixed) > 0, "No mapping between the mixed and pure ICD schemes was found."
                if mixed_version == pure_version:
                    mixed2pure.update({c: {icd} for icd, c in icd2mixed.items()})
                else:
                    # if mixed_version != pure_version, then retrieve
                    # the mapping between ICD-{mixed_version} and ICD-{pure_version}
                    pure_map = CodeMap.get_mapper(self.icd_schemes[mixed_version].name,
                                                  self.icd_schemes[pure_version].name)

                    for icd, mixed_code in icd2mixed.items():
                        if icd in pure_map:
                            mixed2pure[mixed_code].update(pure_map[icd])
                        else:
                            lost_codes.append(mixed_code)

            mixed2pure_maps[pure_version] = mixed2pure
        # register the mapping between the mixed and pure ICD schemes.
        for pure_version, mixed2pure in mixed2pure_maps.items():
            pure_scheme = self.icd_schemes[pure_version]
            conf = CodingSchemeConfig(self.name, pure_scheme.name)

            CodeMap.register_map(self.name,
                                 pure_scheme.name,
                                 CodeMap(conf, mixed2pure))

        lost_df = dataframe[dataframe['code'].isin(lost_codes)]
        if len(lost_df) > 0:
            logging.warning(f"Lost {len(lost_df)} codes when generating the mapping between the Mixed ICD"
                            "scheme and the individual ICD scheme.")
            logging.warning(lost_df.to_string().replace('\n', '\n\t'))

    def register_maps_loaders(self):
        """
        Register the lazy-loading of mapping between the Mixed ICD scheme and the individual ICD scheme.
        """
        for pure_version, pure_scheme in self.icd_schemes.items():
            CodeMap.register_map_loader(self.name, pure_scheme.name, self.generate_maps)

    def as_dataframe(self):
        columns = ['code', 'desc', 'code_index', 'icd_version', 'icd_code']
        return pd.DataFrame([(c, self.desc[c], self.index[c], *c.split(':')) for c in self.codes],
                            columns=columns)


class MIMICIVSQL(Module):
    config: MIMICIVSQLConfig

    def create_engine(self) -> Engine:
        return sqlalchemy.create_engine(
            f'postgresql+psycopg2://{self.config.user}:{self.config.password}@'
            f'{self.config.host}:{self.config.port}/{self.config.dbname}')

    def register_obs_scheme(self, name: str,
                            obs_variables: Optional[Union[pd.DataFrame, Dict[str, List[str]]]] = None):
        """
        From the given selection of observable variables `obs_variables`, generate a new scheme
        that can be used to generate for vectorisation of the timestamped observations.

        Args:
            name : The name of the scheme.
            obs_variables : A dictionary of table names and their corresponding attributes.
                If None, all supported variables will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """

        supported_obs_variables = self.supported_obs_variables
        if obs_variables is None:
            obs_variables = self.supported_obs_variables
        elif isinstance(obs_variables, dict):
            rows = []
            for table_name, attributes in obs_variables.items():
                assert table_name in supported_obs_variables.index, \
                    f"Table {table_name} not supported."
                assert all(a in supported_obs_variables.loc[table_name] for a in attributes), \
                    f"Some attributes {attributes} not in {supported_obs_variables.loc[table_name]}"
                rows.extend([(table_name, a) for a in attributes])
            obs_variables = pd.DataFrame(rows, columns=['table_name', 'attribute']).set_index('table_name', drop=True)

        obs_scheme = ObservableMIMICScheme.from_selection(name, obs_variables)
        FlatScheme.register_scheme(obs_scheme)
        return obs_scheme

    @staticmethod
    def _register_flat_scheme(name: str, codes: List[str], desc: Optional[Dict[str, str]] = None):
        if desc is None:
            desc = dict(zip(codes, codes))
        index = dict(zip(codes, range(len(codes))))
        codes = sorted(codes)
        scheme = FlatScheme(CodingSchemeConfig(name),
                            codes=codes,
                            desc=desc,
                            index=index)
        FlatScheme.register_scheme(scheme)
        return scheme

    def register_icu_input_scheme(self, name: str, itemid_selection: Optional[pd.DataFrame] = None):
        """
        From the given selection of ICU input items `itemid_selection`, generate a new scheme
        that can be used to generate for vectorisation of the ICU inputs. If `itemid_selection` is None,
        all supported items will be used.

        Args:
            name : The name of the new scheme.
            itemid_selection : A dataframe containing the `itemid`s to generate the new scheme. If None, all supported items will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        supported_icu_inputs = self.supported_icu_inputs
        if itemid_selection is None:
            itemid_selection = supported_icu_inputs[self.config.icu_inputs_table.item_id_alias].tolist()
        else:
            itemid_selection = itemid_selection[self.config.icu_inputs_table.item_id_alias].astype(str).tolist()
            assert all(i in supported_icu_inputs[self.config.icu_inputs_table.item_id_alias].tolist()
                       for i in itemid_selection), \
                f"Some item ids {itemid_selection} not in {supported_icu_inputs[self.config.icu_inputs_table.item_id_alias].tolist()}"
        return self._register_flat_scheme(name, itemid_selection)

    def register_icu_procedure_scheme(self, name: str, itemid_selection: Optional[pd.DataFrame] = None):
        """
        From the given selection of ICU procedure items `itemid_selection`, generate a new scheme
        that can be used to generate for vectorisation of the ICU procedures. If `itemid_selection` is None,
        all supported items will be used.

        Args:
            name : The name of the new scheme.
            itemid_selection : A dataframe containing the the `itemid`s of choice. If None, all supported items will be used.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        supported_icu_procedures = self.supported_icu_procedures
        if itemid_selection is None:
            itemid_selection = supported_icu_procedures[self.config.icu_procedures_table.item_id_alias].tolist()
        else:
            itemid_selection = itemid_selection[self.config.icu_procedures_table.item_id_alias].astype(str).tolist()
            assert all(i in supported_icu_procedures[self.config.icu_procedures_table.item_id_alias].tolist()
                       for i in itemid_selection), \
                f"Some item ids {itemid_selection} not in {supported_icu_procedures[self.config.icu_procedures_table.item_id_alias].tolist()}"
        return self._register_flat_scheme(name, itemid_selection)

    def register_hosp_procedure_scheme(self, name: str, icd_version_selection: Optional[pd.DataFrame]):
        """
        From the given selection of hospital procedure items `icd_version_selection`, generate a new scheme.

        Args:
            name : The name of the new scheme.
            icd_version_selection : A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        c_version = self.config.hosp_procedures_table.icd_version_alias
        c_code = self.config.hosp_procedures_table.icd_code_alias
        c_desc = self.config.hosp_procedures_table.icd_desc_alias
        supported_hosp_procedures = self.supported_hosp_procedures
        if icd_version_selection is None:
            icd_version_selection = supported_hosp_procedures[[c_version, c_code]].drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
        else:
            icd_version_selection = icd_version_selection[[c_version, c_code]]
            icd_version_selection = icd_version_selection.drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
            # Check that all the ICD codes-versions are supported.
            for version, codes in icd_version_selection.groupby(self.config.hosp_procedures_table.icd_version_alias):
                support_subset = supported_hosp_procedures[supported_hosp_procedures[c_version] == version]
                assert codes[c_code].isin(support_subset[c_code]).all(), \
                    f"Some ICD codes {codes} not in {supported_hosp_procedures[supported_hosp_procedures[c_version] == version][c_code]}"
        icu_proc_scheme = MixedICDMIMICIVScheme.from_selection(name, icd_version_selection,
                                                               version_alias=c_version,
                                                               icd_code_alias=c_code,
                                                               description_alias=c_desc,
                                                               icd_schemes={'9': 'PrICD9', '10': 'PrICD10'})
        MixedICDMIMICIVScheme.register_scheme(icu_proc_scheme)
        icu_proc_scheme.register_maps_loaders()

        return icu_proc_scheme

    def register_dx_discharge_scheme(self, name: str, icd_version_selection: Optional[pd.DataFrame]):
        """
        From the given selection of discharge diagnosis items `icd_version_selection`, generate a new scheme.

        Args:
            name : The name of the new scheme.
            icd_version_selection : A dataframe containing the `icd_code`s to generate the new scheme. If None, all supported items will be used. The dataframe should have the following columns:
                - icd_version: The version of the ICD.
                - icd_code: The ICD code.

        Returns:
            (CodingScheme.FlatScheme) A new scheme that is also registered in the current runtime.
        """
        c_version = self.config.dx_discharge_table.version_alias
        c_code = self.config.dx_discharge_table.code_alias
        c_desc = self.config.dx_discharge_table.description_alias
        supported_dx_discharge = self.supported_dx_discharge
        if icd_version_selection is None:
            icd_version_selection = supported_dx_discharge[[c_version, c_code]].drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
        else:
            icd_version_selection = icd_version_selection[[c_version, c_code]]
            icd_version_selection = icd_version_selection.drop_duplicates()
            icd_version_selection = icd_version_selection.astype(str)
            # Check that all the ICD codes-versions are supported.
            for version, codes in icd_version_selection.groupby(self.config.dx_discharge_table.version_alias):
                support_subset = supported_dx_discharge[supported_dx_discharge[c_version] == version]
                assert codes[c_code].isin(support_subset[c_code]).all(), \
                    f"Some ICD codes {codes} not in {supported_dx_discharge[supported_dx_discharge[c_version] == version][c_code]}"
        dx_discharge_scheme = MixedICDMIMICIVScheme.from_selection(name, icd_version_selection,
                                                                   version_alias=c_version,
                                                                   icd_code_alias=c_code,
                                                                   description_alias=c_desc,
                                                                   icd_schemes={'9': 'DxICD9', '10': 'DxICD10'})
        MixedICDMIMICIVScheme.register_scheme(dx_discharge_scheme)
        return dx_discharge_scheme

    @cached_property
    def supported_obs_variables(self) -> pd.DataFrame:
        rows = []
        for table in self.config.obs_tables:
            rows.extend([(table.name, a) for a in table.attributes])
        return pd.DataFrame(rows, columns=['table_name', 'attribute']).set_index('table_name', drop=True)

    @cached_property
    def supported_icu_procedures(self) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.icu_procedures_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_icu_inputs(self) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.icu_inputs_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_hosp_procedures(self) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.hosp_procedures_table)
        return table.space(self.create_engine())

    @cached_property
    def supported_dx_discharge(self) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.dx_discharge_table)
        return table.space(self.create_engine())

    def obs_table(self, table_name: str) -> TimestampedMIMICIVSQLTable:
        table_conf = next(t for t in self.config.obs_tables if t.name == table_name)
        return TimestampedMIMICIVSQLTable(table_conf)

    def _extract_static_table(self, engine: Engine) -> pd.DataFrame:
        table = MIMICIVSQTable(self.config.static_table)
        return table(engine)

    def _extract_admissions_table(self, engine: Engine) -> pd.DataFrame:
        table = MIMICIVSQTable(self.config.admissions_table)
        return table(engine)

    def _extract_dx_discharge_table(self, engine: Engine, dx_discharge_scheme: MixedICDMIMICIVScheme) -> pd.DataFrame:
        table = MIMICIVSQTable(self.config.dx_discharge_table)
        dataframe = table(engine)
        c_icd_version = self.config.dx_discharge_table.version_alias
        c_icd_code = self.config.dx_discharge_table.code_alias
        return dx_discharge_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version)

    def _extract_obs_table(self, engine: Engine, obs_scheme: ObservableMIMICScheme) -> pd.DataFrame:
        dfs = dict()
        for table_name, attrs_df in obs_scheme.as_dataframe().groupby('table_name'):
            attributes = attrs_df['attribute'].tolist()
            attr2code = attrs_df.set_index('attribute')['code'].to_dict()
            # download the table.
            table = self.obs_table(table_name)
            obs_df = table(engine, attributes)
            # melt the table. (admission_id, time, attribute, value)
            melted_obs_df = obs_df.melt(id_vars=[table.config.admission_id_alias, table.config.time_alias],
                                        var_name=['attribute'], value_name='value', value_vars=attributes)
            # add the code. (admission_id, time, attribute, value, code)
            melted_obs_df['code'] = melted_obs_df['attribute'].map(attr2code)
            melted_obs_df = melted_obs_df[melted_obs_df.value.notnull()]
            # drop the attribute. (admission_id, time, value, code)
            melted_obs_df = melted_obs_df.drop('attribute', axis=1)
            dfs[table_name] = melted_obs_df

        return pd.concat(dfs.values(), ignore_index=True)

    def _extract_icu_procedures_table(self, engine: Engine, icu_procedure_scheme: FlatScheme) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.icu_procedures_table)
        c_itemid = self.config.icu_procedures_table.item_id_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_itemid].isin(icu_procedure_scheme.codes)]
        return dataframe.reset_index(drop=True)

    def _extract_icu_inputs_table(self, engine: Engine, icu_input_scheme: FlatScheme) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.icu_inputs_table)
        c_itemid = self.config.icu_inputs_table.item_id_alias
        dataframe = table(engine)
        dataframe = dataframe[dataframe[c_itemid].isin(icu_input_scheme.codes)]
        return dataframe.reset_index(drop=True)

    def _extract_hosp_procedures_table(self, engine: Engine,
                                       procedure_icd_scheme: MixedICDMIMICIVScheme) -> pd.DataFrame:
        table = IntervalMIMICIVSQLTable(self.config.hosp_procedures_table)
        c_icd_code = self.config.hosp_procedures_table.icd_code_alias
        c_icd_version = self.config.hosp_procedures_table.icd_version_alias
        dataframe = table(engine)
        return procedure_icd_scheme.mixedcode_format_table(dataframe, c_icd_code, c_icd_version)

    def __call__(self,
                 dx_discharge_scheme: Optional[MixedICDMIMICIVScheme] = None,
                 obs_scheme: Optional[ObservableMIMICScheme] = None,
                 icu_input_scheme: Optional[FlatScheme] = None,
                 hosp_procedures_scheme: Optional[MixedICDMIMICIVScheme] = None,
                 icu_procedures_scheme: Optional[FlatScheme] = None) -> None:
        engine = self.create_engine()
        static_df = self._extract_static_table(engine)
        admissions_df = self._extract_admissions_table(engine)
        dx_discharge_df = self._extract_dx_discharge_table(engine, dx_discharge_scheme)
        obs_df = self._extract_obs_table(engine, obs_scheme)
        icu_input_df = self._extract_icu_inputs_table(engine, icu_input_scheme)
        icu_proc_df = self._extract_icu_procedures_table(engine, icu_procedures_scheme)
        hosp_proc_icd_df = self._extract_hosp_procedures_table(engine, hosp_procedures_scheme)


def register_mimic4_aki_schemes():
    pass


def load_aki_default_input_items():
    df = pd.read_csv(os.path.join(_RSC_DIR, 'mimic4_aki_icu_inputs_itemid_selection.csv.gz'),
                     dtype={'itemid': str})
    return df[['itemid']].reset_index(drop=True)


def load_aki_default_icu_procedure_items():
    df = pd.read_csv(os.path.join(_RSC_DIR, 'mimic4_aki_icu_procedures_itemid_selection.csv.gz'),
                     dtype={'itemid': str})
    return df[['itemid']].reset_index(drop=True)
