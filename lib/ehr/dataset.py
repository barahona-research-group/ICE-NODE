"""."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, ClassVar, Callable, Type, Union
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
import pickle
from datetime import datetime, timedelta
import random
from abc import abstractmethod
import logging
from dateutil.relativedelta import relativedelta

import pandas as pd
import dask.dataframe as dd
import numpy as np
import equinox as eqx

from ..utils import load_config, translate_path
from ..base import AbstractConfig

from .coding_scheme import (scheme_from_classname, OutcomeExtractor,
                            AbstractScheme, ICDCommons, MIMICEth, MIMICInput,
                            MIMICInputGroups, MIMICProcedures,
                            MIMICProcedureGroups, MIMICObservables, NullScheme,
                            Gender, Ethnicity, CPRDIMDCategorical, CPRDGender,
                            load_maps)
from .concepts import (InpatientInput, InpatientObservables, Patient,
                       Admission, StaticInfo, DemographicVectorConfig,
                       CPRDDemographicVectorConfig, CPRDStaticInfo,
                       AggregateRepresentation, InpatientInterventions,
                       LeadingObservableConfig)

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')

StrDict = Dict[str, str]


def try_compute(df):
    if hasattr(df, 'compute'):
        return df.compute()
    else:
        return df


keys = [
    'subject_id', 'admission_id', 'admittime', 'dischtime', 'adm_interval',
    'index', 'code', 'version', 'gender', 'ethnicity', 'imd_decile',
    'anchor_age', 'anchor_year', 'start_time', 'end_time', 'rate',
    'code_index', 'code_source_index', 'value', 'timestamp', 'date_of_birth',
    'age_at_dischtime'
]
default_raw_types = {
    'subject_id': int,
    'admission_id': int,
    'index': int,
    'code': str,
    'version': str
}


class ColumnNames(namedtuple('ColumnNames', keys,
                             defaults=[None] * len(keys))):

    def _asdict(self):
        return {
            k: getattr(self, k)
            for k in keys if getattr(self, k) is not None
        }

    def has(self, key):
        return getattr(self, key) is not None

    @property
    def columns(self):
        return [getattr(self, k) for k in keys if getattr(self, k) is not None]

    @property
    def default_raw_types(self):
        return {
            getattr(self, k): v
            for k, v in default_raw_types.items() if k in self
        }

    def __contains__(self, key):
        return self.has(key)

    @classmethod
    def make(cls, colname: Dict[str, str]) -> 'ColumnNames':
        return cls(**colname)


def random_date(start, end, rng: random.Random):
    """
    https://stackoverflow.com/a/553448
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = rng.randrange(int_delta)
    return start + timedelta(seconds=random_second)


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
    c_value: str
    c_code_index: str
    mean: pd.Series
    std: pd.Series
    use_float16: bool = True

    def __call__(self, df):
        mean = df[self.c_code_index].map(self.mean)
        std = df[self.c_code_index].map(self.std)
        df.loc[:, self.c_value] = ((df[self.c_value] - mean) / std)
        if self.use_float16:
            df = df.astype({self.c_value: np.float16})
        return df

    def unscale(self, array):
        index = np.arange(array.shape[-1])
        return array * self.std.loc[index] + self.mean.loc[index]


class MaxScaler(eqx.Module):
    c_value: str
    c_code_index: str
    max_val: pd.Series
    use_float16: bool = True

    def __call__(self, df):
        max_val = df[self.c_code_index].map(self.max_val)
        df.loc[:, self.c_value] = (df[self.c_value] / max_val)
        if self.use_float16:
            df = df.astype({self.c_value: np.float16})
        return df

    def unscale(self, array):
        index = np.arange(array.shape[-1])
        return array * self.max_val.loc[index]


class AdaptiveScaler(eqx.Module):
    c_value: str
    c_code_index: str
    max_val: pd.Series
    min_val: pd.Series
    mean: pd.Series
    std: pd.Series
    use_float16: bool = True

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
        index = np.arange(array.shape[-1])
        mu = self.mean.loc[index]
        sigma = self.std.loc[index]
        min_val = self.min_val.loc[index]
        max_val = self.max_val.loc[index]
        z_unscaled = array * sigma + mu
        minmax_unscaled = array * max_val + min_val
        return np.where(min_val >= 0.0, minmax_unscaled, z_unscaled)


class DatasetConfig(AbstractConfig):
    path: str
    scheme: Dict[str, str]
    colname: Dict[str, Dict[str, str]]
    files: Dict[str, str] = eqx.field(default_factory=dict)
    sample: Optional[int] = None
    scheme_classname: str = 'DatasetScheme'
    meta_fpath: str = ''
    tag: str = ''


class DatasetScheme(eqx.Module):
    dx: AbstractScheme
    ethnicity: Ethnicity
    gender: Gender
    outcome: Optional[OutcomeExtractor] = None

    @classmethod
    def _assert_valid_maps(cls, source, target):
        attrs = list(k for k in source.__dict__.keys()
                     if k != 'outcome' and not k.startswith('_'))
        for attr in attrs:
            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"

    def __init__(self, config: Dict[str, str], **kwargs):
        super().__init__()
        if config is not None:
            config = config.copy()
            config.update(kwargs)
        else:
            config = kwargs

        if config.get('outcome') is not None:
            self.outcome = OutcomeExtractor(config.pop('outcome'))

        for k, v in config.items():
            if isinstance(v, str):
                setattr(self, k, scheme_from_classname(v))

    def to_dict(self):

        def _to_dict(x):

            def value(v):
                if isinstance(v, AbstractScheme):
                    return v.__class__.__name__
                if isinstance(v, dict):
                    return _to_dict(v)
                return v

            return {k: value(v) for k, v in x.items()}

        return _to_dict(self.__dict__)

    @staticmethod
    def from_dict(config: Dict[str, str], scheme_classname, **kwargs):
        scheme_class = eval(scheme_classname)
        return scheme_class(config, **kwargs)

    def make_target_scheme_config(self, **kwargs):
        assert 'outcome' in kwargs, "Outcome must be specified"
        config = self.to_dict()
        config.update(kwargs)
        return config

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
        return self.dx.mapper_to(target_scheme.dx)

    def ethnicity_mapper(self, target_scheme: DatasetScheme):
        return self.ethnicity.mapper_to(target_scheme.ethnicity)

    @property
    def supported_target_scheme_options(self):
        attrs = tuple(k for k in self.__dict__.keys()
                      if k != 'outcome' and not k.startswith('_'))
        supproted_attr_targets = {
            k: (getattr(self, k).__class__.__name__, ) +
            getattr(self, k).supported_targets
            for k in attrs
        }
        supported_outcomes = OutcomeExtractor.supported_outcomes(self.dx)
        supproted_attr_targets['outcome'] = supported_outcomes
        return supproted_attr_targets


class Dataset(eqx.Module):
    df: Dict[str, pd.DataFrame]
    config: DatasetConfig
    scheme: DatasetScheme
    colname: Dict[str, ColumnNames]
    static_info_class: ClassVar[Type[StaticInfo]] = StaticInfo

    def __init__(self, config: DatasetConfig = None, config_path: str = None):
        if config is None:
            config = AbstractConfig.from_dict(load_config(config_path))

        super().__init__()
        self.config = config
        self.scheme = DatasetScheme.from_dict(config.scheme,
                                              config.scheme_classname)
        self.colname = {
            f: ColumnNames.make(m)
            for f, m in config.colname.items()
        }
        self._load_dataframes()

    @property
    def supported_target_scheme_options(self):
        return self.scheme.supported_target_scheme_options

    @abstractmethod
    def to_subjects(self, **kwargs):
        pass

    @abstractmethod
    def _load_dataframes(self):
        pass

    @classmethod
    def make_dataset_scheme(cls, **kwargs):
        return DatasetScheme(**kwargs)

    def _save_dfs(self, path: Union[str, Path], overwrite: bool):
        for name, df in self.df.items():
            filepath = Path(path).with_suffix(f'.{name}.csv.gz')
            if filepath.exists():
                if overwrite:
                    filepath.unlink()
                else:
                    raise RuntimeError(f'File {path} already exists.')
            dtypes = df.dtypes.to_dict()
            df = df.astype(
                {c: np.float32
                 for c in df.columns if dtypes[c] == np.float16})

            df.to_csv(filepath,
                      index=self.colname[name].has('index'),
                      compression='gzip')

    def _save_dtypes(self, path: Union[str, Path], overwrite: bool):
        dtypes = {}
        for name, df in self.df.items():
            dtypes[name] = df.dtypes.to_dict()
        path = Path(path).with_suffix('.dtypes.pickle')
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f'File {path} already exists.')
        with open(path, 'wb') as file:
            pickle.dump(dtypes, file)

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._save_dfs(path, overwrite)
        self._save_dtypes(path, overwrite)

        rest = eqx.tree_at(lambda x: x.df, self, {})
        path = path.with_suffix('.dataset.pickle')
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise RuntimeError(f'File {path} already exists.')
        with open(path, 'wb') as file:
            pickle.dump(rest, file)

    @staticmethod
    def _load_dfs(path: Union[str, Path]):
        with open(Path(path).with_suffix('.dtypes.pickle'), 'rb') as file:
            dtypes = pickle.load(file)

        df = {}
        for name, dtype in dtypes.items():
            df_path = Path(path).with_suffix(f'.{name}.csv.gz')

            f16_cols = [col for col in dtype if dtype[col] == np.float16]
            parse_dates = [
                col for col in dtype if dtype[col] == 'datetime64[ns]'
            ]

            _dtypes = dtype.copy()
            for col in dtype:
                if col in f16_cols:
                    _dtypes[col] = np.float32
                elif col in parse_dates:
                    _dtypes[col] = str
                else:
                    _dtypes[col] = dtype[col]

            df[name] = pd.read_csv(df_path,
                                   dtype=_dtypes,
                                   parse_dates=parse_dates)
            df[name] = df[name].astype({col: np.float16 for col in f16_cols})
        return df

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        with open(path.with_suffix('.dataset.pickle'), 'rb') as file:
            rest = pickle.load(file)

        df = cls._load_dfs(path)
        for name, cols in rest.colname.items():
            if cols.has('index'):
                df[name] = df[name].set_index(cols.index)
        return eqx.tree_at(lambda x: x.df, rest, df)

    def export_config(self):
        return self.config.to_dict()


class CPRDDatasetScheme(DatasetScheme):
    imd: CPRDIMDCategorical

    def demographic_vector_size(
            self, demographic_vector_config: CPRDDemographicVectorConfig):
        size = DatasetScheme.demographic_vector_size(
            self, demographic_vector_config)
        if demographic_vector_config.imd:
            size += len(self.imd)
        return size


class MIMIC4DatasetScheme(DatasetScheme):
    dx: Union[Dict[str, ICDCommons], ICDCommons]

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        if config is not None:
            config = config.copy()
            config.update(kwargs)
        else:
            config = kwargs

        if isinstance(config['dx'], dict):
            self.dx = {
                version: scheme_from_classname(scheme)
                for version, scheme in config['dx'].items()
            }

    @classmethod
    def _assert_valid_maps(cls, source, target):
        attrs = list(k for k in source.__dict__.keys()
                     if k not in ('outcome', 'dx') and not k.startswith('_'))
        for attr in attrs:
            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"
        for version, s_scheme in source.dx.items():
            t_scheme = target.dx
            assert s_scheme.mapper_to(
                t_scheme
            ), f"Cannot map dx (version={version}) from {s_scheme} to {t_scheme}"

    def dx_mapper(self, target_scheme: DatasetScheme):
        return {
            version: s_dx.mapper_to(target_scheme.dx)
            for version, s_dx in self.dx.items()
        }

    @property
    def supported_target_scheme_options(self):
        attrs = tuple(k for k in self.__dict__.keys()
                      if k not in ('outcome', 'dx') and not k.startswith('_'))
        supproted_attr_targets = {
            k: (getattr(self, k).__class__.__name__, ) +
            getattr(self, k).supported_targets
            for k in attrs
        }
        supported_dx_targets = {
            version: (scheme.__class__.__name__, ) + scheme.supported_targets
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


class MIMIC4ICUDatasetScheme(MIMIC4DatasetScheme):
    int_proc: MIMICProcedures
    int_input: MIMICInput
    obs: MIMICObservables

    def make_target_scheme_config(self, **kwargs):
        assert 'outcome' in kwargs, "Outcome must be specified"
        config = self.to_dict()
        config.update(kwargs)
        config['int_proc'] = 'MIMICProcedureGroups'
        config['int_input'] = 'MIMICInputGroups'
        return config


class MIMIC3Dataset(Dataset):
    df: Dict[str, dd.DataFrame]
    scheme: DatasetScheme

    @classmethod
    def sample_n_subjects(cls, df, c_subject_id, n, seed=None):
        if seed is not None:
            rng = random.Random(seed)

        subjects = df[c_subject_id].unique().compute()
        subjects = rng.sample(subjects.tolist(), n)
        return df[df[c_subject_id].isin(subjects)]

    @classmethod
    def _match_admissions_with_demographics(cls, df, colname):
        adm = df["adm"]
        static = df["static"]
        c_subject_id = colname["adm"].subject_id
        subject_ids = list(set(adm[c_subject_id].unique()) & set(static.index))
        logging.debug(
            f"Removing subjects by matching demographic"
            f"(-{len(set(static.index) - set(subject_ids))})"
            f"and admissions"
            f"(-{len(set(adm[c_subject_id].unique()) - set(subject_ids))})"
            "tables")

        static = static.loc[subject_ids]
        adm = adm[adm[c_subject_id].isin(subject_ids)]
        df["adm"] = adm
        df["static"] = static

    def _load_dataframes(self):
        config = self.config.copy()
        files = config.files
        colname = self.colname
        logging.debug('Loading dataframe files')
        df = {
            k:
            dd.read_csv(os.path.join(config.path, files[k]),
                        usecols=colname[k].columns,
                        dtype=colname[k].default_raw_types)
            for k in files.keys()
        }
        if config.sample is not None:
            df["adm"] = self.sample_n_subjects(df["adm"],
                                               colname["adm"].subject_id,
                                               config.sample, 0)
        logging.debug('[DONE] Loading dataframe files')
        logging.debug('Preprocess admissions')
        adm = df["adm"]
        static = df["static"]
        static = static.set_index(colname["static"].index).compute()
        adm = adm.set_index(colname["adm"].index).compute()

        adm = self._adm_cast_times(adm, colname["adm"])
        adm, colname["adm"] = self._adm_add_adm_interval(
            adm, colname["adm"], 1 / 3600.0)
        adm = self._adm_remove_subjects_with_negative_adm_interval(
            adm, colname["adm"])
        adm, merger_map = self._adm_merge_overlapping_admissions(
            adm, colname["adm"])

        logging.debug('[DONE] Preprocess admissions')

        # admission_id matching
        logging.debug("Matching admission_id")
        df_with_adm_id = {
            name: df[name]
            for name in df if colname[name].has('admission_id')
        }
        df_with_adm_id = self._map_admission_ids(df_with_adm_id, colname,
                                                 merger_map)
        df_with_adm_id = self._match_filter_admission_ids(
            adm, df_with_adm_id, colname)
        df.update(df_with_adm_id)
        logging.debug("[DONE] Matching admission_id")

        df["static"] = static
        df["adm"] = adm
        self.df = df

        logging.debug("Dataframes validation and time conversion")
        self._dx_fix_icd_dots()
        self._dx_filter_unsupported_icd()
        self._match_admissions_with_demographics(self.df, colname)
        self.df = {k: try_compute(v) for k, v in self.df.items()}
        logging.debug("[DONE] Dataframes validation and time conversion")

    def to_subjects(self, subject_ids: List[int], num_workers: int,
                    demographic_vector_config: DemographicVectorConfig,
                    target_scheme: DatasetScheme, **kwargs):

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

        logging.debug('Compiling admissions...')
        c_admittime = self.colname['adm'].admittime
        c_dischtime = self.colname['adm'].dischtime
        adf = self.df['adm']
        adm_dates = dict(
            zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))

        def gen_admission(i):
            return Admission(admission_id=i,
                             admission_dates=adm_dates[i],
                             dx_codes=dx_codes[i],
                             dx_codes_history=dx_codes_history[i],
                             outcome=outcome[i],
                             observables=None,
                             interventions=None)

        def _gen_subject(subject_id):

            _admission_ids = admission_ids[subject_id]
            # for subject_id, subject_admission_ids in admission_ids.items():
            _admission_ids = sorted(_admission_ids,
                                    key=lambda aid: adm_dates[aid][0])
            static_info = self.static_info_class(
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

    def _dx_fix_icd_dots(self):
        c_code = self.colname["dx"].code
        add_dots = self.scheme.dx.add_dots
        df = self.df["dx"]
        df = df.assign(**{c_code: df[c_code].str.strip().map(add_dots)})
        self.df['dx'] = df

    def _dx_filter_unsupported_icd(self):
        c_code = self.colname["dx"].code
        df = self.df["dx"]
        codeset = set(df[c_code])
        scheme = self.scheme.dx
        scheme_codes = set(scheme.codes)

        unrecognised = codeset - scheme_codes
        if len(unrecognised) > 0:
            logging.debug(f'Unrecognised ICD codes: {len(unrecognised)} '
                          f'({len(unrecognised)/len(codeset):.2%})')
            logging.debug(f'Unrecognised {type(scheme)} codes '
                          f'({len(unrecognised)}) '
                          f'to be removed (first 30): '
                          f'{sorted(unrecognised)[:30]}')
        df = df[df[c_code].isin(scheme_codes)]
        self.df['dx'] = df

    def random_splits(self,
                      splits: List[float],
                      subject_ids: Optional[List[int]] = None,
                      random_seed: int = 42,
                      balanced: str = 'subjects',
                      ignore_first_admission: bool = False):
        if subject_ids is None:
            subject_ids = self.subject_ids
        subject_ids = sorted(subject_ids)

        random.Random(random_seed).shuffle(subject_ids)
        subject_ids = np.array(subject_ids)

        c_subject_id = self.colname['adm'].subject_id
        c_adm_interval = self.colname['adm'].adm_interval
        adm_df = self.df['adm']
        adm_df = adm_df[adm_df[c_subject_id].isin(subject_ids)]

        if balanced == 'subjects':
            probs = (np.ones(len(subject_ids)) / len(subject_ids)).cumsum()

        elif balanced == 'admissions':
            n_adms = adm_df.groupby(c_subject_id).size()
            if ignore_first_admission:
                n_adms = n_adms - 1
            p_adms = n_adms.loc[subject_ids] / n_adms.sum()
            probs = p_adms.values.cumsum()

        elif balanced == 'adm_interval':
            subject_intervals_sum = adm_df.groupby(c_subject_id).agg(
                total_interval=(c_adm_interval, 'sum'))
            p_subject_intervals = subject_intervals_sum.loc[
                subject_ids] / subject_intervals_sum.sum()
            probs = p_subject_intervals.values.cumsum()

        splits = np.searchsorted(probs, splits)
        return [a.tolist() for a in np.split(subject_ids, splits)]

    @staticmethod
    def _adm_merge_overlapping_admissions(adm_df,
                                          colname,
                                          interval_inclusive=True):
        logging.debug("adm: Merging overlapping admissions")
        c_subject_id = colname.subject_id
        c_admittime = colname.admittime
        c_dischtime = colname.dischtime
        df = adm_df.copy()

        # Step 1: Collect overlapping admissions
        def _collect_overlaps(s_df):
            s_df = s_df.sort_values(c_admittime)
            disch_cummax = s_df.iloc[:-1][c_dischtime].cummax()
            disch_cummax_idx = disch_cummax.map(lambda x:
                                                (x == disch_cummax).idxmax())

            if interval_inclusive:
                has_parent = s_df[c_admittime].iloc[
                    1:].values <= disch_cummax.values
            else:
                has_parent = s_df[c_admittime].iloc[
                    1:].values < disch_cummax.values

            s_df = s_df.iloc[1:]
            s_df['sup_adm'] = np.where(has_parent, disch_cummax_idx, pd.NA)
            s_df = s_df[s_df.sup_adm.notnull()]
            return s_df['sup_adm'].to_dict()

        subj_df = {sid: sdf for sid, sdf in df.groupby(c_subject_id)}
        ch2pt = {}
        for sid, s_df in subj_df.items():
            ch2pt.update(_collect_overlaps(s_df))
        ch_adms = list(ch2pt.keys())

        # Step 3: Find super admission, in case of multiple levels of overlap.
        def super_adm(ch):
            while ch in ch2pt:
                ch = ch2pt[ch]
            return ch

        ch2sup = dict(zip(ch_adms, map(super_adm, ch_adms)))

        # Step 4: reversed map from super admission to child admissions.
        sup2ch = defaultdict(list)
        for ch, sup in ch2sup.items():
            sup2ch[sup].append(ch)

        # List of super admissiosn.
        sup_adms = list(sup2ch.keys())
        # List of admissions to be removed after merging.
        rem_adms = sum(map(list, sup2ch.values()), [])

        # Step 5: Merge overlapping admissions by extending discharge time.
        df['adm_id'] = df.index
        df.loc[sup_adms, c_dischtime] = df.loc[sup_adms].apply(lambda x: max(
            x[c_dischtime], df.loc[sup2ch[x.adm_id], c_dischtime].max()),
                                                               axis=1)

        # Step 6: Remove merged admissions.
        df = df.drop(index=rem_adms)
        df = df.drop(columns=['adm_id'])

        logging.debug(f"adm: Merged {len(rem_adms)} overlapping admissions")
        return df, ch2sup

    @staticmethod
    def _match_filter_admission_ids(adm_df, dfs, colname):
        dfs = {
            name: _df[_df[colname[name].admission_id].isin(adm_df.index)]
            for name, _df in dfs.items()
        }
        return {name: _df.reset_index() for name, _df in dfs.items()}

    @staticmethod
    def _adm_cast_times(adm_df, colname):
        df = adm_df.copy()
        # Cast timestamps for admissions
        for time_col in (colname.admittime, colname.dischtime):
            df[time_col] = pd.to_datetime(df[time_col])
        return df

    @staticmethod
    def _map_admission_ids(df, colname, merger_map):
        for name, _df in df.items():
            c_adm_id = colname[name].admission_id
            _df[c_adm_id] = _df[c_adm_id].map(lambda x: merger_map.get(x, x))
            df[name] = _df
        return df

    @staticmethod
    def _adm_add_adm_interval(adm_df, colname, seconds_scaler=1 / 3600.0):
        c_admittime = colname.admittime
        c_dischtime = colname.dischtime

        delta = adm_df[c_dischtime] - adm_df[c_admittime]
        adm_df = adm_df.assign(
            adm_interval=(delta.dt.total_seconds() *
                          seconds_scaler).astype(np.float32))
        colname = colname._replace(adm_interval="adm_interval")
        return adm_df, colname

    @staticmethod
    def _adm_remove_subjects_with_negative_adm_interval(adm_df, colname):
        c_adm_interval = colname.adm_interval
        c_subject = colname.subject_id

        subjects_neg_intervals = adm_df[adm_df[c_adm_interval] <
                                        0][c_subject].unique()
        logging.debug(
            f"Removing subjects with at least one negative adm_interval: "
            f"{len(subjects_neg_intervals)}")
        df = adm_df[~adm_df[c_subject].isin(subjects_neg_intervals)]
        return df

    @staticmethod
    def _adm_remove_subjects_with_overlapping_admissions(adm_df, colname):
        c_admittime = colname.admittime
        c_dischtime = colname.dischtime
        c_subject = colname.subject_id
        df = adm_df.copy()

        # df = df.sort_values(c_admittime)

        def minimum_disch_admit_gap(patient_adms_df):
            patient_adms_df = patient_adms_df.sort_values(c_admittime)
            if len(patient_adms_df) < 2:
                return pd.Timedelta(1, unit='h')
            return (patient_adms_df[c_admittime].iloc[1:].values -
                    patient_adms_df[c_dischtime].iloc[:-1].values).min()

        min_gaps = df.groupby(c_subject).apply(minimum_disch_admit_gap)
        subjects_overlapping = min_gaps[min_gaps < pd.Timedelta(0)].index
        logging.debug("Removing subjects with at least "
                      "one overlapping admission: "
                      f"{len(subjects_overlapping)}")
        df = df[~df[c_subject].isin(subjects_overlapping)]
        return df, min_gaps

    @property
    def subject_ids(self):
        return sorted(self.df["static"].index.unique())

    def subject_info_extractor(self, subject_ids: List[int],
                               target_scheme: DatasetScheme):
        """
        Important comment from MIMIC-III documentation at \
            https://mimic.mit.edu/docs/iii/tables/patients/
        > DOB is the date of birth of the given patient. Patients who are \
            older than 89 years old at any time in the database have had their\
            date of birth shifted to obscure their age and comply with HIPAA.\
            The shift process was as follows: the patient’s age at their \
            first admission was determined. The date of birth was then set to\
            exactly 300 years before their first admission.
        """
        assert self.scheme.gender is target_scheme.gender, (
            "No conversion assumed for gender attribute")

        c_gender = self.colname["static"].gender
        c_eth = self.colname["static"].ethnicity
        c_dob = self.colname["static"].date_of_birth

        c_admittime = self.colname["adm"].admittime
        c_dischtime = self.colname["adm"].dischtime
        c_subject_id = self.colname["adm"].subject_id

        adm_df = self.df['adm'][self.df['adm'][c_subject_id].isin(subject_ids)]

        df = self.df['static'].copy()
        df = df.loc[subject_ids]
        gender = df[c_gender].map(self.scheme.gender.codeset2vec)

        subject_gender = gender.to_dict()

        df[c_dob] = pd.to_datetime(df[c_dob])
        last_disch_date = adm_df.groupby(c_subject_id)[c_dischtime].max()
        first_adm_date = adm_df.groupby(c_subject_id)[c_admittime].min()

        last_disch_date = last_disch_date.loc[df.index]
        first_adm_date = first_adm_date.loc[df.index]
        uncertainty = (last_disch_date.dt.year - first_adm_date.dt.year) // 2
        shift = (uncertainty + 89).astype('timedelta64[Y]')
        df.loc[:, c_dob] = df[c_dob].mask(
            (last_disch_date.dt.year - df[c_dob].dt.year) > 150,
            first_adm_date - shift)

        subject_dob = df[c_dob].dt.normalize().to_dict()
        # TODO: check https://mimic.mit.edu/docs/iii/about/time/
        eth_mapper = self.scheme.ethnicity_mapper(target_scheme)

        def eth2vec(eth):
            code = eth_mapper.map_codeset(eth)
            return eth_mapper.codeset2vec(code)

        subject_eth = df[c_eth].map(eth2vec).to_dict()

        return subject_dob, subject_gender, subject_eth

    def adm_extractor(self, subject_ids):
        c_subject_id = self.colname["adm"].subject_id
        df = self.df["adm"]
        df = df[df[c_subject_id].isin(subject_ids)]
        return {
            subject_id: subject_df.index.tolist()
            for subject_id, subject_df in df.groupby(c_subject_id)
        }

    def dx_codes_extractor(self, admission_ids_list,
                           target_scheme: DatasetScheme):
        c_adm_id = self.colname["dx"].admission_id
        c_code = self.colname["dx"].code

        df = self.df["dx"]
        df = df[df[c_adm_id].isin(admission_ids_list)]

        codes_df = {
            adm_id: codes_df
            for adm_id, codes_df in df.groupby(c_adm_id)
        }
        empty_vector = target_scheme.dx.empty_vector()
        mapper = self.scheme.dx_mapper(target_scheme)

        def _extract_codes(adm_id):
            _codes_df = codes_df.get(adm_id)
            if _codes_df is None:
                return (adm_id, empty_vector)
            codeset = mapper.map_codeset(_codes_df[c_code])
            return (adm_id, mapper.codeset2vec(codeset))

        return dict(map(_extract_codes, admission_ids_list))

    def dx_codes_history_extractor(self, dx_codes, admission_ids,
                                   target_scheme):
        for subject_id, subject_admission_ids in admission_ids.items():
            _adm_ids = sorted(subject_admission_ids)
            vec = target_scheme.dx.empty_vector()
            yield (_adm_ids[0], vec)

            for prev_adm_id, adm_id in zip(_adm_ids[:-1], _adm_ids[1:]):
                if prev_adm_id in dx_codes:
                    vec = vec.union(dx_codes[prev_adm_id])
                yield (adm_id, vec)

    def outcome_extractor(self, dx_codes, target_scheme):
        return zip(dx_codes.keys(),
                   map(target_scheme.outcome.mapcodevector, dx_codes.values()))


class CPRDDataset(MIMIC3Dataset):
    static_info_class: ClassVar[Type[StaticInfo]] = CPRDStaticInfo

    def subject_info_extractor(self, subject_ids, target_scheme):

        static_df = self.df['static']
        c_gender = self.colname["static"].gender
        c_eth = self.colname["static"].ethnicity
        c_imd = self.colname["static"].imd_decile
        c_dob = self.colname["static"].date_of_birth

        static_df = static_df.loc[subject_ids]
        gender = static_df[c_gender].map(self.scheme.gender.codeset2vec)
        subject_gender = gender.to_dict()

        subject_dob = static_df[c_dob].dt.normalize().to_dict()
        subject_eth = dict()
        eth_mapper = self.scheme.ethnicity_mapper(target_scheme)
        eth_map = lambda eth: eth_mapper.codeset2vec(
            eth_mapper.map_codeset(eth))
        subject_eth = static_df[c_eth].map(eth_map).to_dict()

        subject_imd = static_df[c_imd].map(
            self.scheme.imd.codeset2vec).to_dict()
        return subject_dob, subject_gender, subject_eth, subject_imd

    def _load_dataframes(self):
        config = self.config.copy()
        colname = self.colname["adm"]

        df = pd.read_csv(translate_path(config.path), sep='\t', dtype=str)

        def listify(s):
            return list(map(lambda e: e.strip(), s.split(',')))

        adm_tups = []
        dx_tups = []
        demo_tups = []
        admission_id = 0
        for subject_id, _subj_df in df.groupby(colname.subject_id):
            subject_id = int(subject_id)
            assert len(_subj_df) == 1, "Each patient should have a single row"
            subject = _subj_df.iloc[0].to_dict()
            codes = listify(subject[colname.code])
            year_month = listify(subject[colname.dischtime])

            # To infer date-of-birth
            age0 = int(float(listify(subject[colname.age_at_dischtime])[0]))
            year_month0 = pd.to_datetime(year_month[0]).normalize()
            date_of_birth = year_month0 + pd.DateOffset(years=-age0)
            gender = subject[colname.gender]
            imd = subject[colname.imd_decile]
            ethnicity = subject[colname.ethnicity]
            demo_tups.append(
                (subject_id, date_of_birth, gender, imd, ethnicity))
            # codes aggregated by year-month.
            dx_codes_ym_agg = defaultdict(set)

            for code, ym in zip(codes, year_month):
                ym = pd.to_datetime(ym).normalize()
                dx_codes_ym_agg[ym].add(code)
            for disch_date in sorted(dx_codes_ym_agg.keys()):
                admit_date = disch_date + pd.DateOffset(days=-1)
                adm_tups.append(
                    (subject_id, admission_id, admit_date, disch_date))

                dx_codes = dx_codes_ym_agg[disch_date]
                dx_tups.extend([(admission_id, dx_code)
                                for dx_code in dx_codes])
                admission_id += 1

        adm_keys = ('subject_id', 'admission_id', 'admittime', 'dischtime')
        dx_keys = ('admission_id', 'code')
        demo_keys = ('subject_id', 'date_of_birth', 'gender', 'imd_decile',
                     'ethnicity')

        adm_cols = ColumnNames.make(
            {k: colname._asdict().get(k, k)
             for k in adm_keys})
        dx_cols = ColumnNames.make(
            {k: colname._asdict().get(k, k)
             for k in dx_keys})
        demo_cols = ColumnNames.make(
            {k: colname._asdict().get(k, k)
             for k in demo_keys})
        adm_cols = adm_cols._replace(index=adm_cols.admission_id)
        demo_cols = demo_cols._replace(index=demo_cols.subject_id)

        adm_df = pd.DataFrame(adm_tups,
                              columns=list(
                                  map(adm_cols._asdict().get, adm_keys)))
        adm_df = adm_df.astype({
            adm_cols.admission_id: int,
            adm_cols.subject_id: int
        }).set_index(adm_cols.index)

        dx_df = pd.DataFrame(dx_tups,
                             columns=list(map(dx_cols._asdict().get, dx_keys)))
        dx_df = dx_df.astype({dx_cols.admission_id: int})

        demo_df = pd.DataFrame(demo_tups,
                               columns=list(
                                   map(demo_cols._asdict().get, demo_keys)))
        demo_df = demo_df.astype({
            demo_cols.subject_id: int
        }).set_index(demo_cols.index)

        self.df = {'adm': adm_df, 'dx': dx_df, 'static': demo_df}
        self.colname = {'adm': adm_cols, 'dx': dx_cols, 'static': demo_cols}
        self._match_admissions_with_demographics(self.df, self.colname)

    def to_subjects(self, subject_ids, num_workers, demographic_vector_config,
                    target_scheme: DatasetScheme, **kwargs):

        (subject_dob, subject_gender, subject_eth,
         subject_imd) = self.subject_info_extractor(subject_ids, target_scheme)
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

        logging.debug('Compiling admissions...')
        c_admittime = self.colname['adm'].admittime
        c_dischtime = self.colname['adm'].dischtime
        adf = self.df['adm']
        adm_dates = dict(
            zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))

        def gen_admission(i):
            return Admission(admission_id=i,
                             admission_dates=adm_dates[i],
                             dx_codes=dx_codes[i],
                             dx_codes_history=dx_codes_history[i],
                             outcome=outcome[i],
                             observables=None,
                             interventions=None)

        def _gen_subject(subject_id):

            _admission_ids = admission_ids[subject_id]
            # for subject_id, subject_admission_ids in admission_ids.items():
            _admission_ids = sorted(_admission_ids,
                                    key=lambda aid: adm_dates[aid][0])
            static_info = self.static_info_class(
                date_of_birth=subject_dob[subject_id],
                gender=subject_gender[subject_id],
                ethnicity=subject_eth[subject_id],
                imd=subject_imd[subject_id],
                demographic_vector_config=demographic_vector_config)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                admissions = list(executor.map(gen_admission, _admission_ids))
            return Patient(subject_id=subject_id,
                           admissions=admissions,
                           static_info=static_info)

        return list(map(_gen_subject, subject_ids))


class MIMIC4Dataset(MIMIC3Dataset):
    scheme: MIMIC4DatasetScheme

    def _dx_fix_icd_dots(self):
        c_code = self.colname["dx"].code
        c_version = self.colname["dx"].version
        df = self.df["dx"]
        df = df.assign(**{c_code: df[c_code].str.strip()})
        self.df['dx'] = df
        for version, scheme in self.scheme.dx.items():
            if isinstance(scheme, ICDCommons):
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
                    f'({len(unrecognised)/len(codeset):.2%})')
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
    scalers_history: Dict[str, Callable] = eqx.field(init=False)
    outlier_remover_history: Dict[str, Callable] = eqx.field(init=False)
    seconds_scaler: ClassVar[float] = 1 / 3600.0  # convert seconds to hours

    def __post_init__(self):
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

    @staticmethod
    def _set_relative_times(df_dict,
                            colname,
                            df_name,
                            time_cols,
                            seconds_scaler=1 / 3600.0):
        c_admittime = colname["adm"].admittime
        c_adm_id = colname[df_name].admission_id

        target_df = df_dict[df_name]
        adm_df = df_dict["adm"][[c_admittime]]
        df = target_df.merge(adm_df,
                             left_on=c_adm_id,
                             right_index=True,
                             how='left')
        df_colname = colname[df_name]._asdict()
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

        MIMIC4Dataset._load_dataframes(self)
        colname = self.colname.copy()
        scheme = self.scheme
        df = self.df

        logging.debug("Time casting..")
        # Cast timestamps for intervensions
        for time_col in ("start_time", "end_time"):
            for file in ("int_proc", "int_input"):
                col = colname[file]._asdict()[time_col]
                df[file][col] = dd.to_datetime(df[file][col])

        # Cast timestamps for observables
        df["obs"][colname["obs"].timestamp] = dd.to_datetime(
            df["obs"][colname["obs"].timestamp])
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

        # self.df["adm"] = self._adm_remove_subjects_with_overlapping_admissions(
        #     self.df["adm"], self.colname["adm"])
        self._int_input_remove_subjects_with_nans()
        self._set_relative_times(df, colname, "int_proc",
                                 ["start_time", "end_time"],
                                 self.seconds_scaler)
        self._set_relative_times(df, colname, "int_input",
                                 ["start_time", "end_time"],
                                 self.seconds_scaler)
        self._set_relative_times(df,
                                 colname,
                                 "obs", ["timestamp"],
                                 seconds_scaler=self.seconds_scaler)
        self.df = {k: try_compute(v) for k, v in df.items()}
        self.colname = colname
        logging.debug("[DONE] Dataframes validation and time conversion")

    def to_subjects(self, subject_ids: List[int], num_workers: int,
                    demographic_vector_config: DemographicVectorConfig,
                    leading_observable_config: LeadingObservableConfig,
                    target_scheme: MIMIC4ICUDatasetScheme, **kwargs):

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
            interventions = interventions.segment_proc(proc_repr)
            interventions = interventions.segment_input()
            obs = observables[i]
            lead_obs = obs.make_leading_observable(leading_observable_config)
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

            static_info = self.static_info_class(
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
            time = x[c_time].iloc[0].item()
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


class SyntheticDataset(MIMIC3Dataset):

    @classmethod
    def make_synthetic_admissions(cls, colname, n_subjects):
        rng = random.Random(0)

        d1 = datetime.strptime('1/1/1950 1:30 PM', '%m/%d/%Y %I:%M %p')
        d2 = datetime.strptime('1/1/2050 4:50 AM', '%m/%d/%Y %I:%M %p')
        adms = []
        for subject_id in range(n_subjects):
            adm_d1 = random_date(d1, d2, rng)

            n_admissions = rng.randint(1, 10)
            adm_dates = sorted([
                random_date(adm_d1, d2, rng) for _ in range(2 * n_admissions)
            ])
            admittime = adm_dates[:n_admissions:2]
            dischtime = adm_dates[1:n_admissions:2]
            adms.append((subject_id, admittime, dischtime))
        df = pd.DataFrame(adms,
                          columns=list(
                              map(colname._asdict().get,
                                  ('subject_id', 'admittime', 'dischtime'))))
        df.index.names = [colname.index]
        return df

    @classmethod
    def make_synthetic_dx(cls, dx_colname, dx_source_scheme, n_dx_codes,
                          adm_df, adm_colname):
        rng = random.Random(0)
        n_dx_codes = rng.choices(range(n_dx_codes), k=len(adm_df))
        codes = []
        for i, adm_id in enumerate(adm_df.index):
            n_codes = n_dx_codes[i]
            codes.extend(
                (adm_id, c)
                for c in rng.choices(dx_source_scheme.codes, k=n_codes))

        return pd.DataFrame(codes, columns=[dx_colname.index, dx_colname.code])

    @classmethod
    def make_synthetic_demographic(cls,
                                   demo_colname,
                                   adm_df,
                                   adm_colname,
                                   ethnicity_scheme=None,
                                   gender_scheme=None):
        rng = random.Random(0)
        subject_ids = adm_df[adm_colname.subject_id].unique()
        demo = []
        for subject_id, df in adm_df.groupby(adm_colname.subject_id):
            first_adm = df[adm_colname.admittime].min()
            if gender_scheme is None:
                gender = None
            else:
                gender = rng.choices(gender_scheme.codes, k=1)[0]

            if ethnicity_scheme is None:
                ethnicity = None
            else:
                ethnicity = rng.choices(ethnicity_scheme.codes, k=1)[0]

            date_of_birth = random_date(first_adm - relativedelta(years=100),
                                        first_adm - relativedelta(years=18),
                                        rng)

            row = (subject_id, date_of_birth) + (c for c in (gender, ethnicity)
                                                 if c is not None)

        columns = [demo_colname.subject_id, demo_colname.date_of_birth]
        if gender_scheme is not None:
            columns.append(demo_colname.gender)
        if ethnicity_scheme is not None:
            columns.append(demo_colname.ethncity)

        df = pd.DataFrame(demo, columns=columns)
        return df


default_config_files = {
    'M3': f'{_META_DIR}/mimic3_meta.json',
    'M3CV': f'{_META_DIR}/mimic3cv_meta.json',
    'M4': f'{_META_DIR}/mimic4_meta.json',
    'CPRD': f'{_META_DIR}/cprd_meta.json',
    'M4ICU': f'{_META_DIR}/mimic4icu_meta.json',
}


def load_dataset_scheme(tag) -> DatasetScheme:
    conf = load_config(default_config_files[tag])
    scheme_conf = conf['scheme']
    return DatasetScheme.from_dict(scheme_conf, conf['scheme_classname'])


def load_dataset_config(tag: str = None,
                        config: AbstractConfig = None,
                        **init_kwargs):
    if config is not None:
        tag = config.tag
    else:
        config = load_config(default_config_files[tag])
        config = AbstractConfig.from_dict(config)
        config = config.update(**init_kwargs)
    return config


def load_dataset(tag: str = None,
                 config: AbstractConfig = None,
                 **init_kwargs):
    config = load_dataset_config(tag=tag, config=config, **init_kwargs)
    if tag is None:
        tag = config.tag
    if tag in ('M3', 'M3CV'):
        return MIMIC3Dataset(config)
    if tag == 'M4':
        return MIMIC4Dataset(config)
    if tag == 'CPRD':
        return CPRDDataset(config)
    if tag == 'M4ICU':
        return MIMIC4ICUDataset(config)
