"""."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, ClassVar, Callable, Type, Union
from collections import namedtuple
import pickle
from datetime import timedelta
import random
from abc import abstractmethod
import logging

import pandas as pd
import numpy as np
import equinox as eqx

from ..utils import load_config, translate_path
from ..base import Config, Module

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

StrDict = Dict[str, str]

keys = [
    'subject_id', 'admission_id', 'admittime', 'dischtime', 'adm_interval',
    'index', 'code', 'version', 'gender', 'ethnicity', 'imd_decile',
    'anchor_age', 'anchor_year', 'start_time', 'end_time', 'rate',
    'code_index', 'code_source_index', 'value', 'timestamp', 'date_of_birth',
    'age_at_dischtime'
]
default_raw_types = {
    'subject_id': str,
    'admission_id': str,
    'index': str,
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

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        if key not in self:
            return default
        return getattr(self, key)

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


class DatasetSchemeConfig(Config):
    dx: str
    ethnicity: str
    gender: str
    outcome: Optional[str] = None


class DatasetScheme(Module):
    config: DatasetSchemeConfig
    dx: AbstractScheme
    ethnicity: Ethnicity
    gender: Gender
    outcome: Optional[OutcomeExtractor] = None

    @property
    def scheme_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k != 'outcome' and isinstance(v, AbstractScheme)
        }

    @classmethod
    def _assert_valid_maps(cls, source, target):
        attrs = source.scheme_dict
        for attr in attrs:
            att_s_scheme = getattr(source, attr)
            att_t_scheme = getattr(target, attr)

            assert att_s_scheme.mapper_to(
                att_t_scheme
            ), f"Cannot map {attr} from {att_s_scheme} to {att_t_scheme}"

    def __init__(self, config: DatasetSchemeConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        config = self.config.as_dict()

        if config.get('outcome') is not None:
            self.outcome = OutcomeExtractor(config.pop('outcome'))

        for k, v in config.items():
            if isinstance(v, str):
                setattr(self, k, scheme_from_classname(v))

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
        return self.dx.mapper_to(target_scheme.dx)

    def ethnicity_mapper(self, target_scheme: DatasetScheme):
        return self.ethnicity.mapper_to(target_scheme.ethnicity)

    @property
    def supported_target_scheme_options(self):
        supproted_attr_targets = {
            k: (getattr(self, k).__class__.__name__, ) +
            getattr(self, k).supported_targets
            for k in self.scheme_dict
        }
        supported_outcomes = OutcomeExtractor.supported_outcomes(self.dx)
        supproted_attr_targets['outcome'] = supported_outcomes
        return supproted_attr_targets


class DatasetConfig(Config):
    path: str
    scheme: DatasetSchemeConfig
    scheme_classname: str
    colname: Dict[str, Dict[str, str]]
    files: Dict[str, str] = eqx.field(default_factory=dict)
    sample: Optional[int] = None
    meta_fpath: str = ''
    tag: str = ''
    overlapping_admissions: str = 'merge'


class Dataset(Module):
    df: Dict[str, pd.DataFrame]
    config: DatasetConfig
    scheme: DatasetScheme
    colname: Dict[str, ColumnNames]

    def __init__(self,
                 config: DatasetConfig = None,
                 config_path: str = None,
                 **kwargs):
        super().__init__(config=config, config_path=config_path, **kwargs)
        self.scheme = DatasetScheme.import_module(self.config.scheme,
                                                  self.config.scheme_classname)
        self.colname = {
            f: ColumnNames.make(m)
            for f, m in self.config.colname.items()
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
            dtypes[name][df.index.name] = df.index.dtype

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
