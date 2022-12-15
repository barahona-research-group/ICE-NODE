"""."""

import os
from pathlib import Path
from typing import Dict
from collections import defaultdict
from absl import logging

import pandas as pd

from ..utils import load_config, LazyDict, translate_path, OOPError

from .coding_scheme import code_scheme as C, ICDCommons, CodeMapper

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')

StrDict = Dict[str, str]


class AbstractEHRDataset:

    @classmethod
    def from_meta_json(cls, meta_fpath):
        raise OOPError('Should be overriden')

    def to_dict(self):
        raise OOPError('Should be overrden')


class MIMIC4EHRDataset(AbstractEHRDataset):

    def __init__(self, df: Dict[str, pd.DataFrame],
                 code_scheme: Dict[str, StrDict], target_scheme: StrDict,
                 code_colname: StrDict, code_scheme_colname: StrDict,
                 adm_colname: StrDict, name: str, **kwargs):

        for info_type, scheme_map in code_scheme.items():
            if info_type not in df or df[info_type] is None:
                continue

            c_code = code_colname[info_type]
            c_scheme = code_scheme_colname[info_type]

            for sch_key, sch_name in scheme_map.items():
                s_scheme = C[sch_name]
                if isinstance(s_scheme, ICDCommons):
                    sch_mask = df[info_type][c_scheme].astype(str) == sch_key
                    df[info_type].loc[sch_mask, c_code] = df[info_type].loc[
                        sch_mask, c_code].apply(s_scheme.add_dot)

            df[info_type] = self._validate_codes(df[info_type], c_code,
                                                 c_scheme, scheme_map)

        self.name = name
        self.target_scheme = target_scheme
        self.code_scheme = code_scheme
        self.adm_colname = adm_colname
        self.code_colname = code_colname
        self.code_scheme_colname = code_scheme_colname
        self.df = df

    @staticmethod
    def load_dataframes(meta):
        m = meta
        files = m['files']
        base_dir = m['base_dir']
        adm_colname = m['adm_colname']
        code_colname = m['code_colname']
        code_sch_colname = m.get('code_scheme_colname')

        adm_df = pd.read_csv(os.path.join(base_dir, files['adm']))

        # Cast columns of dates to datetime64
        adm_df[adm_colname['admittime']] = pd.to_datetime(
            adm_df[adm_colname['admittime']],
            infer_datetime_format=True).dt.normalize()
        adm_df[adm_colname['dischtime']] = pd.to_datetime(
            adm_df[adm_colname['dischtime']],
            infer_datetime_format=True).dt.normalize()

        res = {'adm': adm_df}

        for info_t in ['dx', 'pr']:
            if info_t not in files or files[info_t] is None:
                res[info_t] = None
                continue
            dtype = {code_colname[info_t]: str}
            if code_sch_colname:
                dtype[code_sch_colname[info_t]] = str
            res[info_t] = pd.read_csv(os.path.join(base_dir, files[info_t]),
                                      dtype=dtype)

        return res

    @staticmethod
    def _validate_codes(df, code_col, scheme_col, scheme_map):
        drop_idx = []
        for scheme_key, sch_df in df.groupby(scheme_col):
            codeset = set(sch_df[code_col])
            scheme = C[scheme_map[str(scheme_key)]]
            scheme_codes = set(scheme.codes)

            unrecognised = codeset - scheme_codes
            if len(unrecognised) > 0:
                logging.warning(f"""
                    Unrecognised {type(scheme)} codes ({len(unrecognised)})
                    to be removed: {sorted(unrecognised)}""")

            # Data Loss!
            drop_idx.extend(sch_df[~sch_df[code_col].isin(scheme_codes)].index)

        return df.drop(index=drop_idx)

    def to_dict(self):
        col = self.adm_colname
        adm_id_col = col["admission_id"]
        admt_col = col["admittime"]
        dist_col = col["dischtime"]
        adms = {}
        # Admissions
        for subj_id, subj_adms_df in self.df['adm'].groupby(col["subject_id"]):
            subj_adms = {}

            for idx, adm_row in subj_adms_df.iterrows():
                adm_id = adm_row[adm_id_col]
                subj_adms[adm_id] = {
                    'admission_id': adm_id,
                    'admission_dates': (adm_row[admt_col], adm_row[dist_col]),
                    'dx_codes': set(),
                    'pr_codes': set(),
                    'dx_scheme': self.target_scheme['dx'],
                    'pr_scheme': self.target_scheme.get('pr', 'none')
                }
            adms[subj_id] = {'subject_id': subj_id, 'admissions': subj_adms}

        codes_attribute = {'dx': 'dx_codes', 'pr': 'pr_codes'}
        scheme_attribute = {'dx': 'dx_scheme', 'pr': 'pr_scheme'}

        for info in ["dx", "pr"]:
            if any(info not in d
                   for d in (self.code_colname, self.code_scheme, self.df)):
                continue
            code_col = self.code_colname[info]
            scheme_col = self.code_scheme_colname[info]
            scheme_map = self.code_scheme[info]
            t_sch = self.target_scheme[info]
            code_att = codes_attribute[info]
            scheme_att = scheme_attribute[info]
            df = self.df[info]

            for subj_id, subj_df in df.groupby(col["subject_id"]):
                for adm_id, codes_df in subj_df.groupby(adm_id_col):
                    codeset = set()
                    for sch_key, sch_codes_df in codes_df.groupby(scheme_col):
                        s_sch = scheme_map[sch_key]
                        m = CodeMapper.get_mapper(s_sch, t_sch)
                        codeset.update(m.map_codeset(sch_codes_df[code_col]))
                    adms[subj_id]['admissions'][adm_id][code_att] = codeset
        return adms

    @classmethod
    def from_meta_json(cls, meta_fpath):
        meta = load_config(meta_fpath)
        meta['base_dir'] = os.path.expandvars(meta['base_dir'])
        meta['df'] = cls.load_dataframes(meta)
        return cls(**meta)


class MIMIC3EHRDataset(MIMIC4EHRDataset):

    def __init__(self, df, code_scheme, code_colname, adm_colname, name,
                 **kwargs):
        target_scheme = {}
        code_scheme_colname = {}
        code_scheme_map = {}

        for info_type in ['dx', 'pr']:
            if info_type not in df or df[info_type] is None:
                continue
            scheme = code_scheme[info_type]
            target_scheme[info_type] = scheme
            code_scheme_colname[info_type] = scheme
            code_scheme_map[info_type] = {scheme: scheme}
            df[info_type][scheme] = scheme

        super().__init__(df=df,
                         code_scheme=code_scheme_map,
                         target_scheme=target_scheme,
                         code_colname=code_colname,
                         code_scheme_colname=code_scheme_colname,
                         adm_colname=adm_colname,
                         name=name,
                         **kwargs)


class CPRDEHRDataset(AbstractEHRDataset):

    def __init__(self, df, colname, code_scheme, target_scheme, name,
                 **kwargs):

        self.name = name
        self.target_scheme = target_scheme
        self.code_scheme = code_scheme
        self.colname = colname
        self.df = df

    def to_dict(self):
        listify = lambda s: list(map(lambda e: e.strip(), s.split(',')))
        col = self.colname
        adms = {}
        # Admissions
        for subj_id, subj_df in self.df.groupby(col["subject_id"]):

            subj_adms = {}

            codes = listify(subj_df[col["dx"]])
            year_month = listify(subj_df[col["year_month"]])
            _adms = defaultdict(set)
            for code, ym in zip(codes, year_month):
                ym = pd.to_datetime(ym, infer_datetime_format=True).normalize()
                _adms[ym].add(code)

            adm_dates = sorted(_adms.keys())

            for idx, adm_date in enumerate(adm_dates):
                disch_date = adm_date + pd.DateOffset(days=1)
                adm_codes = _adms[adm_date]

                m = CodeMapper.get_mapper(self.code_scheme, self.target_scheme)
                codeset = m.map_codeset(adm_codes)
                subj_adms[idx] = {
                    'admission_id': idx,
                    'admission_dates': (adm_date, disch_date),
                    'dx_codes': codeset,
                    'pr_codes': set(),
                    'dx_scheme': self.target_scheme['dx'],
                    'pr_scheme': 'none'
                }
            adms[subj_id] = {'subject_id': subj_id, 'admissions': subj_adms}

        return adms

    @classmethod
    def from_meta_json(cls, meta_fpath):
        meta = load_config(meta_fpath)
        filepath = translate_path(meta['filepath'])
        meta['df'] = pd.read_csv(filepath, sep='\t', dtype=str)
        return cls(**meta)


datasets = LazyDict({
    'M3':
    lambda: MIMIC3EHRDataset.from_meta_json(f'{_META_DIR}/mimic3_meta.json'),
    'M4':
    lambda: MIMIC4EHRDataset.from_meta_json(f'{_META_DIR}/mimic4_meta.json'),
    'CPRD':
    lambda: CPRDEHRDataset.from_meta_json(f'{_META_DIR}/cprd_meta.json')
})
