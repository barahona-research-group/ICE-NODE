"""."""

import os
from pathlib import Path
from typing import Dict
from absl import logging

import pandas as pd

from ..utils import load_config, LazyDict, translate_path

from .coding_scheme import code_scheme as C, ICDCommons, CodeMapper

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')


class AbstractSchemeEHRDataset:

    def __init__(self, df, code_scheme, target_scheme, code_colname,
                 code_scheme_colname, adm_colname, name, **kwargs):
        for info_type, scheme_map in code_scheme.items():
            if info_type not in df or df[info_type] is None:
                continue

            c_code = code_colname[info_type]
            c_scheme = code_scheme_colname[info_type]

            for sch_name in scheme_map.values():
                s_scheme = C[sch_name]
                if isinstance(s_scheme, ICDCommons):
                    df[info_type][c_code] = df[info_type][c_code].apply(
                        s_scheme.add_dot)

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
                    'admission_dates': (adm_row[admt_col], adm_row[dist_col])
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
            code_att = codes_attribute[info]
            scheme_att = scheme_attribute[info]
            df = self.df[info]

            for subj_id, subj_df in df.groupby(col["subject_id"]):
                for adm_id, codes_df in subj_df.groupby(adm_id_col):
                    scheme = set(scheme_map[str(s)]
                                 for s in codes_df[scheme_col])
                    assert len(
                        scheme
                    ) == 1, "Inconsistent code scheme for one admission!"
                    scheme = scheme.pop()
                    codeset = set(codes_df[code_col])
                    _adm = adms[subj_id]['admissions'][adm_id]
                    _adm[code_att] = codeset
                    _adm[scheme_att] = scheme

        return adms


class ConsistentSchemeEHRDataset(AbstractSchemeEHRDataset):

    def __init__(self, df, code_scheme, code_colname, adm_colname, name,
                 **kwargs):

        for info_type, scheme_name in code_scheme.items():

            if info_type not in df or df[info_type] is None:
                continue

            s_scheme = C[scheme_name]

            # 1. Add ICD dots if they are absent.
            if isinstance(s_scheme, ICDCommons):
                col = code_colname[info_type]
                _df = df[info_type]
                _df[col] = _df[col].apply(s_scheme.add_dot)
                df[info_type] = _df

            # 2. Validate codes
            df[info_type] = SingleSchemeEHRDataset._validate_codes(
                df[info_type], code_colname[info_type], code_scheme[info_type])

        self.name = name
        self.code_scheme = code_scheme
        self.adm_colname = adm_colname
        self.code_colname = code_colname
        self.df = df


class MIMIC3Dataset(SingleSchemeEHRDataset):

    def __init__(self, base_dir: str, files: Dict[str, str],
                 adm_colname: Dict[str, str], code_scheme: Dict[str, str],
                 code_colname: Dict[str, str], **kwargs):

        adm_df = pd.read_csv(os.path.join(base_dir, files['adm']))

        # Cast columns of dates to datetime64
        adm_df[adm_colname['admittime']] = pd.to_datetime(
            adm_df[adm_colname['admittime']],
            infer_datetime_format=True).dt.normalize()
        adm_df[adm_colname['dischtime']] = pd.to_datetime(
            adm_df[adm_colname['dischtime']],
            infer_datetime_format=True).dt.normalize()

        dx_df = pd.read_csv(os.path.join(base_dir, files['dx']),
                            dtype={code_colname['dx']: str})

        if 'pr' in files:
            pr_df = pd.read_csv(os.path.join(base_dir, files['pr']),
                                dtype={code_colname['pr']: str})
        else:
            pr_df = None

        super().__init__(code_scheme=code_scheme,
                         adm_colname=adm_colname,
                         code_colname=code_colname,
                         adm_df=adm_df,
                         dx_df=dx_df,
                         pr_df=pr_df,
                         **kwargs)

    @staticmethod
    def from_meta_json(meta_fpath):
        meta = load_config(meta_fpath)
        meta['base_dir'] = os.path.expandvars(meta['base_dir'])
        return MIMIC3Dataset(**meta)


datasets = LazyDict({
    'M3':
    lambda: MIMIC3Dataset.from_meta_json(f'{_META_DIR}/mimic3_meta.json'),
    'M4v2.0':
    lambda: MIMIC3Dataset.from_meta_json(f'{_META_DIR}/mimic4v2.0_meta.json')
})
