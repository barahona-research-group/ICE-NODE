"""."""

import os
from pathlib import Path
from typing import Dict
from absl import logging

import pandas as pd

from ..utils import load_config, LazyDict, translate_path

from .coding_scheme import code_scheme as C, ICDCommons

_DIR = os.path.dirname(__file__)
_PROJECT_DIR = Path(_DIR).parent.parent.absolute()
_META_DIR = os.path.join(_PROJECT_DIR, 'datasets_meta')


class AbstractEHRDataset:

    def __init__(self, adm_df, dx_df, pr_df, code_scheme, code_colname,
                 adm_colname, name, **kwargs):
        # Add ICD dots if they are absent.
        if isinstance(C[code_scheme["dx"]], ICDCommons):
            dx_scheme = C[code_scheme["dx"]]
            col = code_colname["dx"]
            dx_df[col] = dx_df[col].apply(dx_scheme.add_dot)

        if pr_df is not None and isinstance(C[code_scheme["pr"]], ICDCommons):
            pr_scheme = C[code_scheme["pr"]]
            col = code_colname["pr"]
            pr_df[col] = pr_df[col].apply(pr_scheme.add_dot)

        dx_df = AbstractEHRDataset._validate_codes(dx_df, code_colname["dx"],
                                                   code_scheme["dx"])
        if pr_df is not None:
            pr_df = AbstractEHRDataset._validate_codes(pr_df,
                                                       code_colname["pr"],
                                                       code_scheme["pr"])
        self.name = name
        self.code_scheme = code_scheme
        self.adm_colname = adm_colname
        self.code_colname = code_colname
        self.adm_df = adm_df
        self.dx_df = dx_df
        self.pr_df = pr_df

    @staticmethod
    def _validate_codes(df, colname, scheme):
        codeset = set(df[colname])
        validset = set(C[scheme].codes)
        unrecognised = codeset - validset
        if len(unrecognised) > 0:
            logging.warning(f"""
                Unrecognised {type(C[scheme])} codes ({len(unrecognised)})
                to be removed: {sorted(unrecognised)}""")

            # Data Loss!
            df = df[df[colname].isin(validset)]

        return df

    def to_dict(self):
        col = self.adm_colname
        adm_id_col = col["admission_id"]
        admt_col = col["admittime"]
        dist_col = col["dischtime"]
        dx_col = self.code_colname["dx"]
        pr_col = self.code_colname.get("pr")
        dx_scheme = self.code_scheme["dx"]
        pr_scheme = self.code_scheme.get("pr", "none")

        adms = {}
        # Admissions
        for subj_id, subj_adms_df in self.adm_df.groupby(col["subject_id"]):
            subj_adms = {}

            for idx, adm_row in subj_adms_df.iterrows():
                adm_id = adm_row[adm_id_col]
                subj_adms[adm_id] = {
                    'admission_id': adm_id,
                    'admission_dates': (adm_row[admt_col], adm_row[dist_col]),
                    'dx_codes': set(),
                    'dx_scheme': dx_scheme,
                    'pr_codes': set(),
                    'pr_scheme': pr_scheme
                }
            adms[subj_id] = {'subject_id': subj_id, 'admissions': subj_adms}

        # dx concepts
        for subj_id, subj_dx_df in self.dx_df.groupby(col["subject_id"]):
            for adm_id, codes_df in subj_dx_df.groupby(adm_id_col):
                dx_codes = set(codes_df[dx_col])
                adms[subj_id]['admissions'][adm_id]['dx_codes'] = dx_codes

        # pr concepts
        if self.pr_df is None:
            return adms
        for subj_id, subject_pr_df in self.pr_df.groupby(col["subject_id"]):
            for adm_id, codes_df in subject_pr_df.groupby(adm_id_col):
                pr_codes = set(codes_df[pr_col])
                adms[subj_id]['admissions'][adm_id]['pr_codes'] = pr_codes

        return adms


class MIMICDataset(AbstractEHRDataset):

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
        return MIMICDataset(**meta)


datasets = LazyDict({
    'M3':
    lambda: MIMICDataset.from_meta_json(f'{_META_DIR}/mimic3_meta.json'),
    'M4':
    lambda: MIMICDataset.from_meta_json(f'{_META_DIR}/mimic4_meta.json')
})
