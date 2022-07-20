import os
from typing import Optional
import pandas as pd

from ..utils import load_config

class AbstractEHRDataset:
    pass


def create_patient_interface(
    processed_mimic_tables_dir: str,
    adm_df_name: str,
    dx_df_name: str,
    pr_df_name: str,
):
    adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
    # Cast columns of dates to datetime64
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df['ADMITTIME'], infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df['DISCHTIME'], infer_datetime_format=True).dt.normalize()
    dx_df = pd.read_csv(f'{processed_mimic_tables_dir}/dx_df.csv.gz',
                        dtype={'ICD9_CODE': str})

    patients = DxSubject.to_list(adm_df=adm_df, dx_df=dx_df)

    return DxInterface_JAX(patients)


class MIMICDataset(AbstractEHRDataset):

    def __init__(self,
                 base_dir: str,
                 adm_fname: str = 'adm_df.csv.gz',
                 admittime_colname: str = 'ADMITTIME',
                 dischtime_colname: str = 'DISCHTIME',
                 dx_fname: str = 'dx_df.csv.gz',
                 dx_code_colname: str = 'ICD9_CODE',
                 dx_scheme: str = 'dx_icd9',
                 pr_fname: Optional[str] = None,
                 pr_code_colname: Optional[str] = 'ICD9_CODE',
                 pr_scheme: Optional[str] = 'pr_icd9'):

        adm_df = pd.read_csv(os.path.join(base_dir, adm_fname))

        # Cast columns of dates to datetime64
        adm_df[admittime_colname] = pd.to_datetime(
            adm_df[admittime_colname],
            infer_datetime_format=True).dt.normalize()
        adm_df[dischtime_colname] = pd.to_datetime(
            adm_df[dischtime_colname],
            infer_datetime_format=True).dt.normalize()

        dx_df = pd.read_csv(os.path.join(base_dir, dx_fname),
                            dtype={dx_code_colname: str})

        if pr_fname and pr_code_colname:
            pr_df = pd.read_csv(os.path.join(base_dir, pr_fname),
                                dtype={pr_code_colname: str})
        else:
            pr_df = None

        self.dx_
        self.adm_df = adm_df
        self.dx_df = dx_df
        self.pr_df = pr_df

    @staticmethod
    def from_meta_json(meta_fpath):

