import pandas as pd
from icenode.ehr_model.mimic.concept import DiagSubject

def load_mimic_files(adm_file, diag_file):
    adm_df = pd.read_csv(adm_file)
    diag_df = pd.read_csv(diag_file, dtype={'ICD9_CODE': str})
    # Cast columns of dates to datetime64
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df['ADMITTIME'], infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df['DISCHTIME'], infer_datetime_format=True).dt.normalize()
    return DiagSubject.to_list(adm_df, diag_df)
