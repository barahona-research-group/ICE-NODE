"""."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, ClassVar

import dask.dataframe as dd
import pandas as pd

from lib.ehr.tvx_concepts import (Patient, Admission, DemographicVectorConfig, StaticInfo)
from lib.ehr.dataset import Dataset, DatasetScheme




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