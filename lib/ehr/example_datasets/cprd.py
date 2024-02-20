# """."""
# from __future__ import annotations
#
# import logging
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor
#
# import pandas as pd
#
# from ._dataset_mimic3 import MIMIC3Dataset
# from .coding_scheme import CodingScheme
# from .concepts import (Patient, Admission, CPRDDemographicVectorConfig, CPRDStaticInfo)
# from .dataset import DatasetScheme, DatasetSchemeConfig
# from ..utils import translate_path
#
#
# class CPRDDatasetSchemeConfig(DatasetSchemeConfig):
#     imd: str = "CPRDIMDCategorical"
#
#
# class CPRDDatasetScheme(DatasetScheme):
#     imd: CodingScheme
#
#     def __init__(self, config: CPRDDatasetSchemeConfig, **kwargs):
#         super().__init__(config, **kwargs)
#         self.imd = CodingScheme.from_name(config.imd)
#
#     def demographic_vector_size(
#             self, demographic_vector_config: CPRDDemographicVectorConfig):
#         size = DatasetScheme.demographic_vector_size(
#             self, demographic_vector_config)
#         if demographic_vector_config.imd:
#             size += len(self.imd)
#         return size
#
#
# class CPRDDataset(MIMIC3Dataset):
#
#     def subject_info_extractor(self, subject_ids, target_scheme):
#
#         static_df = self.df['static']
#         c_gender = self.colname["static"].gender
#         c_eth = self.colname["static"].ethnicity
#         c_imd = self.colname["static"].imd_decile
#         c_dob = self.colname["static"].date_of_birth
#
#         static_df = static_df.loc[subject_ids]
#         gender = static_df[c_gender].map(self.scheme.gender.codeset2vec)
#         subject_gender = gender.to_dict()
#
#         subject_dob = static_df[c_dob].dt.normalize().to_dict()
#         subject_eth = dict()
#         eth_mapper = self.scheme.ethnicity_mapper(target_scheme)
#         subject_eth = static_df[c_eth].map(lambda eth: eth_mapper.codeset2vec(
#             eth_mapper.map_codeset(eth))).to_dict()
#
#         subject_imd = static_df[c_imd].map(
#             self.scheme.imd.codeset2vec).to_dict()
#         return subject_dob, subject_gender, subject_eth, subject_imd
#
#     def _load_dataframes(self):
#         config = self.config.copy()
#         colname = self.colname["adm"]
#
#         df = pd.read_csv(translate_path(config.path), sep='\t', dtype=str)
#
#         def listify(s):
#             return list(map(lambda e: e.strip(), s.split(',')))
#
#         adm_tups = []
#         dx_tups = []
#         demo_tups = []
#         admission_id = 0
#         for subject_id, _subj_df in df.groupby(colname.subject_id):
#             subject_id = int(subject_id)
#             assert len(_subj_df) == 1, "Each patient should have a single row"
#             subject = _subj_df.iloc[0].to_dict()
#             codes = listify(subject[colname.code])
#             year_month = listify(subject[colname.dischtime])
#
#             # To infer date-of-birth
#             age0 = int(float(listify(subject[colname.age_at_dischtime])[0]))
#             year_month0 = pd.to_datetime(year_month[0]).normalize()
#             date_of_birth = year_month0 + pd.DateOffset(years=-age0)
#             gender = subject[colname.gender]
#             imd = subject[colname.imd_decile]
#             ethnicity = subject[colname.ethnicity]
#             demo_tups.append(
#                 (subject_id, date_of_birth, gender, imd, ethnicity))
#             # codes aggregated by year-month.
#             dx_codes_ym_agg = defaultdict(set)
#
#             for code, ym in zip(codes, year_month):
#                 ym = pd.to_datetime(ym).normalize()
#                 dx_codes_ym_agg[ym].add(code)
#             for disch_date in sorted(dx_codes_ym_agg.keys()):
#                 admit_date = disch_date + pd.DateOffset(days=-1)
#                 adm_tups.append(
#                     (subject_id, admission_id, admit_date, disch_date))
#
#                 dx_codes = dx_codes_ym_agg[disch_date]
#                 dx_tups.extend([(admission_id, dx_code)
#                                 for dx_code in dx_codes])
#                 admission_id += 1
#
#         adm_keys = ('subject_id', 'admission_id', 'admittime', 'dischtime')
#         dx_keys = ('admission_id', 'code')
#         demo_keys = ('subject_id', 'date_of_birth', 'gender', 'imd_decile',
#                      'ethnicity')
#
#         adm_cols = ColumnNames.make({k: colname.get(k, k) for k in adm_keys})
#         dx_cols = ColumnNames.make({k: colname.get(k, k) for k in dx_keys})
#         demo_cols = ColumnNames.make({k: colname.get(k, k) for k in demo_keys})
#         adm_cols = adm_cols._replace(index=adm_cols.admission_id)
#         demo_cols = demo_cols._replace(index=demo_cols.subject_id)
#
#         adm_df = pd.DataFrame(adm_tups,
#                               columns=list(map(adm_cols.get, adm_keys)))
#         adm_df = adm_df.astype({
#             adm_cols.admission_id: int,
#             adm_cols.subject_id: int
#         }).set_index(adm_cols.index)
#
#         dx_df = pd.DataFrame(dx_tups, columns=list(map(dx_cols.get, dx_keys)))
#         dx_df = dx_df.astype({dx_cols.admission_id: int})
#
#         demo_df = pd.DataFrame(demo_tups,
#                                columns=list(map(demo_cols.get, demo_keys)))
#         demo_df = demo_df.astype({
#             demo_cols.subject_id: int
#         }).set_index(demo_cols.index)
#
#         self.df = {'adm': adm_df, 'dx_discharge': dx_df, 'static': demo_df}
#         self.colname = {'adm': adm_cols, 'dx_discharge': dx_cols, 'static': demo_cols}
#         self._match_admissions_with_demographics(self.df, self.colname)
#
#     def to_subjects(self, subject_ids, num_workers, demographic_vector_config,
#                     target_scheme: DatasetScheme, **kwargs):
#
#         (subject_dob, subject_gender, subject_eth,
#          subject_imd) = self.subject_info_extractor(subject_ids, target_scheme)
#         admission_ids = self.adm_extractor(subject_ids)
#         adm_ids_list = sum(map(list, admission_ids.values()), [])
#         logging.debug('Extracting dx_discharge codes...')
#         dx_codes = dict(self.dx_codes_extractor(adm_ids_list, target_scheme))
#         logging.debug('[DONE] Extracting dx_discharge codes')
#         logging.debug('Extracting dx_discharge codes history...')
#         dx_codes_history = dict(
#             self.dx_codes_history_extractor(dx_codes, admission_ids,
#                                             target_scheme))
#         logging.debug('[DONE] Extracting dx_discharge codes history')
#         logging.debug('Extracting outcome...')
#         outcome = dict(self.outcome_extractor(dx_codes, target_scheme))
#         logging.debug('[DONE] Extracting outcome')
#
#         logging.debug('Compiling admissions...')
#         c_admittime = self.colname['adm'].admittime
#         c_dischtime = self.colname['adm'].dischtime
#         adf = self.df['adm']
#         adm_dates = dict(
#             zip(adf.index, zip(adf[c_admittime], adf[c_dischtime])))
#
#         def gen_admission(i):
#             return Admission(admission_id=i,
#                              admission_dates=adm_dates[i],
#                              dx_codes=dx_codes[i],
#                              dx_codes_history=dx_codes_history[i],
#                              outcome=outcome[i],
#                              observables=None,
#                              interventions=None)
#
#         def _gen_subject(subject_id):
#             _admission_ids = admission_ids[subject_id]
#             # for subject_id, subject_admission_ids in admission_ids.items():
#             _admission_ids = sorted(_admission_ids,
#                                     key=lambda aid: adm_dates[aid][0])
#             static_info = CPRDStaticInfo(
#                 date_of_birth=subject_dob[subject_id],
#                 gender=subject_gender[subject_id],
#                 ethnicity=subject_eth[subject_id],
#                 imd=subject_imd[subject_id],
#                 demographic_vector_config=demographic_vector_config)
#
#             with ThreadPoolExecutor(max_workers=num_workers) as executor:
#                 admissions = list(executor.map(gen_admission, _admission_ids))
#             return Patient(subject_id=subject_id,
#                            admissions=admissions,
#                            static_info=static_info)
#
#         return list(map(_gen_subject, subject_ids))
