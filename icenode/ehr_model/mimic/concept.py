"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations
from datetime import date
from typing import Dict, List, Tuple, Set

import pandas as pd

from ..ccs_dag import ccs_dag


class DxAdmission:
    def __init__(self, admission_id: int, admission_dates: Tuple[date, date],
                 dx_icd9_codes: Set[str]):
        self.admission_id = admission_id
        self.admission_dates = admission_dates
        self.dx_icd9_codes = dx_icd9_codes


class DxSubject:
    """
    Subject class encapsulates the patient EHRs diagnostic codes.

    Notes:
        - Some admissions for particular patients have the same ADMITTIME.
    For these cases the one with earlier DISCHTIME will be merged to the other.
    """
    def __init__(self, subject_id: int, admissions: List[DxAdmission]):
        self.subject_id = subject_id

        admissions_disjoint = self.merge_overlaps(admissions)

        self.admissions = sorted(admissions_disjoint,
                                 key=lambda a: a.admission_dates[0])


#         for a1, a2 in zip(self.admissions[:-1], self.admissions[1:]):
#             if a1.admission_dates[0] == a2.admission_dates[0]:
#                 logging.info(f'same day admission: {self.subject_id}')
#             if a1.admission_dates[1] == a2.admission_dates[0]:
#                 logging.info(f'same day readmission: {self.subject_id}')

    @staticmethod
    def merge_overlaps(admissions):
        admissions = sorted(admissions, key=lambda adm: adm.admission_dates[0])

        super_admissions = [admissions[0]]
        for adm in admissions[1:]:
            s_adm = super_admissions[-1]
            (s_admittime, s_dischtime) = s_adm.admission_dates

            assert s_admittime <= s_dischtime, "Precedence of admittime violated!"

            admittime, dischtime = adm.admission_dates
            if admittime <= s_dischtime:
                super_admissions.pop()
                s_interval = (s_admittime, max(dischtime, s_dischtime))
                s_codes = set.union(s_adm.dx_icd9_codes, adm.dx_icd9_codes)
                s_adm = DxAdmission(admission_id=s_adm.admission_id,
                                    admission_dates=s_interval,
                                    dx_icd9_codes=s_codes)
                super_admissions.append(s_adm)
            else:
                super_admissions.append(adm)

        return super_admissions

    @classmethod
    def days(cls, d1, d2):
        return (d1.to_pydatetime() - d2.to_pydatetime()).days

    @classmethod
    def to_list(cls, adm_df: pd.DataFrame, dx_df: pd.DataFrame):
        ehr = {}
        # Admissions
        for subject_id, subject_admissions_df in adm_df.groupby('SUBJECT_ID'):
            subject_admissions = {}
            for adm_row in subject_admissions_df.itertuples():
                subject_admissions[adm_row.HADM_ID] = {
                    'admission_id': adm_row.HADM_ID,
                    'admission_dates': (adm_row.ADMITTIME, adm_row.DISCHTIME),
                    'dx_icd9_codes': set()
                }
            ehr[subject_id] = {
                'subject_id': subject_id,
                'admissions': subject_admissions
            }

        # dx concepts
        for subject_id, subject_dx_df in dx_df.groupby('SUBJECT_ID'):
            for adm_id, codes_df in subject_dx_df.groupby('HADM_ID'):
                ehr[subject_id]['admissions'][adm_id]['dx_icd9_codes'] = set(
                    codes_df.ICD9_CODE)

        for subject_id in ehr.keys():
            ehr[subject_id]['admissions'] = list(
                map(lambda args: DxAdmission(**args),
                    ehr[subject_id]['admissions'].values()))
        return list(map(lambda args: cls(**args), ehr.values()))


