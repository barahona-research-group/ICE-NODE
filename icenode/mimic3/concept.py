from __future__ import annotations
from datetime import date
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from absl import logging
import pandas as pd

from .dag import CCSDAG


class DiagnosisAdmission:
    def __init__(self, admission_id: int, admission_dates: Tuple[date, date],
                 icd9_diag_codes: Set[str]):
        self.admission_id = admission_id
        self.admission_dates = admission_dates
        self.icd9_diag_codes = icd9_diag_codes

    def get_ccs_diag_multi_codes(self, dag: CCSDAG) -> Set[str]:
        return set(map(dag.get_diag_multi_ccs, self.icd9_diag_codes))


class DiagSubject:
    """
    Subject class encapsulates the patient EHRs diagnostic codes.

    Notes:
        - Some admissions for particular patients have the same ADMITTIME.
    For these cases the one with earlier DISCHTIME will be merged to the other.
    """
    def __init__(self, subject_id: int, admissions: List[DiagnosisAdmission]):
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
                s_codes = set.union(s_adm.icd9_diag_codes, adm.icd9_diag_codes)
                s_adm = DiagnosisAdmission(admission_id=s_adm.admission_id,
                                           admission_dates=s_interval,
                                           icd9_diag_codes=s_codes)
                super_admissions.append(s_adm)
            else:
                super_admissions.append(adm)

        return super_admissions


    @classmethod
    def days(cls, d1, d2):
        return (d1.to_pydatetime() - d2.to_pydatetime()).days

    @classmethod
    def to_list(cls, adm_df: pd.DataFrame, diag_df: pd.DataFrame):
        ehr = {}
        # Admissions
        for subject_id, subject_admissions_df in adm_df.groupby('SUBJECT_ID'):
            subject_admissions = {}
            for adm_row in subject_admissions_df.itertuples():
                subject_admissions[adm_row.HADM_ID] = {
                    'admission_id': adm_row.HADM_ID,
                    'admission_dates': (adm_row.ADMITTIME, adm_row.DISCHTIME),
                    'icd9_diag_codes': set()
                }
            ehr[subject_id] = {
                'subject_id': subject_id,
                'admissions': subject_admissions
            }

        # Diag concepts
        for subject_id, subject_diag_df in diag_df.groupby('SUBJECT_ID'):
            for adm_id, codes_df in subject_diag_df.groupby('HADM_ID'):
                ehr[subject_id]['admissions'][adm_id]['icd9_diag_codes'] = set(
                    codes_df.ICD9_CODE)

        for subject_id in ehr.keys():
            ehr[subject_id]['admissions'] = list(
                map(lambda args: DiagnosisAdmission(**args),
                    ehr[subject_id]['admissions'].values()))
        return list(map(lambda args: cls(**args), ehr.values()))


class AdmissionInfo:
    def __init__(self, subject_id: int, admission_time: int, los: int,
                 admission_id: int, icd9_diag_codes: Set[str]):
        self.subject_id = subject_id
        # Time as days since the first admission
        self.admission_time = admission_time
        # Length of Stay
        self.los = los
        self.admission_id = admission_id
        self.icd9_diag_codes = icd9_diag_codes

    @classmethod
    def subject_to_admissions(
            cls, subject: DiagSubject) -> Dict[int, AdmissionInfo]:
        first_day_date = subject.admissions[0].admission_dates[0]
        adms = []
        for adm in subject.admissions:
            # days since first admission
            time = DiagSubject.days(adm.admission_dates[0],
                                    subject.admissions[0].admission_dates[0])

            los = DiagSubject.days(adm.admission_dates[1],
                                   adm.admission_dates[0]) + 0.5

            # This 0.5 means if a patient is admitted and discharged at
            # the same day, then we assume 0.5 day as length of stay (12 hours)
            # In general, this would generalize the assumption to:
            # Admissions during a day happen at the midnight 00:01
            # While discharges during a day happen at the afternoon 12:00

            adms.append(
                AdmissionInfo(subject_id=subject.subject_id,
                              admission_time=time,
                              los=los,
                              admission_id=adm.admission_id,
                              icd9_diag_codes=adm.icd9_diag_codes))
        return adms
