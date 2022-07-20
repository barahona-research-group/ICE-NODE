"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations
from datetime import date
from typing import Dict, List, Tuple, Set, Optional

import pandas as pd

from .coding_scheme import (code_scheme, code_scheme_cls, AbstractScheme,
                            HierarchicalScheme)


class AbstractAdmission:

    def __init__(self, admission_id: int, admission_dates: Tuple[date, date]):
        self.admission_id = admission_id
        self.admission_dates = admission_dates


class Admission(AbstractAdmission):

    def __init__(self,
                 admission_id: int,
                 admission_dates: Tuple[date, date],
                 dx_codes: Set[str] = set(),
                 pr_codes: Set[str] = set(),
                 dx_scheme: Optional[str] = None,
                 pr_scheme: Optional[str] = None):
        super().__init__(admission_id, admission_dates)
        self.dx_codes = dx_codes
        self.dx_scheme = dx_scheme
        self.pr_codes = pr_codes
        self.pr_scheme = pr_scheme


class Subject:
    """
    Subject class encapsulates the patient EHRs diagnostic/procedure codes.

    Notes:
        - Some admissions for particular patients have the same ADMITTIME.
    For these cases the one with earlier DISCHTIME will be merged to the other.
    """

    def __init__(self, subject_id: int, admissions: List[Admission]):
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
        dx_scheme = admissions[0].dx_scheme
        pr_scheme = admissions[0].pr_scheme

        assert all(a.dx_scheme == dx_scheme and a.pr_scheme == pr_scheme
                   for a in admissions), "Scheme inconsistency"
        super_admissions = [admissions[0]]
        for adm in admissions[1:]:
            s_adm = super_admissions[-1]
            (s_admittime, s_dischtime) = s_adm.admission_dates

            assert s_admittime <= s_dischtime, "Precedence of admittime violated!"

            admittime, dischtime = adm.admission_dates
            if admittime <= s_dischtime:
                super_admissions.pop()
                s_interval = (s_admittime, max(dischtime, s_dischtime))
                s_dx_codes = set().union(s_adm.dx_codes, adm.dx_codes)
                s_pr_codes = set().union(s_adm.pr_codes, adm.pr_codes)

                s_adm = Admission(admission_id=s_adm.admission_id,
                                  admission_dates=s_interval,
                                  dx_codes=s_dx_codes,
                                  dx_scheme=dx_scheme,
                                  pr_codes=s_pr_codes,
                                  pr_scheme=pr_scheme)
                super_admissions.append(s_adm)
            else:
                super_admissions.append(adm)

        return super_admissions

    @classmethod
    def days(cls, d1, d2):
        return (d1.to_pydatetime() - d2.to_pydatetime()).days

    @classmethod
    def to_list(cls,
                adm_df: pd.DataFrame,
                dx_df: pd.DataFrame,
                dx_colname: str,
                dx_scheme: str,
                pr_df: Optional[pd.DataFrame] = None,
                pr_colname: Optional[str] = None,
                pr_scheme: Optional[str] = None):
        ehr = {}
        # Admissions
        for subject_id, subject_admissions_df in adm_df.groupby('SUBJECT_ID'):
            subject_admissions = {}
            for adm_row in subject_admissions_df.itertuples():
                subject_admissions[adm_row.HADM_ID] = {
                    'admission_id': adm_row.HADM_ID,
                    'admission_dates': (adm_row.ADMITTIME, adm_row.DISCHTIME),
                    'dx_codes': set(),
                    'dx_scheme': dx_scheme,
                    'pr_codes': set(),
                    'pr_scheme': pr_scheme
                }
            ehr[subject_id] = {
                'subject_id': subject_id,
                'admissions': subject_admissions
            }

        # dx concepts
        for subject_id, subject_dx_df in dx_df.groupby('SUBJECT_ID'):
            for adm_id, codes_df in subject_dx_df.groupby('HADM_ID'):
                ehr[subject_id]['admissions'][adm_id]['dx_codes'] = set(
                    codes_df[dx_colname])
        if pr_df and pr_colname:
            # dx concepts
            for subject_id, subject_pr_df in pr_df.groupby('SUBJECT_ID'):
                for adm_id, codes_df in subject_pr_df.groupby('HADM_ID'):
                    ehr[subject_id]['admissions'][adm_id]['pr_codes'] = set(
                        codes_df[pr_colname])

        for subject_id in ehr.keys():
            ehr[subject_id]['admissions'] = list(
                map(lambda args: Admission(**args),
                    ehr[subject_id]['admissions'].values()))
        return list(map(lambda args: cls(**args), ehr.values()))
