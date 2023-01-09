"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations
from datetime import date
from collections import defaultdict
from typing import List, Tuple, Set, Callable, Dict, Optional
from absl import logging
from dataclasses import dataclass

import numpy as np

from .outcome import OutcomeExtractor


@dataclass
class AbstractAdmission:
    admission_id: int
    admission_dates: Tuple[date, date]

    @staticmethod
    def days(d1, d2):
        return (d1.to_pydatetime() - d2.to_pydatetime()).days

    def admission_day(self, ref_date):
        return self.days(self.admission_dates[0], ref_date)

    def discharge_day(self, ref_date):
        return self.days(self.admission_dates[1], ref_date)

    @property
    def length_of_stay(self):
        # Length of Stay
        # This 0.5 means if a patient is admitted and discharged at
        # the same day, then we assume 0.5 day as length of stay (12 hours)
        # In general, this would generalize the assumption to:
        # Admissions during a day happen at the midnight 00:01
        # While discharges during a day happen at the afternoon 12:00
        return self.days(self.admission_dates[1],
                         self.admission_dates[0]) + 0.5


@dataclass
class Admission(AbstractAdmission):
    dx_codes: Set[str]
    pr_codes: Set[str]
    dx_scheme: str
    pr_scheme: str


@dataclass
class StaticInfo:
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None
    idx_deprivation: Optional[str] = None
    ethnicity: Optional[str] = None


class Subject:
    """
    Subject class encapsulates the patient EHRs diagnostic/procedure codes.

    Notes:
        - Some admissions for particular patients have the same ADMITTIME.
    For these cases the one with earlier DISCHTIME will be merged to the other.
    """

    def __init__(self,
                 subject_id: int,
                 admissions: List[Admission],
                 static_info: StaticInfo):
        self.subject_id = subject_id
        self.static_info = static_info

        admissions_disjoint = self.merge_overlaps(admissions)

        self.admissions = sorted(admissions_disjoint,
                                 key=lambda a: a.admission_dates[0])

        for a1, a2 in zip(self.admissions[:-1], self.admissions[1:]):
            if a1.admission_dates[0] == a2.admission_dates[0]:
                logging.warning(f'same day admission: {self.subject_id}')
            if a1.admission_dates[1] == a2.admission_dates[0]:
                logging.warning(f'same day readmission: {self.subject_id}')

    @property
    def first_adm_date(self):
        return self.admissions[0].admission_dates[0]

    @staticmethod
    def _event_history(adms_sorted: List[Admission],
                       adm2codeset: Callable[[Admission], Set[str]],
                       absolute_dates=False):
        history = defaultdict(list)
        first_adm_date = adms_sorted[0].admission_dates[0]
        for adm in adms_sorted:
            for code in adm2codeset(adm):
                if absolute_dates:
                    history[code].append(adm.admission_dates)
                else:
                    history[code].append(adm.admission_day(first_adm_date),
                                         adm.discharge_day(first_adm_date))
        return history

    def dx_outcome_history(self,
                           dx_outcome: OutcomeExtractor,
                           absolute_dates=False):

        m = dx_outcome
        return self._event_history(
            self.admissions,
            lambda adm: m.map_codeset(adm.dx_codes, adm.dx_scheme),
            absolute_dates)

    @staticmethod
    def dx_schemes(subjects: List[Subject]):
        # This function handles the case when coding schemes are not consistent
        # across admissions.
        schemes = set()
        for s in subjects:
            for a in s.admissions:
                schemes.add(a.dx_scheme)
        return schemes

    @staticmethod
    def pr_schemes(subjects: List[Subject]):
        # This function handles the case when coding schemes are not consistent
        # across admissions.
        schemes = set()
        for s in subjects:
            for a in s.admissions:
                schemes.add(a.pr_scheme)
        return schemes

    @staticmethod
    def _event_frequency(subjects: List[Subject],
                         adm2codeset: Callable[[Admission], Set[str]],
                         index=None):
        counter = defaultdict(int)
        for subject in subjects:
            for adm in subject.admissions:
                codeset = adm2codeset(adm)
                for c in codeset:
                    counter[c] += 1

        if index:
            counter = {index[c]: counter[c] for c in counter}

        return counter

    @staticmethod
    def _event_frequency_vec(subjects: List[Subject],
                             adm2codeset: Callable[[Admission], Set[str]],
                             index: Dict[str, int]):
        freq_dict = Subject._event_frequency(subjects, adm2codeset, index)
        vec = np.zeros(len(index))
        for idx, count in freq_dict.items():
            vec[idx] = count
        return vec

    @staticmethod
    def dx_outcome_frequency_vec(subjects: List[Subject],
                                 dx_outcome: OutcomeExtractor):
        m = dx_outcome
        return Subject._event_frequency_vec(
            subjects=subjects,
            adm2codeset=lambda adm: m.map_codeset(adm.dx_codes, adm.dx_scheme),
            index=m.index)

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
                if len(s_adm.dx_codes) == 0:
                    s_adm.dx_scheme = adm.dx_scheme

                if len(s_adm.pr_codes) == 0:
                    s_adm.pr_scheme = adm.pr_scheme

                assert len(
                    adm.dx_codes
                ) == 0 or s_adm.dx_scheme == adm.dx_scheme, f"Inconsistent coding schemes ({s_adm.dx_scheme}, {adm.dx_scheme})"
                assert len(
                    adm.pr_codes
                ) == 0 or s_adm.pr_scheme == adm.pr_scheme, f"Inconsistent coding schemes ({s_adm.pr_scheme}, {adm.pr_scheme})"

                super_admissions.pop()
                s_interval = (s_admittime, max(dischtime, s_dischtime))
                s_dx_codes = set().union(s_adm.dx_codes, adm.dx_codes)
                s_pr_codes = set().union(s_adm.pr_codes, adm.pr_codes)

                s_adm = Admission(admission_id=s_adm.admission_id,
                                  admission_dates=s_interval,
                                  dx_codes=s_dx_codes,
                                  dx_scheme=s_adm.dx_scheme,
                                  pr_codes=s_pr_codes,
                                  pr_scheme=s_adm.pr_scheme)
                super_admissions.append(s_adm)
            else:
                super_admissions.append(adm)

        return super_admissions

    @classmethod
    def from_dataset(cls, dataset: "icenode.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()

