"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations
from datetime import date
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple, Set, Callable, Optional, Union, List
from absl import logging
import numpy as np

from .coding_scheme import AbstractScheme, NullScheme


@dataclass
class AbstractAdmission:
    """
    Abstract class for admission data.
    """
    admission_id: int  # Unique ID for each admission
    admission_dates: Tuple[date, date]  # (admission date, discharge date)

    @staticmethod
    def days(d1, d2):
        """Returns the number of days between two dates."""
        return (d1 - d2).days

    def admission_day(self, ref_date):
        """Returns the number of days between admission date and ref_date."""
        return self.days(self.admission_dates[0], ref_date)

    def discharge_day(self, ref_date):
        """Returns the number of days between discharge date and ref_date."""
        return self.days(self.admission_dates[1], ref_date)

    @property
    def length_of_stay(self):
        """Returns the length of stay in days.

        Note: The 0.5 added to address the case when the patient
        is admitted and discharged at the same day, then we assume 0.5 day as length of stay (12 hours)
        In general, this would generalize the assumption to:
        Admission time is fixed at the midnight 00:01
        while discharge time is fixed at the midday 12:01
        """

        return self.days(self.admission_dates[1],
                         self.admission_dates[0]) + 0.5

@dataclass
class Admission(AbstractAdmission):
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    dx_codes: Set[str] # Set of diagnostic codes
    pr_codes: Set[str] # Set of procedure codes
    dx_scheme: AbstractScheme # Coding scheme for diagnostic codes
    pr_scheme: AbstractScheme # Coding scheme for procedure codes


@dataclass
class StaticInfo:
    """
    StaticInfo class encapsulates the patient static information.
    """
    gender: Optional[Union[float, np.ndarray]] = None
    date_of_birth: Optional[date] = None
    idx_deprivation: Optional[float] = None
    ethnicity: Optional[str] = None
    ethnicity_scheme: AbstractScheme = NullScheme()

    def age(self, current_date: date):
        return (current_date - self.date_of_birth).days / 365.25


StaticInfoFlags = namedtuple("StaticInfoFlags",
                             ["gender", "age", "idx_deprivation", "ethnicity"],
                             defaults=(False, False, False, NullScheme()))


class Subject:
    """
    Subject class encapsulates the patient EHRs diagnostic/procedure codes.

    Notes:
        - Some admissions for particular patients have the same ADMITTIME.
    For these cases the one with earlier DISCHTIME will be merged to the other.
    """
    def __init__(self, subject_id: int, admissions: List[Admission],
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
        """This function handles the case when coding schemes are not consistent across admissions."""
        schemes = set()
        for s in subjects:
            for a in s.admissions:
                schemes.add(a.pr_scheme)
        return schemes

    @staticmethod
    def merge_overlaps(admissions):
        """Merge admissions with overlapping time intervals."""

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
    def from_dataset(cls, dataset: "lib.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()


