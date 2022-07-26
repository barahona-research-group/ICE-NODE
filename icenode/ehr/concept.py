"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations
from datetime import date
from collections import defaultdict
from typing import List, Tuple, Set, Optional
from absl import logging

from .coding_scheme import AbstractScheme, CodeMapper
from .dataset import AbstractEHRDataset


class AbstractAdmission:

    def __init__(self, admission_id: int, admission_dates: Tuple[date, date]):
        self.admission_id = admission_id
        self.admission_dates = admission_dates

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

        for a1, a2 in zip(self.admissions[:-1], self.admissions[1:]):
            if a1.admission_dates[0] == a2.admission_dates[0]:
                logging.warning(f'same day admission: {self.subject_id}')
            if a1.admission_dates[1] == a2.admission_dates[0]:
                logging.warning(f'same day readmission: {self.subject_id}')

    @property
    def first_adm_date(self):
        return self.admissions[0].admission_dates[0]

    def dx_history(self, dx_scheme=None, absolute_dates=False):
        history = defaultdict(list)
        mapper = CodeMapper.get_mapper(self.dx_scheme, dx_scheme)

        first_adm_date = self.admissions[0].admission_dates[0]
        for adm in self.admissions:
            for code in mapper.map_codeset(adm.dx_codes):
                if absolute_dates:
                    history[code].append(adm.admission_dates)
                else:
                    history[code].append(adm.admission_day(first_adm_date),
                                         adm.discharge_day(first_adm_date))
        return history

    def pr_history(self, pr_scheme=None):
        pass

    @staticmethod
    def _dx_scheme(admissions):
        s = admissions[0].dx_scheme
        assert all(a.dx_scheme == s
                   for a in admissions), "Scheme inconsistency"
        return s

    @property
    def dx_scheme(self):
        return self._dx_scheme(self.admissions)

    @staticmethod
    def _pr_scheme(admissions):
        s = admissions[0].pr_scheme
        assert all(a.pr_scheme == s
                   for a in admissions), "Scheme inconsistency"
        return s

    @property
    def pr_scheme(self):
        return self._pr_scheme(self.admissions)

    @staticmethod
    def dx_code_frequency(subjects: List[Subject],
                          dx_scheme: Optional[str] = None):
        src_scheme = subjects[0].dx_scheme
        assert all(s.dx_scheme == src_scheme
                   for s in subjects), "Scheme inconsistency"
        mapper = CodeMapper.get_mapper(src_scheme, dx_scheme)

        counter = defaultdict(int)
        for subject in subjects:
            for adm in subject.admissions:
                codeset = mapper.map_codeset(adm.dx_codes)
                for idx in map(mapper.t_index.get, codeset):
                    counter[idx] += 1

        # Return dictionary with zero-frequency codes added.
        return {idx: counter[idx] for idx in mapper.t_index.values()}

    @staticmethod
    def merge_overlaps(admissions):
        admissions = sorted(admissions, key=lambda adm: adm.admission_dates[0])
        dx_scheme = Subject._dx_scheme(admissions)
        pr_scheme = Subject._pr_scheme(admissions)

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
    def from_dataset(cls, dataset: AbstractEHRDataset):
        adms = dataset.to_dict()
        for subject_id in adms.keys():
            adms[subject_id]['admissions'] = list(
                map(lambda kwargs: Admission(**kwargs),
                    adms[subject_id]['admissions'].values()))
        return list(map(lambda kwargs: cls(**kwargs), adms.values()))
