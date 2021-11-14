from __future__ import annotations
import logging
import math
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np
import pandas as pd

concept_logger = logging.getLogger("concept")


class LabTest:
    def __init__(self, test_id: int, unit: str, value: float):
        self.test_id = test_id
        self.unit = unit
        self.value = value

    def __str__(self):
        return f'{self.test_id}: {self.value} {self.unit}'


class Day:
    def __init__(self,
                 date: date = None,
                 icd9_diag_codes: Set[str] = set(),
                 icd9_proc_codes: Set[str] = set(),
                 lab_tests: Dict[str, Any] = dict()):
        self.date = date
        self.icd9_diag_codes = icd9_diag_codes
        self.icd9_proc_codes = icd9_proc_codes
        self.lab_tests = lab_tests

    DERIVED_AGE_HOLDER = 'age: ???'

    def __str__(self):
        lines = []
        lines.append(f'date: {self.date} | {self.DERIVED_AGE_HOLDER}')
        if self.icd9_diag_codes:
            lines.append(f'ICD9 Diagnosis: {" ".join(self.icd9_diag_codes)}')
        if self.icd9_proc_codes:
            lines.append(f'ICD9 Procedures: {" ".join(self.icd9_proc_codes)}')
        if self.lab_tests:
            lines.append("Lab tests: " + "\t".join(self.lab_tests))
        return '\n'.join(lines)


class Patient:
    """
    Patient class encapsulates the patient EHRs as a list of
    tuple(visit_date, Day) sorted chornologically by visit_date.
    """

    DATE_FORMAT = '%Y-%m-%d'

    def __init__(self, person_id: int, days: Dict[date, Day],
                 static_features: Dict[str, Any], birth_year: date):
        """

        """
        self.person_id = person_id
        self.days = days
        self.static_features = static_features
        self.birth_year = birth_year

    def __str__(self):
        lines = []
        lines.append(f'id: {self.person_id}')
        for day_dt, day in self.days.items():
            day_str = str(day).replace(Day.DERIVED_AGE_HOLDER,
                                       f'age: {self.age(day_dt)}')
            lines.extend(map(lambda ln: '\t' + ln, day_str.split('\n')))
        return '\n'.join(lines)

    def __eq__(self, other):
        return self.person_id == other.person_id and self.days == other.days

    def age(self, date: date) -> int:
        return int((date - self.birth_year).days / 365.25)

    @classmethod
    def pydate(cls, dt: str) -> date:
        """
        Create datetime.date object from date string of pattern '%Y-%m-%d'.
        """
        return datetime.strptime(dt, cls.DATE_FORMAT).date()

    @classmethod
    def strdate(cls, pydate: datetime) -> str:
        """
        Represent date object as string of format '%Y-%m-%d'.
        """
        return datetime.strftime(pydate, cls.DATE_FORMAT)

    @classmethod
    def to_list(cls, static_df: pd.DataFrame, ehr_diag_df: pd.DataFrame,
                ehr_proc_df: pd.DataFrame,
                ehr_lab_df: pd.DataFrame) -> List[Patient]:
        """
        Convert DataFrame representation of patients EHR to list of Patient
        representation List[Patient].

        The tabular layout of the static_df should have the following columns:
            person_id: same.
            birth_year: the year of birth of the patient
            ethnic_code: values that represent certain ethnic groups.
            gender:.

        The tablular layout of 'ehr_diag_df' and 'ehr_proc_df' should have the following columns:
            person_id: which corresponds to the Patient.person_id attribute.
            date: which corresponds to the visit_date.
            code: the ICD9 code.

        The tabular layout of 'ehr_lab_df' should have the following columns:
            person_id: which corresponds to the Patient.person_id attribute.
            date: which corresponds to the visit_date.
            test_id: a unique identifier for the test type.
            unit: the physical unit used in case the
                concept corresponds to a numerical value.
            value: the numerical value of the lab test.
        """
        assert all([
            col in static_df.columns
            for col in ['person_id', 'birth_year', 'ethnic_code', 'gender']
        ]), "Columns provided for static_df doesn't match the expected"

        assert (static_df.person_id.is_unique
                ), "person_id column should be unique in static_df"

        ehr = {}

        # Static features
        for row in static_df.itertuples():
            demographics = {
                'ethnic_code': row.ethnic_code,
                'gender': row.gender
            }
            ehr[row.person_id] = Patient(person_id=row.person_id,
                                         days=defaultdict(Day),
                                         static_features=demographics,
                                         birth_year=cls.pydate(row.birth_year))

        # Diag concepts
        for person_id, diag_df in ehr_diag_df.groupby('person_id'):
            for dt, day_df in diag_df.groupby('date'):
                ehr[person_id].days[dt].date = dt
                codes = set(day_df.code.tolist())
                ehr[person_id].days[dt].icd9_diag_codes = codes

        # Proc concepts
        for person_id, proc_df in ehr_proc_df.groupby('person_id'):
            for dt, day_df in proc_df.groupby('date'):
                ehr[person_id].days[dt].date = dt
                codes = set(day_df.code.tolist())
                ehr[person_id].days[dt].icd9_proc_codes = codes

        # Lab tests
        for person_id, lab_df in ehr_lab_df.groupby('person_id'):
            for dt, day_df in lab_df.groupby('date'):
                assert day_df.test_id.is_unique, "No duplicate test_id are expected in ai single day for a patient"
                ehr[person_id].days[dt].date = dt
                lab_tests = map(LabTest, day_df.test_id, day_df.unit,
                                day_df.value)
                ehr[person_id].days[dt].lab_tests = dict(
                    zip(day_df.test_id, lab_tests))

        return list(ehr.values())

    @classmethod
    def to_df(
            cls,
            patients_list: List[Patient]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert List[Patient] representation of patients EHR to DataFrame
        representation.

        Returns:
            static_df: DataFrame for the demographic info for patients, see 'to_list' for documentation of table columns.
            diag_df: DataFrame for diagnoses, see 'to_list' for documentation of table columns.
            proc_df: DataFrame for procedures, see 'to_list' for documentation of table columns.
            lab_df: DataFrame for lab tests, see 'to_list' for documentation of table columns.
        """
        diag_tuples = []
        proc_tuples = []
        lab_tuples = []
        static_tuples = []
        for p in patients_list:
            for dt, day in sorted(p.days.items()):
                str_date = cls.strdate(dt)
                for c in day.icd9_diag_codes:
                    diag_tuples.append((p.person_id, str_date, c))
                for c in day.icd9_proc_codes:
                    proc_tuples.append((p.person_id, str_date, c))
                for t in day.lab_tests.items():
                    lab_tuples.append(
                        (p.person_id, str_date, t.test_id, t.unit, t.value))

            static_tuples.append((p.person_id, cls.strdate(p.birth_year),
                                  p.static_features['ethnic_code'],
                                  p.static_features['gender']))

        diag_df = pd.DataFrame(diag_tuples,
                               columns=['person_id', 'date', 'code'])
        proc_df = pd.DataFrame(proc_tuples,
                               columns=['person_id', 'date', 'code'])
        lab_df = pd.DataFrame(
            lab_tuples,
            columns=['person_id', 'date', 'test_id', 'unit', 'value'])
        static_df = pd.DataFrame(
            static_tuples,
            columns=['person_id', 'birth_year', 'ethnic_code', 'gender'])
        return static_df, diag_df, proc_df, lab_df


    ZScoreScaler = Dict[str, Dict[str, Tuple[float, float]]]

    @classmethod
    def make_zscore_concept_scaler(
            cls, lab_df: pd.DataFrame) -> Patient.ZScoreScaler:
        """
        Z-score normalization for concepts with numerical values.

        Returns:
            A dictionary object of {test_id: {unit: (mean, std)}}, which
            will be used to standardize values by (x-mean)/std for each
            test_id for each unit.
        """
        numeric_df = lab_df[lab_df.value.notnull()].reset_index(drop=True)
        zscore_scaler: Patient.ZScoreScaler = defaultdict(dict)
        for i in set(numeric_df.test_id):
            test_df = numeric_df[numeric_df.test_id == i]
            for u in set(test_df.unit):
                test_unit_df = test_df[test_df.unit == u]
                vals = test_unit_df.value.to_numpy()
                zscore_scaler[i][u] = (vals.mean(), vals.std())
        return zscore_scaler

    @classmethod
    def apply_zscore_concept_scaler(
            cls, lab_df: pd.DataFrame,
            zscore_scaler: Patient.ZScoreScaler) -> pd.DataFrame:
        assert isinstance(
            lab_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        lab_df = lab_df.copy(deep=True)
        for test_id, units in zscore_scaler.items():
            test_df = lab_df[lab_df.test_id == test_id]
            for u, (mean, std) in units.items():
                unit_df = test_df[test_df.unit == u]
                vals = unit_df.value.to_numpy()
                lab_df.loc[unit_df.index, 'value'] = (vals - mean) / std
        return lab_df

    IQRFilter = Dict[str, Dict[str, Tuple[float, float]]]

    @classmethod
    def make_iqr_concept_filter(
            cls, lab_df: pd.DataFrame) -> Patient.IQRFilter:
        """
        Make outlier removal filter using Inter-quartile (IQR) method.
        https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/

        Args:
            patients_df: DataFrame representation of patients EHRs.
        Returns:
            A dictionary object of {test_id: {unit: (min, max)}}, which
            will be used to discard values outside (min, max) range for each
            snomed_code for each unit.
        """
        assert isinstance(
            lab_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        numeric_df = lab_df[lab_df.value.notnull()]
        iqr_filter: Patient.IQRFilter = defaultdict(dict)
        for i in set(numeric_df.test_id):
            test_df = numeric_df[numeric_df.test_id == i]
            for u in set(test_df.unit):
                test_unit_df = test_df[test_df.unit == u]
                vals = test_unit_df.value.to_numpy()
                upper_q = np.percentile(vals, 75)
                lower_q = np.percentile(vals, 25)
                iqr = (upper_q - lower_q) * 1.5
                iqr_filter[i][u] = (lower_q - iqr, upper_q + iqr)
        return iqr_filter

    @classmethod
    def apply_iqr_concept_filter(
            cls, lab_df: pd.DataFrame,
            iqr_filter: Patient.IQRFilter) -> pd.DataFrame:
        """
        Apply outlier removal filter using Inter-quartile (IQR) method.

        Args:
            lab_df: DataFrame representation of patients lab tests.
            iqr_filter: dictionary object of the filter, generated by make_iqr_concept_filter function.
        """
        assert isinstance(
            lab_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        drop = []
        for i, units in iqr_filter.items():
            test_df = lab_df[lab_df.test_id == i]
            for u, (mn, mx) in units.items():
                unit_df = test_df[test_df.unit == u]
                outliers = unit_df[(unit_df.value > mx) | (unit_df.value < mn)]
                drop.extend(outliers.index)
        return lab_df.drop(drop)
