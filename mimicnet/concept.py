from __future__ import annotations
import logging
import math
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from .dag import CCSDAG

concept_logger = logging.getLogger("concept")



class Test:
    def __init__(self, item_id: int, value: float, date: date):
        self.item_id = item_id
        self.value = value
        self.date = date


class HospitalAdmission:
    def __init__(self, admission_id: int, admission_dates: Tuple[date, date],
                 icd9_diag_codes: Set[str], icd9_proc_codes: Set[str]):
        self.admission_id = admission_id
        self.admission_dates = admission_dates
        self.icd9_diag_codes = icd9_diag_codes
        self.icd9_proc_codes = icd9_proc_codes

    DERIVED_AGE_HOLDER = 'age: ???'

    def __str__(self):
        lines = []
        lines.append(
            f'admission dates: {self.admission_dates} | {self.DERIVED_AGE_HOLDER}'
        )
        if self.icd9_diag_codes:
            lines.append(f'ICD9 Diagnosis: {" ".join(self.icd9_diag_codes)}')
        if self.icd9_proc_codes:
            lines.append(f'ICD9 Procedures: {" ".join(self.icd9_proc_codes)}')
        return '\n'.join(lines)

    def get_ccs_diag_multi_codes(self, dag: CCSDAG) -> Set[str]:
        return set(map(dag.get_diag_multi_ccs, self.icd9_diag_codes))

    def get_ccs_proc_multi_codes(self, dag: CCSDAG) -> Set[str]:
        return set(map(dag.get_proc_multi_ccs, self.icd9_proc_codes))

class Subject:
    """
    Subject class encapsulates the patient EHRs.
    """
    def __init__(self, subject_id: int, admissions: List[HospitalAdmission],
                 tests: List[Test], gender: str, ethnic_group: str,
                 date_of_birth: date):
        """

        """
        self.subject_id = subject_id
        self.admissions = sorted(admissions,
                                 key=lambda a: a.admission_dates[0])
        self.tests = sorted(tests, key=lambda t: t.date)
        self.ethnic_group = ethnic_group
        self.gender = gender
        self.date_of_birth = date_of_birth

    def __str__(self):
        lines = []
        lines.append(
            f'id: {self.subject_id}, gender: {self.gender}, ethnic_group: {self.ethnic_group}'
        )
        for i, adm in enumerate(self.admissions):
            interval = (adm.admission_dates[1] - adm.admission_dates[0]).days
            age = self.age(adm.admission_dates[0])
            lines.append(f'Admission #{i}, days: {interval}, age: {age}')
            lines.append(f'\tDiagnoses: {adm.icd9_diag_codes}')
            lines.append(f'\tProcedures: {adm.icd9_proc_codes}')

        tests_by_date = defaultdict(list)
        for t in self.tests:
            tests_by_date[t.date].append(
                f'\titem_id: {t.item_id}  value: {t.value}')

        for date, tests in tests_by_date.items():
            age = self.age(date)
            lines.append(f'Tests, age: {age}')
            lines.extend(tests)
        return '\n'.join(lines)

    @classmethod
    def days(cls, d1, d2):
        return (d1.to_pydatetime() - d2.to_pydatetime()).days

    @classmethod
    def years(cls, d1, d2):
        return cls.days(d1, d2) / 365.25

    def age(self, date: date) -> int:
        return self.years(date, self.date_of_birth)

    @classmethod
    def to_list(cls, static_df: pd.DataFrame, adm_df: pd.DataFrame,
                diag_df: pd.DataFrame, proc_df: pd.DataFrame,
                tests_df: pd.DataFrame) -> List[Subject]:
        """
        Convert DataFrame representation of patients EHR to list of Patient
        representation List[Patient].

        The tabular layout of static_df should have the following columns:
            SUBJECT_ID: patient id, linked to the Subject.subject_id.
            DOB: date of birth, linked to Subject.date_of_birth.
            ETHNIC_GROUP: values that represent certain ethnic groups.
            GENDER: 'M' or 'F'.

        The tabular layout of adm_df:
            SUBJECT_ID: same as above.
            HADM_ID: admission id, used to link with diagnosis and procedure codes in diag_df and proc_df, respectively.
            ADMITTIME: start of admission date.
            DISCHTIME: end of admission date.

        The tablular layout of 'diag_df' and 'proc_df' should have the following columns:
            SUBJECT_ID: same as above.
            HADM_ID: admission id, used to link with admissions in adm_df
            ICD9_CODE: an ICD9 code recorded for this SUBJECT_ID patient during HADM_ID admission.

        The tabular layout of 'tests_df' should have the following columns:
            SUBJECT_ID: same as above.
            ITEMID: the test type, linked to Test.item_id.
            DATE: test date (e.g. fluid extraction date).
            VALUE: the numerical value of the lab test.
        """
        assert set(static_df.columns) == set([
            'SUBJECT_ID', 'DOB', 'ETHNIC_GROUP', 'GENDER'
        ]), "Columns provided for static_df doesn't match the expected"

        # TODO: Add remaining tests.

        assert all(
            map(ptypes.is_datetime64_any_dtype, [
                static_df['DOB'], adm_df['ADMITTIME'], adm_df['DISCHTIME'],
                tests_df['DATE']
            ])), "Columns of dates should be casted to datetime64 type first."

        ehr = {}

        # Static features
        for row in static_df.itertuples():
            ehr[row.SUBJECT_ID] = {
                'subject_id': row.SUBJECT_ID,
                'admissions': {},
                'tests': [],
                'gender': row.GENDER,
                'ethnic_group': row.ETHNIC_GROUP,
                'date_of_birth': row.DOB
            }

        # Admissions
        for subject_id, subject_admissions_df in adm_df.groupby('SUBJECT_ID'):
            for adm_row in subject_admissions_df.itertuples():
                ehr[subject_id]['admissions'][adm_row.HADM_ID] = {
                    'admission_id': adm_row.HADM_ID,
                    'admission_dates': (adm_row.ADMITTIME, adm_row.DISCHTIME),
                    'icd9_diag_codes': set(),
                    'icd9_proc_codes': set()
                }

        # Diag concepts
        for subject_id, subject_diag_df in diag_df.groupby('SUBJECT_ID'):
            for adm_id, codes_df in subject_diag_df.groupby('HADM_ID'):
                ehr[subject_id]['admissions'][adm_id]['icd9_diag_codes'] = set(
                    codes_df.ICD9_CODE)

        # Proc concepts
        for subject_id, subject_proc_df in proc_df.groupby('SUBJECT_ID'):
            for adm_id, codes_df in subject_proc_df.groupby('HADM_ID'):
                ehr[subject_id]['admissions'][adm_id]['icd9_proc_codes'] = set(
                    codes_df.ICD9_CODE)

        for subject_id in ehr.keys():
            ehr[subject_id]['admissions'] = list(
                map(lambda args: HospitalAdmission(**args),
                    ehr[subject_id]['admissions'].values()))

        # Lab tests
        for subject_id, subject_tests_df in tests_df.groupby('SUBJECT_ID'):
            tests = []
            for tests_date, date_df in subject_tests_df.groupby('DATE'):
                for test_row in date_df.itertuples():
                    tests.append(
                        Test(item_id=test_row.ITEMID,
                             date=tests_date,
                             value=test_row.VALUE))
            ehr[subject_id]['tests'] = tests

        return list(map(lambda args: Subject(**args), ehr.values()))

    @classmethod
    def to_df(
        cls, patients_list: List[Subject]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.DataFrame]:
        """
        Convert List[Subject] representation of patients EHR to DataFrame
        representation.

        Returns:
            static_df: DataFrame for the demographic info for patients, see 'to_list' for documentation of table columns.
            adm_df: DataFrame for hospital admissions info of patients, see 'to_list' for documentation of table columns.
            diag_df: DataFrame for diagnoses, see 'to_list' for documentation of table columns.
            proc_df: DataFrame for procedures, see 'to_list' for documentation of table columns.
            tests_df: DataFrame for tests, see 'to_list' for documentation of table columns.
        """
        diag_t = []
        proc_t = []
        tests_t = []
        static_t = []
        adm_t = []
        for p in patients_list:
            static_t.append(
                (p.subject_id, p.date_of_birth, p.ethnic_group, p.gender))

            for adm in p.admissions:
                adm_t.append((p.subject_id, adm.admission_id,
                              adm.admission_dates[0], adm.admission_dates[1]))
                for diag_code in adm.icd9_diag_codes:
                    diag_t.append((p.subject_id, adm.admission_id, diag_code))
                for proc_code in adm.icd9_proc_codes:
                    proc_t.append((p.subject_id, adm.admission_id, proc_code))

            for test in p.tests:
                tests_t.append(
                    (p.subject_id, test.item_id, test.date, test.value))

        static_df = pd.DataFrame(
            static_t, columns=['SUBJECT_ID', 'DOB', 'ETHNIC_GROUP', 'GENDER'])
        adm_df = pd.DataFrame(
            adm_t, columns=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME'])
        diag_df = pd.DataFrame(diag_t,
                               columns=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
        proc_df = pd.DataFrame(proc_t,
                               columns=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
        tests_df = pd.DataFrame(
            tests_t, columns=['SUBJECT_ID', 'ITEMID', 'DATE', 'VALUE'])
        return static_df, adm_df, diag_df, proc_df, tests_df

    ZScoreScaler = Dict[str, Tuple[float, float]]

    @classmethod
    def make_zscore_concept_scaler(
            cls,
            test_df: pd.DataFrame,
            value_col: Optional[str] = 'VALUENUM') -> Subject.ZScoreScaler:
        """
        Z-score normalization for concepts with numerical values.

        Returns:
            A dictionary object of {test_id: (mean, std)}, which
            will be used to standardize values by (x-mean)/std for each
            test_id.
        """
        assert isinstance(
            test_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        assert test_df[value_col].notnull().all(
        ), "All values are expected to be not NaN"

        zscore_scaler: Subject.ZScoreScaler = {}
        for test_id, df in test_df.groupby('ITEMID'):
            vals = df[value_col].to_numpy()
            zscore_scaler[test_id] = (vals.mean(), vals.std())
        return zscore_scaler

    @classmethod
    def apply_zscore_concept_scaler(
            cls,
            test_df: pd.DataFrame,
            zscore_scaler: Subject.ZScoreScaler,
            value_col: Optional[str] = 'VALUENUM') -> pd.DataFrame:
        assert isinstance(
            test_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        new_test_df = test_df.copy(deep=True)
        mean, std = zip(*list(map(zscore_scaler.get, test_df.ITEMID)))
        mean = np.array(mean)
        std = np.array(std)
        vals = test_df[value_col].to_numpy()
        new_test_df[value_col] = (vals - mean) / std
        return new_test_df

    IQRFilter = Dict[str, Tuple[float, float]]

    @classmethod
    def make_iqr_concept_filter(
            cls,
            test_df: pd.DataFrame,
            value_col: Optional[str] = 'VALUENUM') -> Subject.IQRFilter:
        """
        Make outlier removal filter using Inter-quartile (IQR) method.
        https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/

        Args:
            test_df: DataFrame representation of tests.
        Returns:
            A dictionary object of {test_id: (min, max)}, which
            will be used to discard values outside (min, max) range for each
            ITEMID.
        """
        assert isinstance(
            test_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        assert test_df[value_col].notnull().all(
        ), "All values are expected to be not NaN"

        iqr_filter: Subject.IQRFilter = {}
        for test_id, df in test_df.groupby('ITEMID'):
            vals = df[value_col].to_numpy()
            upper_q = np.percentile(vals, 75)
            lower_q = np.percentile(vals, 25)
            iqr = (upper_q - lower_q) * 1.5
            iqr_filter[test_id] = (lower_q - iqr, upper_q + iqr)
        return iqr_filter

    @classmethod
    def apply_iqr_concept_filter(
            cls,
            test_df: pd.DataFrame,
            iqr_filter: Subject.IQRFilter,
            value_col: Optional[str] = 'VALUENUM') -> pd.DataFrame:
        """
        Apply outlier removal filter using Inter-quartile (IQR) method.

        Args:
            lab_df: DataFrame representation of patients lab tests.
            iqr_filter: dictionary object of the filter, generated by make_iqr_concept_filter function.
        """
        assert isinstance(
            test_df, pd.DataFrame
        ), "Only pandas.DataFrame is accepted in this function. You may need to convert List[Patient] into dataframe format using Patient.to_df class method."
        drop = []
        for i, (mn, mx) in iqr_filter.items():
            df = test_df[test_df['ITEMID'] == i]
            outliers = df[(df[value_col] > mx) | (df[value_col] < mn)]
            drop.extend(outliers.index)
        return test_df.drop(drop)


class SubjectPoint:
    def __init__(self, subject_id: int, days_ahead: int, age: float,
                 icd9_diag_codes: Set[str], icd9_proc_codes: Set[set],
                 tests: List[Test]):
        self.subject_id = subject_id
        self.days_ahead = days_ahead
        self.age = age
        self.icd9_diag_codes = icd9_diag_codes
        self.icd9_proc_codes = icd9_proc_codes
        self.tests = tests

    @classmethod
    def subject_to_points(cls, subject: Subject) -> Dict[int, SubjectPoint]:
        def _first_day_date(subject):
            first_admission_date = subject.admissions[0].admission_dates[0]
            first_test_date = subject.tests[0].date
            if Subject.days(first_test_date, first_admission_date) > 0:
                return first_admission_date
            else:
                return first_test_date

        first_day_date = _first_day_date(subject)

        points = {}
        for adm in subject.admissions:
            for i, adm_date in enumerate(pd.date_range(*adm.admission_dates)):
                days_ahead = Subject.days(adm_date, first_day_date)
                age = subject.age(adm_date)

                # Strong assumption: diagnosis only for first day in the
                # admission interval.
                if i == 0:
                    icd9_diag_codes = adm.icd9_diag_codes
                    icd9_proc_codes = set()
                else:
                    icd9_diag_codes = set()
                    icd9_proc_codes = adm.icd9_proc_codes

                points[days_ahead] = cls(subject_id=subject.subject_id,
                                         days_ahead=days_ahead,
                                         age=age,
                                         icd9_diag_codes=icd9_diag_codes,
                                         icd9_proc_codes=icd9_proc_codes,
                                         tests=[])

        for test in subject.tests:
            days_ahead = Subject.days(test.date, first_day_date)
            if days_ahead in points:
                points[days_ahead].tests.append(test)
            else:
                age = subject.age(test.date)
                points[days_ahead] = cls(subject_id=subject.subject_id,
                                         days_ahead=days_ahead,
                                         age=age,
                                         icd9_diag_codes=set(),
                                         icd9_proc_codes=set(),
                                         tests=[test])

        return list(sorted(points.values(), key=lambda p: p.days_ahead))
