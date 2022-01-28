from __future__ import annotations
import logging
import math
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from enum import Flag, auto

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import jax.numpy as jnp

from .mimic3.concept import (Subject, SubjectPoint, Test)
from .mimic3.dag import CCSDAG

jax_interface_logger = logging.getLogger("jax_interface")


class AbstractSubjectJAXInterface:
    def __init__(self, subjects: List[Subject], test_id_set: Set[int],
                 dag: CCSDAG):

        self.subjects = dict(
            zip(map(lambda s: s.subject_id, subjects), subjects))
        self.dag: CCSDAG = dag
        self.test_idx = dict(zip(test_id_set, range(len(test_id_set))))

        self.diag_ccs_idx = dict(
            zip(dag.diag_ccs_codes, range(len(dag.diag_ccs_codes))))
        self.diag_flatccs_idx = dict(
            zip(dag.diag_flatccs_codes, range(len(dag.diag_flatccs_codes))))

        self.proc_ccs_idx = dict(
            zip(dag.proc_ccs_codes, range(len(dag.proc_ccs_codes))))

        self.diag_ccs_ancestors_mat = self.make_ccs_ancestors_mat(
            self.diag_ccs_idx)
        self.proc_ccs_ancestors_mat = self.make_ccs_ancestors_mat(
            self.proc_ccs_idx)

    def make_ccs_ancestors_mat(self, code2index) -> jnp.ndarray:
        ancestors_mat = []

        for code in sorted(code2index.keys()):
            ancestors_npvec = np.zeros(len(code2index), dtype=bool)
            ancestors = [
                a for a in self.dag.get_ccs_parents(code) if a in code2index
            ]
            for ancestor_j in map(code2index.get, ancestors):
                ancestors_npvec[ancestor_j] = 1
            ancestors_mat.append(jnp.array(ancestors_npvec))
        return jnp.vstack(ancestors_mat)

    def diag_ccs_to_vec(self, diag_ccs_codes):
        if len(diag_ccs_codes) == 0:
            return None

        n_cols = len(self.diag_ccs_idx)
        mask = np.zeros(n_cols)
        for c in diag_ccs_codes:
            mask[self.diag_ccs_idx[c]] = 1
        return jnp.array(mask)

    def diag_flatccs_to_vec(self, diag_flatccs_codes):
        if len(diag_flatccs_codes) == 0:
            return None

        n_cols = len(self.diag_flatccs_idx)
        mask = np.zeros(n_cols)
        for c in diag_flatccs_codes:
            mask[self.diag_flatccs_idx[c]] = 1

        return jnp.array(mask)

    def proc_ccs_to_vec(self, proc_ccs_codes):
        if len(proc_ccs_codes) == 0:
            return None

        n_cols = len(self.proc_ccs_idx)
        mask = np.zeros(n_cols)
        for c in proc_ccs_codes:
            mask[self.proc_ccs_idx[c]] = 1
        return jnp.array(mask)

    def diag_flatccs_frequency(self, subjects: Optional[List[int]] = None):
        subjects = subjects or self.subjects.keys()
        counter = defaultdict(int)
        for subject_id in subjects:
            for adm in self.subjects[subject_id].admissions:
                ccs_codes = set(
                    map(self.dag.diag_icd2flatccs.get, adm.icd9_diag_codes))
                for code in ccs_codes:
                    counter[self.diag_flatccs_idx[code]] += 1
        return counter

    def diag_flatccs_frequency_vec(self, subjects: Optional[List[int]] = None):
        counts = self.diag_flatccs_frequency(subjects)
        n_cols = len(self.diag_flatccs_idx)

        counts_vec = np.zeros(n_cols)
        for i, c in counts.items():
            counts_vec[i] = c
        return jnp.array(counts_vec)

    def diag_ccs_frequency(self, subjects: Optional[List[int]] = None):
        subjects = subjects or self.subjects.keys()
        counter = defaultdict(int)
        for subject_id in subjects:
            for adm in self.subjects[subject_id].admissions:
                ccs_codes = set(
                    map(self.dag.diag_icd2ccs.get, adm.icd9_diag_codes))
                for code in ccs_codes:
                    counter[self.diag_ccs_idx[code]] += 1
        return counter

    def diag_ccs_frequency_vec(self, subjects: Optional[List[int]] = None):
        counts = self.diag_ccs_frequency(subjects)
        n_cols = len(self.diag_ccs_idx)

        counts_vec = np.zeros(n_cols)
        for i, c in counts.items():
            counts_vec[i] = c
        return jnp.array(counts_vec)

    def diag_flatccs_by_percentiles(self,
                                    section_percentage: float = 20,
                                    subjects: Optional[List[int]] = None):
        n_sections = int(100 / section_percentage)
        sections = list(
            zip(range(0, 100, section_percentage),
                range(section_percentage, 101, section_percentage)))

        frequency = self.diag_flatccs_frequency(subjects)

        frequency_df = pd.DataFrame({
            'code': frequency.keys(),
            'frequency': frequency.values()
        })

        frequency_df = frequency_df.sort_values('frequency')
        frequency_df['cum_sum'] = frequency_df['frequency'].cumsum()
        frequency_df['cum_perc'] = 100 * frequency_df[
            'cum_sum'] / frequency_df["frequency"].sum()

        codes_by_percentiles = []
        for l, u in sections:
            codes = frequency_df[(frequency_df['cum_perc'] > l)
                                 & (frequency_df['cum_perc'] <= u)].code
            codes_by_percentiles.append(set(codes))

        return codes_by_percentiles

    def diag_ccs_by_percentiles(self,
                                    section_percentage: float = 20,
                                    subjects: Optional[List[int]] = None):
        n_sections = int(100 / section_percentage)
        sections = list(
            zip(range(0, 100, section_percentage),
                range(section_percentage, 101, section_percentage)))

        frequency = self.diag_ccs_frequency(subjects)

        frequency_df = pd.DataFrame({
            'code': frequency.keys(),
            'frequency': frequency.values()
        })

        frequency_df = frequency_df.sort_values('frequency')
        frequency_df['cum_sum'] = frequency_df['frequency'].cumsum()
        frequency_df['cum_perc'] = 100 * frequency_df[
            'cum_sum'] / frequency_df["frequency"].sum()

        codes_by_percentiles = []
        for l, u in sections:
            codes = frequency_df[(frequency_df['cum_perc'] > l)
                                 & (frequency_df['cum_perc'] <= u)].code
            codes_by_percentiles.append(set(codes))

        return codes_by_percentiles



class SubjectDiagSequenceJAXInterface(AbstractSubjectJAXInterface):
    def __init__(self, subjects: List[Subject], test_id_set: Set[int],
                 dag: CCSDAG):
        super().__init__(subjects, test_id_set, dag)
        self.diag_sequences = self.make_diag_sequences()

    def diag_ccs_to_vec(self, diag_ccs_codes):
        n_cols = len(self.diag_ccs_idx)
        mask = np.zeros(n_cols)
        for c in diag_ccs_codes:
            mask[self.diag_ccs_idx[c]] = 1
        return jnp.array(mask)

    def diag_flatccs_to_vec(self, diag_flatccs_codes):
        n_cols = len(self.diag_flatccs_idx)
        mask = np.zeros(n_cols)
        for c in diag_flatccs_codes:
            mask[self.diag_flatccs_idx[c]] = 1

        return jnp.array(mask)

    def make_diag_sequences(self):
        _diag_sequences = {}
        for subject_id, subject in self.subjects.items():
            _diag_sequences[subject_id] = {
                'diag_ccs_vec': [],
                'diag_flatccs_vec': []
            }

            for adm in subject.admissions:
                codes = adm.icd9_diag_codes
                diag_flatccs_codes = set(
                    map(self.dag.diag_icd2flatccs.get, codes))
                _s = self.diag_flatccs_to_vec(diag_flatccs_codes)

                diag_ccs_codes = set(map(self.dag.diag_icd2ccs.get, codes))
                _m = self.diag_ccs_to_vec(diag_ccs_codes)

                _diag_sequences[subject_id]['diag_ccs_vec'].append(_m)
                _diag_sequences[subject_id]['diag_flatccs_vec'].append(_s)

        return _diag_sequences

    def diag_sequences_batch(self, subjects_batch):
        return {i: self.diag_sequences[i] for i in subjects_batch}


class SubjectJAXInterface(AbstractSubjectJAXInterface):
    def __init__(self, subjects: List[Subject], test_id_set: Set[int],
                 dag: CCSDAG):
        super().__init__(subjects, test_id_set, dag)

        self.static_features, self.static_idx = self.make_static2vec()
        self.nth_points = self.make_nth_points()
        self.n_support = sorted(list(self.nth_points.keys()))

    def make_static2vec(self):
        genders = list(sorted(set(map(lambda s: s.gender, self.subjects.values()))))
        ethnics = list(sorted(
            set(map(lambda s: s.ethnic_group, self.subjects.values()))))

        static_columns = genders + ethnics
        static_idx = dict(zip(static_columns, range(len(static_columns))))

        static_features = {}
        n_cols = len(static_idx)

        for subject_id, subject in self.subjects.items():
            subject_features = np.zeros(n_cols)
            ones_indices = map(static_idx.get,
                               [subject.gender, subject.ethnic_group])
            subject_features[list(ones_indices)] = 1
            static_features[subject_id] = jnp.array(subject_features)

        return static_features, static_idx

    def tests2vec(self, tests: List[Test]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if len(self.test_idx) == 0 or len(tests) == 0:
            return None

        n_cols = len(self.test_idx)
        vals = np.zeros(n_cols)
        mask = np.zeros(n_cols)
        for test in tests:
            idx = self.test_idx[test.item_id]
            vals[idx] = test.value
            mask[idx] = 1

        return jnp.array(vals), jnp.array(mask)

    def make_nth_points(self):
        nth_points = defaultdict(dict)

        for subject_id, subject in self.subjects.items():
            for n, point in enumerate(SubjectPoint.subject_to_points(subject)):
                jaxified = self.jaxify_subject_point(point)
                if jaxified:
                    nth_points[n][subject_id] = jaxified

        return nth_points

    def jaxify_subject_point(self, point):
        diag_flatccs_codes = set(
            map(self.dag.diag_icd2flatccs.get, point.icd9_diag_codes))

        diag_ccs_codes = set(
            map(self.dag.diag_icd2ccs.get, point.icd9_diag_codes))

        proc_ccs_codes = set(
            map(self.dag.proc_icd2ccs.get, point.icd9_proc_codes))

        jaxified = {
            'age': point.age,
            'days_ahead': point.days_ahead,
            'diag_ccs_vec': self.diag_ccs_to_vec(diag_ccs_codes),
            'diag_flatccs_vec': self.diag_flatccs_to_vec(diag_flatccs_codes),
            'proc_ccs_vec': self.proc_ccs_to_vec(proc_ccs_codes),
            'tests': self.tests2vec(point.tests)
        }
        # Check if none of the following information exist, then return None.
        check = ['diag_ccs_vec', 'proc_ccs_vec', 'tests']
        if any(jaxified[l] is not None for l in check):
            return jaxified
        else:
            return None

    def nth_points_batch(self, n: int, batch: List[int]):
        if n not in self.nth_points:
            return {}
        return {k: v for k, v in self.nth_points[n].items() if k in batch}

    def subject_static(self, subject_id):
        return self.static_features[subject_id]


def create_patient_interface(processed_mimic_tables_dir: str,
                             data_tag=None,
                             ignore_tests=False,
                             ignore_proc=False):
    static_df = pd.read_csv(f'{processed_mimic_tables_dir}/static_df.csv.gz')
    static_df['DOB'] = pd.to_datetime(
        static_df.DOB, infer_datetime_format=True).dt.normalize()

    if ignore_proc:
        proc_df = None
    else:
        proc_df = pd.read_csv(f'{processed_mimic_tables_dir}/proc_df.csv.gz',
                              dtype={'ICD9_CODE': str})
    if ignore_tests:
        tests_df = None
        test_items = set()
    else:
        tests_df = pd.read_csv(f'{processed_mimic_tables_dir}/test_df.csv.gz')
        tests_df['DATE'] = pd.to_datetime(
            tests_df.DATE, infer_datetime_format=True).dt.normalize()
        tests_items = set(tests_df['ITEMID'])

    adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
    # Cast columns of dates to datetime64
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df['ADMITTIME'], infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df['DISCHTIME'], infer_datetime_format=True).dt.normalize()
    diag_df = pd.read_csv(f'{processed_mimic_tables_dir}/diag_df.csv.gz',
                          dtype={'ICD9_CODE': str})

    patients = Subject.to_list(static_df=static_df,
                               adm_df=adm_df,
                               diag_df=diag_df,
                               proc_df=proc_df,
                               tests_df=tests_df)

    # CCS Knowledge Graph
    k_graph = CCSDAG()

    return SubjectJAXInterface(patients, test_items, k_graph)
