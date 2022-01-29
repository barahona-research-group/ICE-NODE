from __future__ import annotations
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .mimic3.concept import (DiagSubject, AdmissionInfo)
from .mimic3.dag import CCSDAG


class AbstractSubjectJAXInterface:
    def __init__(self, subjects: List[DiagSubject], dag: CCSDAG):

        self.subjects = dict(
            zip(map(lambda s: s.subject_id, subjects), subjects))
        self.dag: CCSDAG = dag

        self.diag_ccs_idx = dict(
            zip(dag.diag_ccs_codes, range(len(dag.diag_ccs_codes))))
        self.diag_flatccs_idx = dict(
            zip(dag.diag_flatccs_codes, range(len(dag.diag_flatccs_codes))))

        self.diag_ccs_ancestors_mat = self.make_ccs_ancestors_mat(
            self.diag_ccs_idx)

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
    def __init__(self, subjects: List[DiagSubject], dag: CCSDAG):
        super().__init__(subjects, dag)
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
                'diag_flatccs_vec': [],
                'admission_id': []
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
                _diag_sequences[subject_id]['admission_id'].append(
                    adm.admission_id)

        return _diag_sequences

    def diag_sequences_batch(self, subjects_batch):
        return {i: self.diag_sequences[i] for i in subjects_batch}


class DiagnosisJAXInterface(AbstractSubjectJAXInterface):
    def __init__(self, subjects: List[DiagSubject], dag: CCSDAG):
        super().__init__(subjects, dag)
        self.nth_admission = self.make_nth_admission()
        self.n_support = sorted(list(self.nth_admission.keys()))

    def make_nth_admission(self):
        nth_admission = defaultdict(dict)

        for subject_id, subject in self.subjects.items():
            admissions = AdmissionInfo.subject_to_admissions(subject)
            for n, adm in enumerate(admissions):
                nth_admission[n][subject_id] = self.jaxify_subject_admission(
                    adm)

        return nth_admission

    def jaxify_subject_admission(self, admission: AdmissionInfo):
        diag_flatccs_codes = set(
            map(self.dag.diag_icd2flatccs.get, admission.icd9_diag_codes))

        diag_ccs_codes = set(
            map(self.dag.diag_icd2ccs.get, admission.icd9_diag_codes))

        return {
            'time': admission.admission_time,
            'los': admission.los,
            'diag_ccs_vec': self.diag_ccs_to_vec(diag_ccs_codes),
            'diag_flatccs_vec': self.diag_flatccs_to_vec(diag_flatccs_codes),
            'admission_id': admission.admission_id
        }

    def nth_admission_batch(self, n: int, batch: List[int]):
        if n not in self.nth_admission:
            return {}
        return {k: v for k, v in self.nth_admission[n].items() if k in batch}


def create_patient_interface(processed_mimic_tables_dir: str, data_tag=None):
    adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
    # Cast columns of dates to datetime64
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df['ADMITTIME'], infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df['DISCHTIME'], infer_datetime_format=True).dt.normalize()
    diag_df = pd.read_csv(f'{processed_mimic_tables_dir}/diag_df.csv.gz',
                          dtype={'ICD9_CODE': str})

    patients = DiagSubject.to_list(adm_df=adm_df, diag_df=diag_df)

    # CCS Knowledge Graph
    k_graph = CCSDAG()

    return DiagnosisJAXInterface(patients, k_graph)
