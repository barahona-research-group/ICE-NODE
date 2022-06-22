"""JAX storage and interface for EHR predictive models"""

from __future__ import annotations
from collections import defaultdict
import random
from typing import List, Optional, Dict, Set

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .mimic.concept import DxSubject
from .ccs_dag import ccs_dag


class AdmissionInfo:

    def __init__(self, subject_id: int, admission_time: int, los: int,
                 admission_id: int, dx_icd9_codes: Set[str]):
        self.subject_id = subject_id
        # Time as days since the first admission
        self.admission_time = admission_time
        # Length of Stay
        self.los = los
        self.admission_id = admission_id
        self.dx_icd9_codes = dx_icd9_codes
        self.dx_ccs_codes = self.dx_ccs_jax(dx_icd9_codes)
        self.dx_flatccs_codes = self.dx_flatccs_jax(dx_icd9_codes)

    @classmethod
    def dx_ccs_jax(cls, dx_icd9_codes):
        dx_ccs_codes = set(map(ccs_dag.dx_icd2ccs.get, dx_icd9_codes))
        n_cols = len(ccs_dag.dx_ccs_idx)
        mask = np.zeros(n_cols)
        for c in dx_ccs_codes:
            mask[ccs_dag.dx_ccs_idx[c]] = 1
        return jnp.array(mask)

    @classmethod
    def dx_flatccs_jax(self, dx_icd9_codes):
        dx_flatccs_codes = set(map(ccs_dag.dx_icd2flatccs.get, dx_icd9_codes))
        n_cols = len(ccs_dag.dx_flatccs_idx)
        mask = np.zeros(n_cols)
        for c in dx_flatccs_codes - {None}:
            mask[ccs_dag.dx_flatccs_idx[c]] = 1

        return jnp.array(mask)

    @classmethod
    def subject_to_admissions(cls,
                              subject: DxSubject) -> Dict[int, AdmissionInfo]:
        first_day_date = subject.admissions[0].admission_dates[0]
        adms = []
        for adm in subject.admissions:
            # days since first admission
            time = DxSubject.days(adm.admission_dates[0],
                                  subject.admissions[0].admission_dates[0])

            los = DxSubject.days(adm.admission_dates[1],
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
                              dx_icd9_codes=adm.dx_icd9_codes))
        return adms


class WindowFeatures:

    def __init__(self, past_admissions: List[AdmissionInfo]):
        self.dx_ccs_features = self.dx_ccs_jax(past_admissions)
        self.dx_flatccs_features = self.dx_flatccs_jax(past_admissions)

    @staticmethod
    def dx_ccs_jax(past_admissions: List[AdmissionInfo]):
        past_ccs_codes = jnp.vstack(
            [adm.dx_ccs_codes for adm in past_admissions])
        return jnp.max(past_ccs_codes, axis=0)

    @staticmethod
    def dx_flatccs_jax(past_admissions: List[AdmissionInfo]):
        past_flatccs_codes = jnp.vstack(
            [adm.dx_flatccs_codes for adm in past_admissions])
        return jnp.max(past_flatccs_codes, axis=0)


class DxInterface_JAX:
    """
    Class to prepare EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """

    def __init__(self, subjects: List[DxSubject]):
        self.dx_ccs_ancestors_mat = self.make_ccs_ancestors_mat(
            ccs_dag.dx_ccs_idx)

        # Filter subjects with admissions less than two.
        subjects = [s for s in subjects if len(s.admissions) > 1]
        self.subjects = self.jaxify_subject_admissions(subjects)

    def random_splits(self,
                      split1: float,
                      split2: float,
                      random_seed: int = 42):
        rng = random.Random(random_seed)
        subject_ids = list(sorted(self.subjects.keys()))
        rng.shuffle(subject_ids)

        split1 = int(split1 * len(subject_ids))
        split2 = int(split2 * len(subject_ids))

        train_ids = subject_ids[:split1]
        valid_ids = subject_ids[split1:split2]
        test_ids = subject_ids[split2:]
        return train_ids, valid_ids, test_ids

    def dx_ccs_history(self, subject_id):
        history = set()
        for adm in self.subjects[subject_id]:
            ccs_codes = set(map(ccs_dag.dx_icd2ccs.get, adm.dx_icd9_codes))
            history.update(ccs_codes)
        history = list(history)
        return history, list(map(ccs_dag.dx_ccs_idx.get, history))

    def dx_flatccs_history(self, subject_id):
        history = defaultdict(list)
        for adm in self.subjects[subject_id]:
            flatccs_codes = set(
                map(ccs_dag.dx_icd2flatccs.get, adm.dx_icd9_codes)) - {None}
            for code in flatccs_codes:
                history[code].append(
                    (adm.admission_time, adm.admission_time + adm.los))
        return history

    def adm_times(self, subject_id):
        adms_info = self.subjects[subject_id]
        return [(adm.admission_time, adm.admission_time + adm.los)
                for adm in adms_info]

    def make_ccs_ancestors_mat(self, code2index) -> jnp.ndarray:
        ancestors_mat = np.zeros((len(code2index), len(code2index)),
                                 dtype=bool)
        for code_i, i in code2index.items():
            for ancestor_j in ccs_dag.get_ccs_parents(code_i):
                j = code2index[ancestor_j]
                ancestors_mat[i, j] = 1

        return jnp.array(ancestors_mat)

    def dx_flatccs_frequency(self, subjects: Optional[List[int]] = None):
        subjects = subjects or self.subjects.keys()
        counter = defaultdict(int)
        for subject_id in subjects:
            for adm in self.subjects[subject_id]:
                ccs_codes = set(
                    map(ccs_dag.dx_icd2flatccs.get, adm.dx_icd9_codes))
                for code in ccs_codes - {None}:
                    counter[ccs_dag.dx_flatccs_idx[code]] += 1
        return counter

    def dx_flatccs_frequency_vec(self, subjects: Optional[List[int]] = None):
        counts = self.dx_flatccs_frequency(subjects)
        n_cols = len(ccs_dag.dx_flatccs_idx)

        counts_vec = np.zeros(n_cols)
        for i, c in counts.items():
            counts_vec[i] = c
        return jnp.array(counts_vec)

    def dx_ccs_frequency(self, subjects: Optional[List[int]] = None):
        subjects = subjects or self.subjects.keys()
        counter = defaultdict(int)
        for subject_id in subjects:
            for adm in self.subjects[subject_id]:
                ccs_codes = set(map(ccs_dag.dx_icd2ccs.get, adm.dx_icd9_codes))
                for code in ccs_codes:
                    counter[ccs_dag.dx_ccs_idx[code]] += 1
        return counter

    def dx_ccs_frequency_vec(self, subjects: Optional[List[int]] = None):
        counts = self.dx_ccs_frequency(subjects)
        n_cols = len(ccs_dag.dx_ccs_idx)

        counts_vec = np.zeros(n_cols)
        for i, c in counts.items():
            counts_vec[i] = c
        return jnp.array(counts_vec)

    def dx_flatccs_by_percentiles(self,
                                  section_percentage: float = 20,
                                  subjects: Optional[List[int]] = None):
        n_sections = int(100 / section_percentage)
        sections = list(
            zip(range(0, 100, section_percentage),
                range(section_percentage, 101, section_percentage)))

        frequency = self.dx_flatccs_frequency(subjects)

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

    def dx_ccs_by_percentiles(self,
                              section_percentage: float = 20,
                              subjects: Optional[List[int]] = None):
        n_sections = int(100 / section_percentage)
        sections = list(
            zip(range(0, 100, section_percentage),
                range(section_percentage, 101, section_percentage)))

        frequency = self.dx_ccs_frequency(subjects)

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

    def batch_nth_admission(self, batch: List[int]):
        nth_admission = defaultdict(dict)
        for subject_id in batch:
            adms = self.subjects[subject_id]
            for n, adm in enumerate(adms):
                nth_admission[n][subject_id] = adm
        return nth_admission

    def subject_admission_sequence(self, subject_id):
        return self.subjects[subject_id]

    def jaxify_subject_admissions(self, subjects):
        return {
            subject.subject_id: AdmissionInfo.subject_to_admissions(subject)
            for subject in subjects
        }


class DxWindowedInterface_JAX:

    def __init__(self, dx_interface: DxInterface_JAX):
        self.dx_interface = dx_interface
        self.dx_win_features = self._compute_window_features(dx_interface)

    @staticmethod
    def _compute_window_features(dx_interface: DxInterface_JAX):
        features = {}
        for subj_id, adms in dx_interface.subjects.items():
            current_window = []
            # Windowed features only contain information about the past adms.
            # First element, corresponding to first admission time, is None.
            window_features = [None]

            for adm in adms[:-1]:
                current_window.append(adm)
                window_features.append(WindowFeatures(current_window))
            features[subj_id] = window_features
        return features

    def tabular_features(self, batch: Optional[List[int]] = None):
        """
        Features are the past window of CCS codes, and labels
        are the past window of Flat CCS codes.
        """
        batch = batch or sorted(self.dx_win_features.keys())
        X = []
        y = []
        for subj_id in batch:
            adms = self.dx_interface.subjects[subj_id]
            features = self.dx_win_features[subj_id]
            for adm, feats in zip(adms[1:], features[1:]):
                X.append(feats.dx_ccs_features)
                y.append(adm.dx_flatccs_codes)

        return np.vstack(X), np.vstack(y)

    def n_features(self):
        return len(ccs_dag.dx_ccs_idx)

    def n_targets(self):
        return len(ccs_dag.dx_flatccs_idx)


def create_patient_interface(processed_mimic_tables_dir: str):
    adm_df = pd.read_csv(f'{processed_mimic_tables_dir}/adm_df.csv.gz')
    # Cast columns of dates to datetime64
    adm_df['ADMITTIME'] = pd.to_datetime(
        adm_df['ADMITTIME'], infer_datetime_format=True).dt.normalize()
    adm_df['DISCHTIME'] = pd.to_datetime(
        adm_df['DISCHTIME'], infer_datetime_format=True).dt.normalize()
    dx_df = pd.read_csv(f'{processed_mimic_tables_dir}/dx_df.csv.gz',
                        dtype={'ICD9_CODE': str})

    patients = DxSubject.to_list(adm_df=adm_df, dx_df=dx_df)

    return DxInterface_JAX(patients)
