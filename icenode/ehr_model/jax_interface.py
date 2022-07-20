"""JAX storage and interface for EHR predictive models"""

from __future__ import annotations
from collections import defaultdict
import random
from typing import List, Optional, Dict, Set
from datetime import datetime

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .concept import Subject, Admission

from .coding_scheme import (code_scheme, code_scheme_cls, AbstractScheme,
                            HierarchicalScheme)
from .outcome import dx_outcome_filter, DxCodeOutcomeFilter


class Admission_JAX:

    def __init__(self, adm: Admission, first_adm_date: datetime,
                 dx_scheme: str, dx_outcome_filter_label: DxCodeOutcomeFilter,
                 pr_scheme: Optional[str]):
        # Time as days since the first admission
        self.admission_time = Subject.days(adm.admission_dates[0],
                                           first_adm_date)
        # Length of Stay
        # This 0.5 means if a patient is admitted and discharged at
        # the same day, then we assume 0.5 day as length of stay (12 hours)
        # In general, this would generalize the assumption to:
        # Admissions during a day happen at the midnight 00:01
        # While discharges during a day happen at the afternoon 12:00
        self.los = Subject.days(adm.admission_dates[1],
                                adm.admission_dates[0]) + 0.5
        self.admission_id = adm.admission_id
        self.dx_codes = self.jaxify_dx_codes(adm, dx_scheme)
        self.dx_outcome = self.jaxify_dx_outcome(adm, dx_outcome_filter_label)

        if pr_scheme and adm.pr_codes:
            self.pr_codes = self.jaxify_pr_codes(adm, pr_scheme)
        else:
            self.pr_codes = None

    @staticmethod
    def jaxify_dx_codes(adm, dx_scheme):
        dx_scheme = code_scheme[dx_scheme]
        return jnp.array(
            AbstractScheme.codeset2vec(adm.dx_codes, adm.dx_scheme, dx_scheme))

    @staticmethod
    def jaxify_dx_outcome(adm, dx_outcome_filter_label):
        dx_outcome = dx_outcome_filter[dx_outcome_filter_label]
        return jnp.array(dx_outcome(adm))

    @classmethod
    def jaxify_pr_codes(cls, adm, pr_scheme):
        pr_scheme = code_scheme[pr_scheme]
        return jnp.array(
            AbstractScheme.codeset2vec(adm.pr_codes, adm.pr_scheme, pr_scheme))

    @classmethod
    def subject_to_admissions(
            cls, subject: Subject, dx_scheme: str,
            dx_outcome_filter_label: str,
            pr_scheme: Optional[str]) -> Dict[int, Admission_JAX]:
        kwargs = dict(first_adm_date=subject.admissions[0].admission_dates[0],
                      dx_scheme=dx_scheme,
                      dx_outcome_filter_label=dx_outcome_filter_label,
                      pr_scheme=pr_scheme)
        return [Admission_JAX(adm, **kwargs) for adm in subject.admissions]


class WindowFeatures:

    def __init__(self, past_admissions: List[Admission_JAX]):
        self.dx_features = self.dx_jax(past_admissions)

    @staticmethod
    def dx_jax(past_admissions: List[Admission_JAX]):
        past_codes = jnp.vstack([adm.dx_codes for adm in past_admissions])
        return jnp.max(past_codes, axis=0)


class Subject_JAX:
    """
    Class to prepare EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """

    def __init__(self,
                 subjects: List[Subject],
                 dx_scheme: str,
                 dx_outcome_filter_label: str,
                 pr_scheme: Optional[str] = None):

        _dx_scheme = code_scheme[dx_scheme]
        _pr_scheme = code_scheme[pr_scheme] if pr_scheme else None

        if _dx_scheme.hierarchical():
            self.dx_ancestors_mat = jnp.array(_dx_scheme.make_ancestors_mat())

        if _pr_scheme and _pr_scheme.hierarchical():
            self.pr_ancestors_mat = jnp.array(_pr_scheme.make_ancestors_mat())

        # Filter subjects with admissions less than two.
        subjects = [s for s in subjects if len(s.admissions) > 1]
        self.subjects = self.jaxify_subject_admissions(
            subjects,
            dx_scheme=dx_scheme,
            dx_outcome_filter_label=dx_outcome_filter_label,
            pr_scheme=pr_scheme)

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

    def _dx_history(self, subject_id, code_index, transformer_dict):
        history = defaultdict(list)
        for adm in self.subjects[subject_id]:
            code_set = adm.dx_icd9_codes
            if transformer_dict:
                code_set = set(map(transformer_dict.get,
                                   adm.dx_icd9_codes)) - {None}
            for code in code_set:
                history[code].append(
                    (adm.admission_time, adm.admission_time + adm.los))
        return history

    def dx_ccs_history(self, subject_id):
        return self._dx_history(subject_id, ccs_dag.dx_ccs_idx,
                                ccs_dag.dx_icd2ccs)

    def dx_flatccs_history(self, subject_id):
        return self._dx_history(subject_id, ccs_dag.dx_flatccs_idx,
                                ccs_dag.dx_icd2flatccs)

    def adm_times(self, subject_id):
        adms_info = self.subjects[subject_id]
        return [(adm.admission_time, adm.admission_time + adm.los)
                for adm in adms_info]

    def make_ancestors_mat(self, code2index) -> jnp.ndarray:
        ancestors_mat = np.zeros((len(code2index), len(code2index)),
                                 dtype=bool)
        for code_i, i in code2index.items():
            for ancestor_j in ccs_dag.get_ccs_parents(code_i):
                j = code2index[ancestor_j]
                ancestors_mat[i, j] = 1

        return jnp.array(ancestors_mat)

    def _dx_code_frequency(self, subjects: List[int], code_index: Dict[str,
                                                                       int],
                           transformer_dict: Dict[str, str]):
        counter = defaultdict(int)
        for subject_id in subjects:
            for adm in self.subjects[subject_id]:
                code_set = adm.dx_icd9_codes
                if transformer_dict:
                    code_set = set(map(transformer_dict.get,
                                       code_set)) - {None}
                for code in code_set:
                    counter[code_index[code]] += 1

        # Return dictionary with zero-frequency codes added.
        return {idx: counter[idx] for idx in code_index.values()}

    def dx_flatccs_frequency(self, subjects: Optional[List[int]] = None):
        return self._dx_code_frequency(subjects or self.subjects.keys(),
                                       ccs_dag.dx_flatccs_idx,
                                       ccs_dag.dx_icd2flatccs)

    def dx_ccs_frequency(self, subjects: Optional[List[int]] = None):
        return self._dx_code_frequency(subjects or self.subjects.keys(),
                                       ccs_dag.dx_ccs_idx, ccs_dag.dx_icd2ccs)

    def _dx_frequency_vec(self, frequency_dict, code_index):
        vec = np.zeros(len(code_index))
        for idx, count in frequency_dict.items():
            vec[idx] = count
        return jnp.array(vec)

    def dx_flatccs_frequency_vec(self, subjects: Optional[List[int]] = None):
        return self._dx_frequency_vec(self.dx_flatccs_frequency(subjects),
                                      ccs_dag.dx_flatccs_idx)

    def dx_ccs_frequency_vec(self, subjects: Optional[List[int]] = None):
        return self._dx_frequency_vec(self.dx_ccs_frequency(subjects),
                                      ccs_dag.dx_ccs_idx)

    @staticmethod
    def _code_frequency_partitions(percentile_range, code_frequency):
        sections = list(range(0, 100, percentile_range)) + [100]
        sections[0] = -1

        frequency_df = pd.DataFrame({
            'code':
            sorted(code_frequency),
            'frequency':
            list(map(code_frequency.get, sorted(code_frequency)))
        })

        frequency_df = frequency_df.sort_values('frequency')
        frequency_df['cum_sum'] = frequency_df['frequency'].cumsum()
        frequency_df['cum_perc'] = 100 * frequency_df[
            'cum_sum'] / frequency_df["frequency"].sum()

        codes_by_percentiles = []
        for i in range(1, len(sections)):
            l, u = sections[i - 1], sections[i]
            codes = frequency_df[(frequency_df['cum_perc'] > l)
                                 & (frequency_df['cum_perc'] <= u)].code
            codes_by_percentiles.append(set(codes))

        return codes_by_percentiles

    def dx_flatccs_by_percentiles(self,
                                  percentile_range: float = 20,
                                  subjects: Optional[List[int]] = None):
        return self._code_frequency_partitions(
            percentile_range, self.dx_flatccs_frequency(subjects))

    def dx_ccs_by_percentiles(self,
                              percentile_range: float = 20,
                              subjects: Optional[List[int]] = None):
        return self._code_frequency_partitions(percentile_range,
                                               self.dx_ccs_frequency(subjects))

    def batch_nth_admission(self, batch: List[int]):
        nth_admission = defaultdict(dict)
        for subject_id in batch:
            adms = self.subjects[subject_id]
            for n, adm in enumerate(adms):
                nth_admission[n][subject_id] = adm
        return nth_admission

    def subject_admission_sequence(self, subject_id):
        return self.subjects[subject_id]

    def jaxify_subject_admissions(self, subjects, **kwargs):
        return {
            subject.subject_id:
            Admission_JAX.subject_to_admissions(subject, **kwargs)
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


def create_patient_interface(processed_mimic_tables_dir: str,
                             adm_df_name: str,
                             dx_df_name: str,
                             pr_df_name: str,
                             ):
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
