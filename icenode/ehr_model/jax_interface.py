"""JAX storage and interface for EHR predictive models"""

from __future__ import annotations
from collections import defaultdict
import random
from typing import List, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import jax.numpy as jnp

from .concept import Subject, Admission
from .dataset import AbstractEHRDataset
from .coding_scheme import (code_scheme, AbstractScheme)
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

        self.dx_scheme = dx_scheme
        self.dx_outcome_filter_label = dx_outcome_filter_label
        self.pr_scheme = pr_scheme

        _dx_scheme = code_scheme[dx_scheme]
        _pr_scheme = code_scheme[pr_scheme] if pr_scheme else None

        if _dx_scheme.hierarchical():
            self.dx_ancestors_mat = jnp.array(_dx_scheme.make_ancestors_mat())

        if _pr_scheme and _pr_scheme.hierarchical():
            self.pr_ancestors_mat = jnp.array(_pr_scheme.make_ancestors_mat())

        # Filter subjects with admissions less than two.
        subjects = [s for s in subjects if len(s.admissions) > 1]

        self.subjects = {s.subject_id: s for s in subjects}
        self.subjects_jax = self.jaxify_subject_admissions(
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

    def dx_history(self, subject_id, dx_scheme=None, absolute_dates=False):
        return self.subjects[subject_id].dx_history(dx_scheme, absolute_dates)

    def adm_times(self, subject_id):
        adms_info = self.subjects_jax[subject_id]
        return [(adm.admission_time, adm.admission_time + adm.los)
                for adm in adms_info]

    def dx_code_frequency(self,
                          subjects: List[int],
                          dx_code_scheme: Optional[str] = None):
        return Subject.dx_code_frequency([self.subjects[i] for i in subjects])

    def dx_frequency_vec(self,
                         subjects: Optional[List[int]] = None,
                         dx_code_scheme=None):
        subjects = subjects or self.subjects
        freq_dict = self.dx_code_frequency(subjects, dx_code_scheme)
        vec = np.zeros(len(freq_dict))
        for idx, count in freq_dict.items():
            vec[idx] = count
        return jnp.array(vec)

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

    def dx_codes_by_percentiles(self,
                                percentile_range: float = 20,
                                subjects: Optional[List[int]] = None,
                                dx_scheme: Optional[str] = None):
        return self._code_frequency_partitions(
            percentile_range, self.dx_code_frequency(subjects, dx_scheme))

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


class WindowedInterface_JAX:

    def __init__(self, interface: Subject_JAX):
        self.interface = interface
        self.win_features = self._compute_window_features(interface)

    @staticmethod
    def _compute_window_features(interface: Subject_JAX):
        features = {}
        for subj_id, adms in interface.subjects_jax.items():
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
        batch = batch or sorted(self.win_features.keys())
        X = []
        y = []
        for subj_id in batch:
            adms = self.interface.subjects_jax[subj_id]
            features = self.win_features[subj_id]
            for adm, feats in zip(adms[1:], features[1:]):
                X.append(feats.dx_codes)
                y.append(adm.dx_outcome)

        return np.vstack(X), np.vstack(y)

    def n_features(self):
        adm = list(self.interface.subjects_jax.values())[0][0]
        return len(adm.dx_codes)

    def n_targets(self):
        adm = list(self.interface.subjects_jax.values())[0][0]
        return len(adm.dx_outcome)

    @classmethod
    def from_dataset(cls, dataset: AbstractEHRDataset):
        subjects = Subject.from_dataset(dataset)
        return cls(subjects)
