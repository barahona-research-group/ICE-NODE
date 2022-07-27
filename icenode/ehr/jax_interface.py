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
from .coding_scheme import CodeMapper
from .outcome import DxCodeOutcomeFilter


class Admission_JAX:

    def __init__(self, adm: Admission, first_adm_date: datetime,
                 dx_mapper: CodeMapper, dx_outcome: DxCodeOutcomeFilter,
                 pr_mapper: Optional[CodeMapper]):
        # Time as days since the first admission
        self.admission_time = adm.admission_day(first_adm_date)
        self.los = adm.length_of_stay
        self.admission_id = adm.admission_id

        self.dx_codes = dx_mapper.map_codeset(adm.dx_codes)
        self.dx_vec = jnp.array(dx_mapper.codeset2vec(self.dx_codes))

        self.pr_codes = pr_mapper.map_codeset(adm.pr_codes)
        self.pr_vec = jnp.array(pr_mapper.codeset2vec(self.pr_codes))

        self.dx_outcome = jnp.array(dx_outcome.codeset2vec(adm.dx_codes))


class WindowFeatures:

    def __init__(self, past_admissions: List[Admission_JAX]):
        self.dx_features = self.dx_jax(past_admissions)

    @staticmethod
    def dx_jax(past_admissions: List[Admission_JAX]):
        past_codes = jnp.vstack([adm.dx_vec for adm in past_admissions])
        return jnp.max(past_codes, axis=0)


class Subject_JAX(dict):
    """
    Class to prepare EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """

    def __init__(self, subjects: List[Subject], code_scheme: Dict[str, str]):

        # Filter subjects with admissions less than two.
        subjects = [s for s in subjects if len(s.admissions) > 1]

        self._dx_mapper = CodeMapper.get_mapper(subjects[0].dx_scheme,
                                                code_scheme['dx'])
        self._pr_mapper = CodeMapper.get_mapper(subjects[0].pr_scheme,
                                                code_scheme.get('pr', 'none'))

        self._dx_outcome = DxCodeOutcomeFilter(
            input_dx_scheme=subjects[0].dx_scheme, conf=code_scheme['dx_outcome'])

        self._subjects = {s.subject_id: s for s in subjects}
        self.update(
            self.jaxify_subject_admissions(subjects,
                                           dx_mapper=self.dx_mapper,
                                           dx_outcome=self.dx_outcome,
                                           pr_mapper=self.pr_mapper))

    @property
    def subjects(self):
        return self._subjects

    @property
    def dx_mapper(self):
        return self._dx_mapper

    @property
    def dx_source_scheme(self) -> str:
        return self.dx_mapper.s_scheme

    @property
    def dx_scheme(self) -> str:
        return self.dx_mapper.t_scheme

    @property
    def dx_index(self) -> Dict[str, int]:
        return self.dx_mapper.t_index

    def dx_make_ancestors_mat(self):
        return self.dx_mapper.t_scheme.make_ancestors_mat()

    @property
    def dx_dim(self):
        return len(self.dx_index)

    @property
    def pr_mapper(self):
        return self._pr_mapper

    @property
    def pr_source_scheme(self) -> str:
        return self.pr_mapper.s_scheme

    @property
    def pr_scheme(self) -> str:
        return self.pr_mapper.t_scheme

    @property
    def pr_index(self) -> Dict[str, int]:
        return self.pr_mapper.t_index

    def pr_make_ancestors_mat(self):
        return self.pr_mapper.t_scheme.make_ancestors_mat()

    @property
    def pr_dim(self):
        return len(self.pr_index)

    @property
    def dx_outcome(self):
        return self._dx_outcome

    @property
    def dx_outcome_index(self) -> Dict[str, int]:
        return self.dx_outcome.index

    @property
    def dx_outcome_desc(self) -> Dict[str, str]:
        return self.dx_outcome.desc

    @property
    def dx_outcome_dim(self):
        return len(self.dx_outcome.index)

    def random_splits(self,
                      split1: float,
                      split2: float,
                      random_seed: int = 42):
        rng = random.Random(random_seed)
        subject_ids = list(sorted(self._subjects.keys()))
        rng.shuffle(subject_ids)

        split1 = int(split1 * len(subject_ids))
        split2 = int(split2 * len(subject_ids))

        train_ids = subject_ids[:split1]
        valid_ids = subject_ids[split1:split2]
        test_ids = subject_ids[split2:]
        return train_ids, valid_ids, test_ids

    def dx_history(self, subject_id, absolute_dates=False):
        return self.subjects[subject_id].dx_history(self.dx_scheme.name,
                                                    absolute_dates)

    def dx_outcome_history(self, subject_id, absolute_dates=False):
        return self.subjects[subject_id].dx_outcome_history(
            self.dx_outcome, absolute_dates)

    def adm_times(self, subject_id):
        adms_info = self[subject_id]
        return [(adm.admission_time, adm.admission_time + adm.los)
                for adm in adms_info]

    def dx_code_frequency(self, subjects: List[int]):
        return Subject.dx_code_frequency([self._subjects[i] for i in subjects],
                                         self.dx_mapper.t_scheme)

    def dx_frequency_vec(self,
                         subjects: Optional[List[int]] = None,
                         dx_code_scheme=None):
        subjects = subjects or self.keys()
        freq_dict = self.dx_code_frequency(subjects)
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
                                subjects: List[int] = []):
        subjects = subjects or self._subjects
        return self._code_frequency_partitions(
            percentile_range, self.dx_code_frequency(subjects))

    def batch_nth_admission(self, batch: List[int]):
        nth_admission = defaultdict(dict)
        for subject_id in batch:
            adms = self[subject_id]
            for n, adm in enumerate(adms):
                nth_admission[n][subject_id] = adm
        return nth_admission

    @staticmethod
    def subject_to_admissions(subject: Subject,
                              **kwargs) -> Dict[int, Admission_JAX]:
        return [
            Admission_JAX(adm, first_adm_date=subject.first_adm_date, **kwargs)
            for adm in subject.admissions
        ]

    @staticmethod
    def jaxify_subject_admissions(subjects, **kwargs):
        return {
            subject.subject_id:
            Subject_JAX.subject_to_admissions(subject, **kwargs)
            for subject in subjects
        }

    @classmethod
    def from_dataset(cls, dataset: AbstractEHRDataset, *args, **kwargs):
        subjects = Subject.from_dataset(dataset)
        return cls(subjects, *args, **kwargs)


class WindowedInterface_JAX(Subject_JAX):

    def __init__(self, interface: Subject_JAX):
        self._dx_mapper = interface._dx_mapper
        self._pr_mapper = interface._pr_mapper
        self._dx_outcome = interface._dx_outcome
        self._subjects = interface._subjects
        self.update(interface)
        self.features = self._compute_window_features(interface)

    @staticmethod
    def _compute_window_features(interface: Subject_JAX):
        features = {}
        for subj_id, adms in interface.items():
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
        batch = batch or sorted(self.features.keys())
        X = []
        y = []
        for subj_id in batch:
            adms = self[subj_id]
            features = self.features[subj_id]
            for adm, feats in zip(adms[1:], features[1:]):
                X.append(feats.dx_features)
                y.append(adm.dx_outcome)

        return np.vstack(X), np.vstack(y)

    @property
    def n_features(self):
        return self.dx_dim

    @property
    def n_targets(self):
        return self.dx_outcome_dim
