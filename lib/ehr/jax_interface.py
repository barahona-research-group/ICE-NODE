"""JAX storage and interface for EHR predictive models"""

from __future__ import annotations
from collections import defaultdict
import random
from typing import List, Optional, Dict, Any, Set, Callable, Union
from datetime import datetime
from dataclasses import dataclass
import os
from absl import logging
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx

from .concept import Subject, Admission, StaticInfo, StaticInfoFlags
from .dataset import AbstractEHRDataset
from .coding_scheme import AbstractScheme, NullScheme, HierarchicalScheme
from .outcome import OutcomeExtractor


class StaticInfo_JAX(eqx.Module):
    static_info: StaticInfo
    flags: StaticInfoFlags
    static_control_vec: jnp.ndarray

    def __init__(self, static_info, flags):
        self.static_info = static_info
        self.flags = flags
        self.static_control_vec = self._static_control_vector()

    def _static_control_vector(self):
        vec = []

        # Ethnicity
        if isinstance(
                self.static_info.ethnicity_scheme,
                AbstractScheme) and self.flags.ethnicity is not NullScheme():
            assert self.static_info.ethnicity_scheme is not NullScheme(
            ), "Ethnicity info requested while it is not provided in the dataset."
            m = self.static_info.ethnicity_scheme.mapper_to(
                self.flags.ethnicity)
            codeset = m.map_codeset({self.static_info.ethnicity})
            vec.append(m.codeset2vec(codeset))

        # Gender
        if self.flags.gender:
            assert self.static_info.gender is not None, "Gender info requested while it is not provided in the dataset."
            vec.append(np.array(self.static_info.gender, dtype=float))

        # IMD
        if self.flags.idx_deprivation:
            assert self.static_info.idx_deprivation is not None, "IMD info requested while it is not provided in the dataset."
            vec.append(np.array(self.static_info.idx_deprivation, dtype=float))

        if len(vec) > 0:
            return jnp.hstack(vec)
        else:
            return jnp.array([])

    def _dynamic_control_vector(self, current_date):
        vec = []
        if self.flags.age:
            assert self.static_info.date_of_birth is not None, "Age is requested while date of birth is not provided in the dataset."
            vec.append(
                np.array(self.static_info.age(current_date), dtype=float))

        if len(vec) > 0:
            return jnp.hstack(vec)
        else:
            return jnp.array([])

    def control_vector(self, current_date):
        d_vec = self._dynamic_control_vector(current_date)
        return jnp.hstack((d_vec, self.static_control_vec))


class Admission_JAX(eqx.Module):
    admission_time: int
    los: int
    admission_id: int
    admission_date: datetime
    dx_mapper: Any
    pr_mapper: Any
    dx_codes: Set[str]
    pr_codes: Set[str]
    dx_vec: jnp.ndarray
    pr_vec: jnp.ndarray
    dx_outcome: jnp.ndarray

    def __init__(self,
                 adm: Admission,
                 first_adm_date: datetime,
                 dx_scheme: str,
                 dx_outcome_extractor: OutcomeExtractor,
                 pr_scheme: str,
                 dx_dagvec=False,
                 pr_dagvec=False):
        # Time as days since the first admission
        self.admission_time = adm.admission_day(first_adm_date)
        self.los = adm.length_of_stay
        self.admission_id = adm.admission_id
        self.admission_date = adm.admission_dates[0]

        dx_mapper = adm.dx_scheme.mapper_to(dx_scheme)
        pr_mapper = adm.pr_scheme.mapper_to(pr_scheme)
        self.dx_mapper = dx_mapper
        self.pr_mapper = pr_mapper

        dx_codes = dx_mapper.map_codeset(adm.dx_codes)
        if dx_dagvec:
            self.dx_vec = jnp.array(dx_mapper.codeset2dagvec(dx_codes))
            self.dx_codes = dx_mapper.codeset2dagset(dx_codes)
        else:
            self.dx_vec = jnp.array(dx_mapper.codeset2vec(dx_codes))
            self.dx_codes = dx_codes

        pr_codes = pr_mapper.map_codeset(adm.pr_codes)
        if pr_dagvec:
            self.pr_vec = jnp.array(pr_mapper.codeset2dagvec(pr_codes))
            self.pr_codes = pr_mapper.codeset2dagset(pr_codes)
        else:
            self.pr_vec = jnp.array(pr_mapper.codeset2vec(pr_codes))
            self.pr_codes = pr_codes

        self.dx_outcome = jnp.array(
            dx_outcome_extractor.codeset2vec(adm.dx_codes, adm.dx_scheme))


@dataclass
class SubjectPredictedRisk:
    admission: Admission_JAX
    prediction: jnp.ndarray
    trajectory: Optional[jnp.ndarray] = None

    @property
    def ground_truth(self):
        return self.admission.dx_outcome

    def __str__(self):
        return f"""
                adm_id: {self.admission.admission_id}\n
                prediction: {self.prediction}\n
                """


class BatchPredictedRisks(dict):

    def __init__(self):
        self.embeddings = dict()

    def __str__(self):
        subjects_str = []
        for subj_id, _risks in self.items():
            subjects_str.extend([
                f'subject_id:{subj_id}\n{_risk}' for _risk in _risks.values()
            ])
        return '\n========\n'.join(subjects_str)

    def set_subject_embeddings(self, subject_id, embeddings):
        self.embeddings[subject_id] = embeddings

    def get_subject_embeddings(self, subject_id):
        return self.embeddings[subject_id]

    def add(self,
            subject_id: int,
            admission: Admission_JAX,
            prediction: jnp.ndarray,
            trajectory: Optional[jnp.ndarray] = None):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][admission.admission_id] = SubjectPredictedRisk(
            admission=admission, prediction=prediction, trajectory=trajectory)

    def get_subjects(self):
        return sorted(self.keys())

    def get_risks(self, subject_id):
        risks = self[subject_id]
        return list(map(risks.get, sorted(risks)))

    def subject_prediction_loss(self, subject_id, loss_f):
        loss = [
            loss_f(r.admission.dx_outcome, r.prediction)
            for r in self[subject_id].values()
        ]
        return sum(loss) / len(loss)

    def prediction_loss(self, loss_f):
        loss = [
            self.subject_prediction_loss(subject_id, loss_f)
            for subject_id in self.keys()
        ]
        return sum(loss) / len(loss)

    def equals(self, other: BatchPredictedRisks):
        for subj_i, s_preds in self.items():
            s_preds_other = other[subj_i]
            for a, a_oth in zip(s_preds.values(), s_preds_other.values()):
                if ((a.ground_truth != a_oth.ground_truth).any()
                        or (a.prediction != a_oth.prediction).any()):
                    return False
        return True


class Subject_JAX(dict):
    """
    Class to prepare EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """

    def __init__(self,
                 subjects: List[Subject],
                 code_scheme: Dict[str, AbstractScheme],
                 static_info_flags: StaticInfoFlags = StaticInfoFlags(),
                 data_max_size_gb=None):
        # If the data size on the JAX device will exceed this preset variable,
        # let the interface contain subject data amounting to that limit, and the
        # remaining subjects will be loaded to the device everytime one of these
        # subjects is requested.
        # This will rectify the memory consumption, but will lead to time
        # consumption for data loading between host/device memories.
        self._data_max_size_gb = data_max_size_gb or float(
            os.environ.get('ICENODE_INTERFACE_MAX_SIZE_GB', 1))

        # Filter subjects with admissions less than two.
        subjects = [s for s in subjects if len(s.admissions) > 1]
        self._subjects = {s.subject_id: s for s in subjects}
        self._static_info = {
            i: StaticInfo_JAX(subj.static_info, static_info_flags)
            for i, subj in self._subjects.items()
        }

        self._dx_scheme = code_scheme['dx']
        self._pr_scheme = code_scheme.get('pr', NullScheme())

        self._dx_outcome_extractor = code_scheme['dx_outcome']

        # The interface will map all Dx/Pr codes to DAG space if:
        # 1. The experiment explicitly requests so through (dx|pr)_dagvec
        # 2. At least one code mapper maps to DAG space. Note that there might
        # be multiple code mappers whenthe dataset have mixed coding schemes
        # for either Pr/Dx. For example, MIMIC-IV have mixed schemes of
        # ICD9/ICD10 for both Dx/Pr codes.
        self._dx_dagvec = code_scheme.get('dx_dagvec', False) or any(
            m.t_dag_space for m in self.dx_mappers)
        self._pr_dagvec = code_scheme.get('pr_dagvec', False) or any(
            m.t_dag_space for m in self.pr_mappers)

        self._jaxify_subject_admissions()

    @property
    def dx_dim(self):
        if self._dx_dagvec:
            return len(self.dx_scheme.dag_index)
        else:
            return len(self.dx_scheme.index)

    @property
    def pr_dim(self):
        if self._pr_dagvec:
            return len(self.pr_scheme.dag_index)
        else:
            return len(self.pr_scheme.index)

    @property
    def control_dim(self):
        ctrl = self.subject_control(
            list(self._static_info.keys())[0], datetime.today())
        if ctrl is None:
            return 0
        else:
            return len(ctrl)

    def subject_control(self, subject_id, current_date):
        static_info = self._static_info[subject_id]
        return static_info.control_vector(current_date)

    @staticmethod
    def probe_admission_size_gb(dx_scheme: AbstractScheme, dx_dagvec: bool,
                                pr_scheme: AbstractScheme, pr_dagvec: bool,
                                outcome_extractor: OutcomeExtractor):

        if dx_dagvec:
            dx_index = dx_scheme.dag_index
        else:
            dx_index = dx_scheme.index

        if pr_dagvec:
            pr_index = pr_scheme.dag_index
        else:
            pr_index = pr_scheme.index

        n_bytes = len(dx_index) + len(pr_index) + len(outcome_extractor.index)
        return n_bytes * 2**-30

    def __getitem__(self, k):
        v = super().__getitem__(k)
        # If value is callable, then
        if callable(v):
            return v()
        return v

    def get(self, k, default=None):
        if k in self:
            return self.__getitem__(k)
        return default

    @property
    def subjects(self):
        return self._subjects

    @property
    def dx_mappers(self):
        mappers = set()
        s_schemes = Subject.dx_schemes(self._subjects.values())
        for s in s_schemes:
            mappers.add(s.mapper_to(self.dx_scheme))
        return mappers

    @property
    def pr_mappers(self):
        mappers = set()
        s_schemes = Subject.pr_schemes(self._subjects.values())
        for s in s_schemes:
            mappers.add(s.mapper_to(self.pr_scheme))
        return mappers

    @property
    def dx_scheme(self) -> AbstractScheme:
        return self._dx_scheme

    @property
    def data_max_size_gb(self) -> float:
        return self._data_max_size_gb

    def dx_make_ancestors_mat(self):
        return self.dx_scheme.make_ancestors_mat()

    @property
    def pr_scheme(self) -> AbstractScheme:
        return self._pr_scheme

    def pr_make_ancestors_mat(self):
        return self.pr_scheme.make_ancestors_mat()

    @property
    def dx_outcome_extractor(self):
        return self._dx_outcome_extractor

    @property
    def dx_outcome_dim(self):
        return len(self.dx_outcome_extractor.index)

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
        return self.subjects[subject_id].dx_history(self.dx_scheme,
                                                    absolute_dates)

    def dx_outcome_history(self, subject_id, absolute_dates=False):
        return self.subjects[subject_id].dx_outcome_history(
            self.dx_outcome_extractor, absolute_dates)

    def adm_times(self, subject_id):
        adms_info = self[subject_id]
        return [(adm.admission_time, adm.admission_time + adm.los)
                for adm in adms_info]

    def code_first_occurrence(self, subject_id):

        adms_info = self[subject_id]
        first_occurrence = np.empty_like(adms_info[0].dx_outcome, dtype=int)
        first_occurrence[:] = -1
        for adm in adms_info:
            update_mask = (first_occurrence < 0) & adm.dx_outcome
            first_occurrence[update_mask] = adm.admission_id
        return first_occurrence

    def dx_outcome_frequency_vec(self, subjects: List[Subject]):
        np_res = Subject.dx_outcome_frequency_vec(
            map(self._subjects.get, subjects), self.dx_outcome_extractor)
        return jnp.array(np_res)

    def dx_batch_history_vec(self, subjects: List[Subject], skip_first_adm=True):
        history = jnp.zeros((self.dx_dim, ), dtype=int)
        for adms in (self[i] for i in subjects):
            if skip_first_adm:
                adms = adms[1:]
            history += sum(adm.dx_vec for adm in adms)
        return (history > 0).astype(int)

    @staticmethod
    def _code_frequency_partitions(percentile_range, code_frequency_vec):
        sections = list(range(0, 100, percentile_range)) + [100]
        sections[0] = -1

        frequency_df = pd.DataFrame({
            'code': range(len(code_frequency_vec)),
            'frequency': code_frequency_vec
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

    def dx_outcome_by_percentiles(self,
                                  percentile_range: float = 20,
                                  subjects: List[int] = []):
        return self._code_frequency_partitions(
            percentile_range, self.dx_outcome_frequency_vec(subjects))

    def batch_nth_admission(self, batch: List[int]):
        nth_admission = defaultdict(dict)
        for subject_id in batch:
            adms = self[subject_id]
            for n, adm in enumerate(adms):
                nth_admission[n][subject_id] = adm
        return nth_admission

    def dx_augmented_coocurrence(self,
                                 subjects: List[int],
                                 window_size_days: int = 1):
        assert isinstance(
            self._dx_scheme, HierarchicalScheme
        ), "Augmented Coocurrence is only allowed for hierarchical coding schemes"
        return self._coocurrence(self._dx_scheme,
                                 adm_codes_f=lambda adm: adm.dx_codes,
                                 augmented=True,
                                 subjects=subjects,
                                 window_size_days=window_size_days)

    def pr_augmented_coocurrence(self,
                                 subjects: List[int],
                                 window_size_days: int = 1):
        assert isinstance(
            self._pr_scheme, HierarchicalScheme
        ), "Augmented Coocurrence is only allowed for hierarchical coding schemes"
        return self._coocurrence(self._pr_scheme,
                                 adm_codes_f=lambda adm: adm.pr_codes,
                                 augmented=True,
                                 subjects=subjects,
                                 window_size_days=window_size_days)

    def dx_coocurrence(self, subjects: List[int], window_size_days: int = 1):
        return self._coocurrence(self._dx_scheme,
                                 adm_codes_f=lambda adm: adm.dx_codes,
                                 augmented=False,
                                 subjects=subjects,
                                 window_size_days=window_size_days)

    def pr_coocurrence(self, subjects: List[int], window_size_days: int = 1):
        return self._coocurrence(self._pr_scheme,
                                 adm_codes_f=lambda adm: adm.pr_codes,
                                 augmented=False,
                                 subjects=subjects,
                                 window_size_days=window_size_days)

    def _coocurrence(self,
                     scheme: Union[AbstractScheme, HierarchicalScheme],
                     adm_codes_f: Callable[[Admission_JAX], Set[str]],
                     augmented: bool,
                     subjects: List[int],
                     window_size_days: int = 1):
        """
        Build cooccurrence matrix from timestamped CCS codes.

        Args:
            code_type: the code type to consider for the coocurrences. 'dx' is for diagnostic codes, 'pr' for procedure codes.
            subjects: a list of subjects (patients) from which diagnosis codes are extracted for GloVe initialization.
            dx_idx: a mapping between CCS diagnosis codes as strings to integer indices.
            proc_idx: a mapping between CCS procedure codes as strings to integer indices.
            ccs_dag: CCSDAG object that supports DAG operations on ICD9 codes in their hierarchical organization within CCS.
            window_size_days: the moving time window of the context.
        """

        # Filter and augment all the codes, i.e. by adding the parent codes in the CCS hierarchy.
        # As described in the paper of GRAM, ancestors duplications are allowed and informative.

        def _augment_codes(codes):
            _aug_codes = []
            for c in codes:
                _aug_codes.extend(scheme.code_ancestors_bfs(c, True))
            return _aug_codes

        adms_lists = []
        for subj_id in subjects:
            subject_adms = self[subj_id]
            adms_list = []
            for adm in subject_adms:
                if augmented:
                    codes = _augment_codes(adm_codes_f(adm))
                else:
                    codes = adm_codes_f(adm)
                adms_list.append((adm.admission_time, codes))
            adms_lists.append(adms_list)

        index = scheme.dag_index if augmented else scheme.index
        cooccurrences = defaultdict(int)
        for subj_adms_list in adms_lists:
            for adm_day, adm_codes in subj_adms_list:

                def is_context(other_adm):
                    # Symmetric context (left+right)
                    return abs(adm_day - other_adm[0]) <= window_size_days

                context_admissions = list(filter(is_context, subj_adms_list))
                codes = [c for _, _codes in context_admissions for c in _codes]

                code_count = defaultdict(int)
                for c in codes:
                    code_count[index[c]] += 1

                for i, count_i in code_count.items():
                    for j, count_j in code_count.items():
                        cooccurrences[(i, j)] += count_i * count_j
                        cooccurrences[(j, i)] += count_i * count_j

        coocurrence_mat = np.zeros((len(index), len(index)))
        for (i, j), count in cooccurrences.items():
            coocurrence_mat[i, j] = count
        return coocurrence_mat

    def _jaxify_subject_admissions(self):

        def _jaxify_adms(subj):
            return [
                Admission_JAX(adm,
                              first_adm_date=subj.first_adm_date,
                              dx_scheme=self.dx_scheme,
                              dx_outcome_extractor=self.dx_outcome_extractor,
                              pr_scheme=self.pr_scheme,
                              dx_dagvec=self._dx_dagvec,
                              pr_dagvec=self._pr_dagvec)
                for adm in subj.admissions
            ]

        def _lazy_load(subj):
            return lambda: _jaxify_adms(subj)

        acc_size_gb = 0.0
        n_subject_loaded = 0
        adm_data_size_gb = Subject_JAX.probe_admission_size_gb(
            dx_scheme=self.dx_scheme,
            dx_dagvec=self._dx_dagvec,
            pr_scheme=self.pr_scheme,
            pr_dagvec=self._pr_dagvec,
            outcome_extractor=self.dx_outcome_extractor)

        for subject_id, subject in self.subjects.items():
            acc_size_gb += len(subject.admissions) * adm_data_size_gb
            if acc_size_gb < self.data_max_size_gb:
                self[subject_id] = _jaxify_adms(subject)
                n_subject_loaded += 1
            else:
                self[subject_id] = _lazy_load(subject)

        logging.info(
            f'Data of {n_subject_loaded}/{len(self.subjects)} subjects are loaded to device, rest are lazily loaded.'
        )

    @classmethod
    def from_dataset(cls, dataset: AbstractEHRDataset, *args, **kwargs):
        subjects = Subject.from_dataset(dataset)
        return cls(subjects, *args, **kwargs)

    def random_predictions(self, train_split, test_split, seed=0):
        predictions = BatchPredictedRisks()
        key = jrandom.PRNGKey(seed)

        for subject_id in test_split:
            # Skip first admission, not targeted for prediction.
            adms = self[subject_id][1:]
            for adm in adms:
                (key, ) = jrandom.split(key, 1)

                pred = jrandom.normal(key, shape=adm.dx_outcome.shape)
                predictions.add(subject_id=subject_id,
                                admission=adm,
                                prediction=pred)
        return predictions

    def cheating_predictions(self, train_split, test_split):
        predictions = BatchPredictedRisks()
        for subject_id in test_split:
            adms = self[subject_id][1:]
            for adm in adms:
                predictions.add(subject_id=subject_id,
                                admission=adm,
                                prediction=adm.dx_outcome * 1.0)
        return predictions

    def mean_predictions(self, train_split, test_split):
        predictions = BatchPredictedRisks()
        # Outcomes from training split
        outcomes = jnp.vstack(
            [a.dx_outcome for i in train_split for a in self[i]])
        outcome_mean = jnp.mean(outcomes, axis=0)

        # Train on mean outcomes
        for subject_id in test_split:
            adms = self[subject_id][1:]
            for adm in adms:
                predictions.add(subject_id=subject_id,
                                admission=adm,
                                prediction=outcome_mean)
        return predictions

    def recency_predictions(self, train_split, test_split):
        predictions = BatchPredictedRisks()

        # Use last admission outcome as it is
        for subject_id in test_split:
            adms = self[subject_id]
            for i in range(1, len(adms)):
                predictions.add(subject_id=subject_id,
                                admission=adms[i],
                                prediction=adms[i - 1].dx_outcome * 1.0)

        return predictions

    def historical_predictions(self, train_split, test_split):
        predictions = BatchPredictedRisks()

        # Aggregate all previous history for the particular subject.
        for subject_id in test_split:
            adms = self[subject_id]
            outcome = adms[0].dx_outcome
            for i in range(1, len(adms)):
                predictions.add(subject_id=subject_id,
                                admission=adms[i],
                                prediction=outcome * 1.0)
                outcome = jnp.maximum(outcome, adms[i - 1].dx_outcome)

        return predictions


class WindowFeatures:

    def __init__(self, past_admissions: List[Admission_JAX]):
        self.dx_features = self.dx_jax(past_admissions)

    @staticmethod
    def dx_jax(past_admissions: List[Admission_JAX]):
        past_codes = jnp.vstack([adm.dx_vec for adm in past_admissions])
        return jnp.max(past_codes, axis=0)


class WindowedInterface_JAX(Subject_JAX):

    def __init__(self, interface: Subject_JAX):
        self._interface_ = interface
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
        return self._interface_.dx_dim

    @property
    def n_targets(self):
        return self._interface_.dx_outcome_dim
