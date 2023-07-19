from __future__ import annotations
from datetime import date, datetime
import os
from collections import namedtuple, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Set, Callable, Optional, Union, Dict
import random
from absl import logging
import numpy as np
import jax.numpy as jnp
import pandas as pd

import equinox as eqx
from .concept import StaticInfo, AbstractAdmission, StaticInfoFlags
from .coding_scheme import (AbstractScheme, NullScheme,
                            AbstractGroupedProcedures)
from .jax_interface import Admission_JAX


@dataclass
class InpatientObservables:
    time: jnp.narray
    value: jnp.ndarray
    mask: jnp.ndarray


class Aggregator(eqx.Module):
    subset: jnp.array

    def __call__(self, x):
        raise NotImplementedError


class WeightedSum(Aggregator):
    linear: eqx.Linear

    def __call__(self, x):
        return self.linear(x[self.subset])


class Sum(Aggregator):
    def __call__(self, x):
        return jnp.sum(x[self.subset])


class OR(Aggregator):
    def __call__(self, x):
        return jnp.any(x[self.subset]) * 1.0


class AggregateRepresentation(eqx.Module):
    aggregators: OrderedDict[str, Aggregator]

    def __init__(self, source_scheme: AbstractScheme,
                 target_scheme: AbstractGroupedProcedures):
        groups = target_scheme.groups
        aggs = target_scheme.aggregation
        agg_map = {'or': OR, 'sum': Sum, 'w_sum': WeightedSum}
        self.aggregators = OrderedDict()
        for g in sorted(groups.keys(), key=lambda g: target_scheme.index[g]):
            subset_index = sorted(target_scheme.index[c] for c in groups[g])
            agg_cls = agg_map[aggs[g]]
            self.aggregators[g] = agg_cls(jnp.array(subset_index))

    def __call__(self, inpatient_input: jnp.ndarray):
        return jnp.concatenate(
            [agg(inpatient_input) for agg in self.aggregators.values()])


@dataclass
class InpatientInput:
    index: jnp.ndarray
    rate: jnp.ndarray
    starttime: jnp.ndarray
    endtime: jnp.ndarray
    size: int

    def __call__(self, t):
        mask = (self.starttime <= t) & (t < self.endtime)
        index = self.index[mask]
        rate = self.rate[mask]
        adm_input = jnp.zeros(self.size)
        return adm_input.at[index].add(rate)


@dataclass
class InpatientSegmentedInput:
    time_segments: List[Tuple[float, float]]
    input_segments: List[jnp.ndarray]

    @classmethod
    def from_input(cls, inpatient_input: InpatientInput,
                   agg: AggregateRepresentation, start_time: float,
                   end_time: float):
        ii = inpatient_input
        st = set(np.clip(ii.starttime, start_time, end_time))
        et = set(np.clip(ii.endtime, start_time, end_time))
        jump_times = sorted(st | et | {start_time, end_time})
        time_segments = list(zip(jump_times[:-1], jump_times[1:]))
        input_segments = [agg(ii(t)) for t in jump_times[:-1]]
        return cls(time_segments, input_segments)

    def __call__(self, t):
        included = map(lambda s: s[0] <= t < s[1], self.time_segments)
        index = list(included).index(True)
        return self.input_segments[index]

    def concatenate(self, other: InpatientSegmentedInput):
        timestamps = sorted(
            set.union(*self.time_segments, *other.time_segments))
        time_segments = list(zip(timestamps[:-1], timestamps[1:]))
        input_segments = [
            jnp.hstack((self(t), other(t))) for t in timestamps[:-1]
        ]
        return InpatientSegmentedInput(time_segments, input_segments)


@dataclass
class DxDischargeCodes:
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    dx_codes: Set[str]  # Set of diagnostic codes
    dx_scheme: AbstractScheme  # Coding scheme for diagnostic codes

@dataclass
class InpatientAdmission(AbstractAdmission):
    dx_discharge_codes: DxDischargeCodes
    procedures: InpatientInput
    inputs: InpatientInput
    observables: InpatientObservables

@dataclass
class InpatientAdmissionInterface:
    inpatient_admission: InpatientAdmission
    segmented_procedures: InpatientSegmentedInput
    segmented_inputs: Optional[InpatientSegmentedInput]
    dx_history_vec: jnp.ndarray
    outcome_vec: jnp.ndarray


@dataclass
class InpatientPrediction:
    outcome_vec: jnp.ndarray
    state_trajectory: InpatientObservables
    observables: InpatientObservables


@dataclass
class Inpatient:
    subject_id: str
    static_info: StaticInfo
    admissions: List[InpatientAdmission]

    @classmethod
    def from_dataset(cls, dataset: "lib.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()


@dataclass
class InpatientPredictedRisk:

    admission: InpatientAdmission
    prediction: InpatientPrediction
    other: Optional[Dict[str, jnp.ndarray]] = None


class BatchPredictedRisks(dict):
    def add(self,
            subject_id: int,
            admission: InpatientAdmission,
            prediction: InpatientPrediction,
            other: Optional[Dict[str, jnp.ndarray]] = None):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][admission.admission_id] = InpatientPredictedRisk(
            admission=admission, prediction=prediction, other=other)

    def get_subjects(self):
        return sorted(self.keys())

    def get_predictions(self, subject_id):
        predictions = self[subject_id]
        return list(map(predictions.get, sorted(predictions)))

    def subject_prediction_loss(self, subject_id, dx_code_loss, obs_loss):
        dx_true, dx_pred, obs_true, obs_pred, obs_mask = [], [], [], [], []
        for r in self[subject_id].values():
            dx_true.append(r.admission.dx_discharge_codes.dx_codes)
            dx_pred.append(r.prediction.dx_discharge_codes.dx_codes)
            obs_true.append(r.admission.observables.value)
            obs_pred.append(r.prediction.observables.value)
            obs_mask.append(r.admission.observables.mask)

        dx_true = jnp.vstack(dx_true)
        dx_pred = jnp.vstack(dx_pred)
        obs_true = jnp.vstack(obs_true)
        obs_pred = jnp.vstack(obs_pred)
        obs_mask = jnp.vstack(obs_mask)

        return dx_code_loss(dx_true, dx_pred) + obs_loss(
            obs_true, obs_pred, obs_mask)

    def prediction_loss(self, dx_code_loss, obs_loss):
        loss = [
            self.subject_prediction_loss(subject_id, dx_code_loss, obs_loss)
            for subject_id in self.keys()
        ]
        return jnp.nanmean(jnp.array(loss))


class StaticInfo_JAX(eqx.Module):
    """JAX storage and interface for static information"""
    static_info: StaticInfo
    flags: StaticInfoFlags
    static_control_vec: jnp.ndarray

    def __init__(self, static_info, flags):
        super().__init__()
        self.static_info = static_info
        self.flags = flags
        self.static_control_vec = self._static_control_vector()

    def _static_control_vector(self):

        vec = []

        # Ethnicity
        if isinstance(self.static_info.ethnicity_scheme,
                      AbstractScheme) and self.flags.ethnicity:
            assert not isinstance(
                self.static_info.ethnicity_scheme, NullScheme
            ), "Ethnicity info requested while it is not provided in the dataset."
            m = self.static_info.ethnicity_scheme.mapper_to(
                self.flags.ethnicity)
            codeset = m.map_codeset({self.static_info.ethnicity})
            vec.append(m.codeset2vec(codeset))

        # Gender
        if self.flags.gender:
            assert self.static_info.gender is not None, "Gender info requested while it is not provided in the dataset."
            vec.append(np.array(self.static_info.gender, dtype=float))

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


class Subject_JAX(dict):
    """
    JAX storage and interface for subject information.
    It prepares EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """
    def __init__(self,
                 subjects: List[Inpatient],
                 code_scheme: Dict[str, AbstractScheme],
                 static_info_flags: StaticInfoFlags = StaticInfoFlags(),
                 data_max_size_gb=4):
        """
        Args:
            subjects: list of subjects.
            code_scheme: dictionary of code schemes.
            static_info_flags: flags for static information.
            data_max_size_gb: maximum size of data on the JAX device.
        """
        # If the data size on the JAX device will exceed this preset variable,
        # let the interface contain subject data amounting to that limit, and the
        # remaining subjects will be loaded to the device everytime one of these
        # subjects is requested.
        # This will rectify the memory consumption, but will lead to time
        # consumption for data loading between host/device memories.
        self._data_max_size_gb = data_max_size_gb

        self._subjects = {s.subject_id: s for s in subjects}
        self._static_info = {
            i: StaticInfo_JAX(subj.static_info, static_info_flags)
            for i, subj in self._subjects.items()
        }

        self.scheme = code_scheme

        self._jaxify_subject_admissions()

    def demographics(self, subject_id, current_date):
        static_info = self._static_info[subject_id]
        return static_info.control_vector(current_date)

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
        return self.subjects[subject_id].dx_history(self.scheme['dx'],
                                                    absolute_dates)

    def outcome_history(self, subject_id, absolute_dates=False):
        return self.subjects[subject_id].outcome_history(
            self.outcome_extractor, absolute_dates)

    def adm_times(self, subject_id):
        adms_info = self[subject_id]
        return [(adm.admission_time, adm.admission_time + adm.los)
                for adm in adms_info]

    def code_first_occurrence(self, subject_id):

        adms_info = self[subject_id]
        first_occurrence = np.empty_like(adms_info[0].get_outcome(), dtype=int)
        first_occurrence[:] = -1
        for adm in adms_info:
            update_mask = (first_occurrence < 0) & adm.get_outcome()
            first_occurrence[update_mask] = adm.admission_id
        return first_occurrence

    def outcome_frequency_vec(self, subjects: List[int]):
        subjs = list(map(self._subjects.get, subjects))
        return jnp.array(self.outcome_extractor.outcome_frequency_vec(subjs))

    def dx_batch_history_vec(self, subjects: List[Subject]):
        history = jnp.zeros((self.dx_dim, ), dtype=int)
        for adms in (self[i] for i in subjects):
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

    def outcome_by_percentiles(self,
                               percentile_range: float = 20,
                               subjects: Optional[List[int]] = None):

        subjects = subjects or list(self.keys())
        return self._code_frequency_partitions(
            percentile_range, self.outcome_frequency_vec(subjects))

    def batch_nth_admission(self, batch: List[int]):
        nth_admission = defaultdict(dict)
        for subject_id in batch:
            adms = self[subject_id]
            for n, adm in enumerate(adms):
                nth_admission[n][subject_id] = adm
        return nth_admission

    def dx_augmented_coocurrence(self,
                                 subjects: List[int],
                                 window_size_days: Optional[int] = None,
                                 context_size: Optional[int] = None):
        assert isinstance(
            self._dx_scheme, HierarchicalScheme
        ), "Augmented Coocurrence is only allowed for hierarchical coding schemes"
        return self._coocurrence(
            self._dx_scheme,
            adm_codes_f=lambda adm: adm.dx_codes,
            adm_mapper_f=lambda adm: adm.dx_scheme.mapper_to(self.dx_scheme),
            augmented=True,
            subjects=subjects,
            window_size_days=window_size_days,
            context_size=context_size)

    def pr_augmented_coocurrence(self,
                                 subjects: List[int],
                                 window_size_days: Optional[int] = None,
                                 context_size: Optional[int] = None):
        assert isinstance(
            self._pr_scheme, HierarchicalScheme
        ), "Augmented Coocurrence is only allowed for hierarchical coding schemes"

        return self._coocurrence(
            self._pr_scheme,
            adm_codes_f=lambda adm: adm.pr_codes,
            adm_mapper_f=lambda adm: adm.pr_scheme.mapper_to(self.pr_scheme),
            augmented=True,
            subjects=subjects,
            window_size_days=window_size_days,
            context_size=context_size)

    def dx_coocurrence(self,
                       subjects: List[int],
                       window_size_days: Optional[int] = None,
                       context_size: Optional[int] = None):
        return self._coocurrence(
            self._dx_scheme,
            adm_codes_f=lambda adm: adm.dx_codes,
            adm_mapper_f=lambda adm: adm.dx_scheme.mapper_to(self.dx_scheme),
            augmented=False,
            subjects=subjects,
            window_size_days=window_size_days,
            context_size=context_size)

    def pr_coocurrence(self,
                       subjects: List[int],
                       window_size_days: Optional[int] = None,
                       context_size: Optional[int] = None):
        return self._coocurrence(
            self._pr_scheme,
            adm_codes_f=lambda adm: adm.pr_codes,
            adm_mapper_f=lambda adm: adm.pr_scheme.mapper_to(self.pr_scheme),
            augmented=False,
            subjects=subjects,
            window_size_days=window_size_days,
            context_size=context_size)

    @staticmethod
    def _time_window_coocurrence(adms_list, window_size_days, index):
        for subj_adms_list in adms_list:
            for adm_day, _ in subj_adms_list:

                def is_context(other_adm):
                    # Symmetric context (left+right)
                    return abs(adm_day - other_adm[0]) <= window_size_days

                context_admissions = list(filter(is_context, subj_adms_list))
                codes = [c for _, _codes in context_admissions for c in _codes]

                code_count = defaultdict(int)
                for c in codes:
                    code_count[index[c]] += 1

                yield code_count

    @staticmethod
    def _seq_window_coocurrence(adms_list, context_size, index):
        for subj_adms_list in adms_list:
            sequence = [
                c for (_, codes) in subj_adms_list for c in sorted(codes)
            ]
            for i in range(len(sequence)):
                code_count = defaultdict(int)
                first_i = max(0, i - context_size)
                last_i = min(len(sequence) - 1, i + context_size)
                for c in sequence[first_i:last_i]:
                    code_count[index[c]] += 1
                yield code_count

    def _coocurrence(self, scheme: Union[AbstractScheme, HierarchicalScheme],
                     adm_codes_f: Callable[[Admission], Set[str]],
                     adm_mapper_f: Callable[[Admission],
                                            AbstractScheme], augmented: bool,
                     subjects: List[int], window_size_days: Optional[int],
                     context_size: Optional[int]):
        assert (window_size_days is None) != (
            context_size is
            None), 'Should pass either window_size_days or context_size'

        # Filter and augment all the codes, i.e. by adding the parent codes in the CCS hierarchy.
        # As described in the paper of GRAM, ancestors duplications are allowed and informative.

        def _augment_codes(codes):
            _aug_codes = []
            for c in codes:
                _aug_codes.extend(scheme.code_ancestors_bfs(c, True))
            return _aug_codes

        adms_list = []
        for subj_id in subjects:
            subject_adms = self._subjects[subj_id].admissions
            first_adm_date = subject_adms[0].admission_dates[0]

            subj_adms_list = []
            for adm in subject_adms:
                adm_day = adm.admission_day(first_adm_date)
                codes = adm_codes_f(adm)
                mapper = adm_mapper_f(adm)
                mapped_codes = mapper.map_codeset(codes)
                if augmented:
                    mapped_codes = _augment_codes(mapped_codes)
                subj_adms_list.append((adm_day, mapped_codes))
            adms_list.append(subj_adms_list)

        index = scheme.dag_index if augmented else scheme.index
        cooccurrences = defaultdict(int)

        def _add_counts(code_count):
            for i, count_i in code_count.items():
                for j, count_j in code_count.items():
                    cooccurrences[(i, j)] += count_i * count_j
                    cooccurrences[(j, i)] += count_i * count_j

        if window_size_days is not None:
            count_gen = self._time_window_coocurrence(adms_list,
                                                      window_size_days, index)
        else:
            count_gen = self._seq_window_coocurrence(adms_list, context_size,
                                                     index)

        for counter in count_gen:
            _add_counts(counter)

        coocurrence_mat = np.zeros((len(index), len(index)))
        for (i, j), count in cooccurrences.items():
            coocurrence_mat[i, j] = count
        return coocurrence_mat

    def _jaxify_subject_admissions(self):
        def _jaxify_adms(subj):
            outcomes = self.outcome_extractor.subject_outcome(subj)
            adms = []
            for adm, (outcome, mask) in zip(subj.admissions, outcomes):
                dx_mapper = adm.dx_scheme.mapper_to(self.dx_scheme)
                pr_mapper = adm.pr_scheme.mapper_to(self.pr_scheme)
                dx_codes = dx_mapper.map_codeset(adm.dx_codes)
                if self._dx_dagvec:
                    dx_vec = jnp.array(dx_mapper.codeset2dagvec(dx_codes))
                else:
                    dx_vec = jnp.array(dx_mapper.codeset2vec(dx_codes))

                pr_codes = pr_mapper.map_codeset(adm.pr_codes)
                if self._pr_dagvec:
                    pr_vec = jnp.array(pr_mapper.codeset2dagvec(pr_codes))
                else:
                    pr_vec = jnp.array(pr_mapper.codeset2vec(pr_codes))

                adms.append(
                    Admission_JAX(admission_time=adm.admission_day(
                        subj.first_adm_date),
                                  los=adm.length_of_stay,
                                  admission_id=adm.admission_id,
                                  admission_date=adm.admission_dates[0],
                                  dx_vec=dx_vec,
                                  pr_vec=pr_vec,
                                  outcome=(outcome, mask)))

            return adms

        def _lazy_load(subj):
            return lambda: _jaxify_adms(subj)

        acc_size_gb = 0.0
        n_subject_loaded = 0
        adm_data_size_gb = Subject_JAX.probe_admission_size_gb(
            dx_scheme=self.dx_scheme,
            dx_dagvec=self._dx_dagvec,
            pr_scheme=self.pr_scheme,
            pr_dagvec=self._pr_dagvec,
            outcome_extractor=self.outcome_extractor)

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

                pred = jrandom.normal(key, shape=adm.get_outcome().shape)
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
                                prediction=adm.get_outcome() * 1.0)
        return predictions

    def mean_predictions(self, train_split, test_split):
        predictions = BatchPredictedRisks()
        # Outcomes from training split
        outcomes = jnp.vstack(
            [a.get_outcome() for i in train_split for a in self[i]])
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
                                prediction=adms[i - 1].get_outcome() * 1.0)

        return predictions

    def historical_predictions(self, train_split, test_split):
        predictions = BatchPredictedRisks()

        # Aggregate all previous history for the particular subject.
        for subject_id in test_split:
            adms = self[subject_id]
            outcome = adms[0].get_outcome()
            for i in range(1, len(adms)):
                predictions.add(subject_id=subject_id,
                                admission=adms[i],
                                prediction=outcome * 1.0)
                outcome = jnp.maximum(outcome, adms[i - 1].get_outcome())

        return predictions

    def _compute_window_features(self, subjects: List[int]):
        features = {}
        for subj_id in subjects:
            adms = self[subj_id]
            # Windowed features only contain information about the past adms.
            dx_vec_list = []
            subject_features = []
            for i in range(len(adms) - 1):
                dx_vec_list.append(adms[i].dx_vec)
                current_features = jnp.max(jnp.vstack(dx_vec_list), axis=0)
                subject_features.append(current_features)
            features[subj_id] = subject_features
        return features

    def tabular_features(self, subjects: List[int]):

        features = self._compute_window_features(subjects)
        X = []
        for subj_id in subjects:
            X.extend(features[subj_id])

        return np.vstack(X)
