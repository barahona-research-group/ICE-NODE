from __future__ import annotations
from datetime import date, datetime
import os
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Set, Callable, Optional, Union, Dict
from absl import logging
import numpy as np
import jax.numpy as jnp
import pandas as pd

import equinox as eqx
from .coding_scheme import AbstractScheme, Singleton, _RSC_DIR
from .concept import StaticInfo, AbstractAdmission
from .jax_interface import Admission_JAX


@dataclass
class InterventionGroupScheme:
    """
    InterventionGroup class encapsulates the similar interventions.
    """
    group: List[str]
    uom: str
    aggregation: str


@dataclass
class InterventionScheme:
    input_index: Dict[str, int]
    output_index: Dict[str, int]
    groups: Dict[str, InterventionGroupScheme]
    group_name: Dict[str, str]


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
class InpatientObservables:
    observed: List[jnp.ndarray]
    mask: List[jnp.ndarray]
    time: List[float]


class Aggregator(eqx.Module):
    partition: Tuple[int, int]

    def __call__(self, x):
        raise NotImplementedError


class WeightedSum(Aggregator):
    linear: eqx.Linear

    def __call__(self, x):
        return self.linear(x[self.partition[0]:self.partition[1]])


class Sum(Aggregator):
    def __call__(self, x):
        return jnp.sum(x[self.partition[0]:self.partition[1]])


class OR(Aggregator):
    def __call__(self, x):
        return jnp.any(x[self.partition[0]:self.partition[1]]) * 1.0


class AggregateRepresentation(eqx.Module):
    aggregators: OrderedDict[str, Aggregator]

    def __call__(self, inpatient_input: InpatientInput, t):
        inin = inpatient_input(t)
        return jnp.concatenate(
            [agg(inin) for agg in self.aggregators.values()])


class EthMIMICIV(AbstractScheme, Singleton):
    _SCHEME_FILE = 'mimic4_eth.csv'
    NAME = 'mimic4_eth'
    ETH_CODE_CNAME = 'eth'
    ETH_DESC_CNAME = 'eth_desc'

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, self._SCHEME_FILE)
        df = pd.read_csv(filepath, dtype=str)
        desc = dict()
        for eth_code, eth_df in df.groupby(self.ETH_CODE_CNAME):
            eth_set = set(eth_df[self.ETH_DESC_CNAME])
            assert len(eth_set) == 1, "Ethnicity description should be unique"
            (eth_desc, ) = eth_set
            desc[eth_code] = eth_desc

        codes = sorted(set(df[self.ETH_CODE_CNAME]))

        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name=self.NAME)


@dataclass
class DxDischargeCodes(AbstractAdmission):
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    dx_codes: Set[str]  # Set of diagnostic codes
    dx_scheme: AbstractScheme  # Coding scheme for diagnostic codes


@dataclass
class InpatientAdmission(AbstractAdmission):
    dx_discharge_codes: DxDischargeCodes


@dataclass
class Inpatient:
    subject_id: str
    static_info: StaticInfo
    admissions: List[InpatientAdmission]

    @classmethod
    def from_dataset(cls, dataset: "lib.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()



@dataclass
class Inpatient_JAX:
    """JAX storage and interface for admission information"""

    admission_time: int
    los: int
    admission_id: int
    admission_date: datetime
    dx_vec_previous: jnp.array
    dx_vec_discharge: jnp.array
    inpatient_input: InpatientInput
    inpatient_observables: InpatientObservables


@dataclass
class SubjectPredictedRisk:

    admission: Admission_JAX
    prediction: jnp.ndarray
    trajectory: Optional[jnp.ndarray] = None
    other: Optional[Dict[str, jnp.ndarray]] = None

    def __str__(self):
        return f"""
                adm_id: {self.admission.admission_id}\n
                prediction: {self.prediction}\n
                """

    def get_outcome(self):
        return self.admission.get_outcome()

    def get_mask(self):
        return self.admission.get_mask()


class BatchPredictedRisks(dict):
    """JAX storage and interface for batch of predicted risks"""
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
            trajectory: Optional[jnp.ndarray] = None,
            other: Optional[Dict[str, jnp.ndarray]] = None):

        if subject_id not in self:
            self[subject_id] = {}

        self[subject_id][admission.admission_id] = SubjectPredictedRisk(
            admission=admission,
            prediction=prediction,
            trajectory=trajectory,
            other=other)

    def get_subjects(self):
        return sorted(self.keys())

    def get_risks(self, subject_id):
        risks = self[subject_id]
        return list(map(risks.get, sorted(risks)))

    def subject_prediction_loss(self, subject_id, loss_f: LossFunction):
        outcome = []
        prediction = []
        for r in self[subject_id].values():
            outcome.append(r.admission.get_outcome())
            prediction.append(r.prediction)

        outcome = jnp.vstack(outcome)
        prediction = jnp.vstack(prediction)
        mask = jnp.ones_like(outcome)

        return loss_f(outcome, prediction, mask)

    def prediction_loss(self, loss_f: LossFunction):
        loss = [
            self.subject_prediction_loss(subject_id, loss_f)
            for subject_id in self.keys()
        ]
        return jnp.nanmean(jnp.array(loss))

    def equals(self, other: BatchPredictedRisks):
        for subj_i, s_preds in self.items():
            s_preds_other = other[subj_i]
            for a, a_oth in zip(s_preds.values(), s_preds_other.values()):
                if ((a.get_outcome() != a_oth.get_outcome()).any()
                        or (a.prediction != a_oth.prediction).any()):
                    return False
        return True


class Subject_JAX(dict):
    """
    JAX storage and interface for subject information.
    It prepares EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """
    def __init__(self,
                 subjects: List[Subject],
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

        # Filter subjects with admissions less than two.
        subjects = [s for s in subjects if len(s.admissions) > 1]
        self._subjects = {s.subject_id: s for s in subjects}
        self._static_info = {
            i: StaticInfo_JAX(subj.static_info, static_info_flags)
            for i, subj in self._subjects.items()
        }
        self._dx_scheme = code_scheme['dx']
        self._pr_scheme = code_scheme.get('pr', NullScheme())

        self._outcome_extractor = code_scheme['outcome']

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
    def outcome_extractor(self):
        return self._outcome_extractor

    @property
    def outcome_dim(self):
        return len(self.outcome_extractor.index)

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
