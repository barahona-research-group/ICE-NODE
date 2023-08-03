from __future__ import annotations
from datetime import date
from collections import namedtuple, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import (List, Tuple, Set, Callable, Optional, Union, Dict,
                    ClassVar, Union)
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
from .concept import AbstractAdmission
from .coding_scheme import (AbstractScheme, NullScheme,
                            AbstractGroupedProcedures)


class InpatientObservables(eqx.Module):
    time: jnp.narray
    value: jnp.ndarray
    mask: jnp.ndarray

    @staticmethod
    def empty(size: int):
        return InpatientObservables(time=np.zeros(0, dtype=np.float32),
                                    value=np.zeros((0, size),
                                                   dtype=np.float16),
                                    mask=np.zeros((0, size), dtype=bool))

    def segment(self, t_sep: jnp.ndarray):
        if len(t_sep) == 0:
            return [self]

        split = np.searchsorted(self.time, t_sep)
        time = np.split(self.time, split)
        value = np.vsplit(self.value, split)
        mask = np.vsplit(self.mask, split)

        return [
            InpatientObservables(t, v, m)
            for t, v, m in zip(time, value, mask)
        ]

    def __len__(self):
        return len(self.time)


class Aggregator(eqx.Module):
    subset: jnp.array

    def __call__(self, x):
        raise NotImplementedError


class WeightedSum(Aggregator):
    linear: eqx.Linear

    def __init__(self, subset: jnp.ndarray, key: "jax.random.PRNGKey"):
        super().__init__(subset)
        self.linear = eqx.nn.Linear(subset.shape[0],
                                    1,
                                    use_bias=False,
                                    key=key)

    @eqx.filter_jit
    def __call__(self, x):
        return self.linear(x[self.subset])


class Sum(Aggregator):

    def __call__(self, x):
        return x[self.subset].sum()


class OR(Aggregator):

    def __call__(self, x):
        return x[self.subset].any().astype(bool)


class AggregateRepresentation(eqx.Module):
    aggregators: OrderedDict[str, Aggregator]

    def __init__(self,
                 source_scheme: AbstractScheme,
                 target_scheme: AbstractGroupedProcedures,
                 key: "jax.random.PRNGKey" = None):
        super().__init__()
        groups = target_scheme.groups
        self.aggregators = OrderedDict()
        for g in sorted(groups.keys(), key=lambda g: target_scheme.index[g]):
            subset_index = sorted(source_scheme.index[c] for c in groups[g])
            if target_scheme.aggregation[g] == 'w_sum':
                self.aggregators[g] = WeightedSum(np.array(subset_index), key)
                (key, ) = jrandom.split(key, 1)
            elif target_scheme.aggregation[g] == 'sum':
                self.aggregators[g] = Sum(np.array(subset_index))
            elif target_scheme.aggregation[g] == 'or':
                self.aggregators[g] = OR(np.array(subset_index))
            else:
                raise ValueError(
                    f"Aggregation {target_scheme.aggregation[g]} not supported"
                )

    def np_agg(self, inpatient_input: np.ndarray):
        return np.hstack(
            [agg(inpatient_input) for agg in self.aggregators.values()])

    @eqx.filter_jit
    def jnp_agg(self, inpatient_input: jnp.ndarray):
        return jnp.hstack(
            [agg(inpatient_input) for agg in self.aggregators.values()])


maximum_padding = 50


class InpatientInput(eqx.Module):
    index: jnp.ndarray
    rate: jnp.ndarray
    starttime: jnp.ndarray
    endtime: jnp.ndarray
    size: int

    @staticmethod
    def pad_array(array: np.ndarray):
        n = len(array)
        n_pad = maximum_padding - (n % maximum_padding)
        if n_pad == maximum_padding:
            return array

        return np.pad(array,
                      pad_width=(0, n_pad),
                      mode='constant',
                      constant_values=0)

    def __init__(self, index: np.ndarray, rate: np.ndarray,
                 starttime: np.ndarray, endtime: np.ndarray, size: int):
        super().__init__()
        self.index = self.pad_array(index)
        self.rate = self.pad_array(rate)
        self.starttime = self.pad_array(starttime)
        self.endtime = self.pad_array(endtime)
        self.size = size

    def __call__(self, t):
        mask = (self.starttime <= t) & (t < self.endtime)
        index = self.index[mask]
        rate = self.rate[mask]

        if isinstance(rate, np.ndarray):
            adm_input = np.zeros(self.size, dtype=rate.dtype)
            adm_input[index] += rate
            return adm_input
        else:
            adm_input = jnp.zeros(self.size)
            return adm_input.at[index].add(rate)

    @classmethod
    def empty(cls, size: int):
        zvec = np.zeros(0, dtype=bool)
        return cls(zvec.astype(int), zvec, zvec, zvec, size)


class InpatientInterventions(eqx.Module):
    proc: Optional[InpatientInput]
    input_: Optional[InpatientInput]

    time: jnp.ndarray
    segmented_input: Optional[List[np.ndarray]]
    segmented_proc: Optional[List[np.ndarray]]

    def __init__(self, proc: InpatientInput, input_: InpatientInput,
                 adm_interval: float):
        super().__init__()
        self.proc = proc
        self.input_ = input_
        self.segmented_proc = None
        self.segmented_input = None

        time = [
            np.clip(t, 0.0, adm_interval)
            for t in (proc.starttime, proc.endtime, input_.starttime,
                      input_.endtime)
        ]
        self.time = np.unique(
            np.hstack(time + [0.0, adm_interval], dtype=np.float32))

    @property
    def t0(self):
        """Start times for segmenting the interventions"""
        return self.time[:-1]

    @property
    def t1(self):
        """End times for segmenting the interventions"""
        return self.time[1:]

    @property
    def t_sep(self):
        """Separation times for segmenting the interventions"""
        return self.time[1:-1]

    @property
    def interval(self):
        """Length of the admission interval"""
        return self.time[-1] - self.time[0]

    def segment_proc(self, proc_repr: AggregateRepresentation):
        proc_segments = [proc_repr.np_agg(self.proc(t)) for t in self.t0]
        update = eqx.tree_at(lambda x: x.segmented_proc,
                             self,
                             proc_segments,
                             is_leaf=lambda x: x is None)
        update = eqx.tree_at(lambda x: x.proc, update, None)
        return update

    def segment_input(self, input_repr: AggregateRepresentation):
        inp_segments = [input_repr.jnp_agg(self.input_(t)) for t in self.t0]
        update = eqx.tree_at(lambda x: x.segmented_input,
                             self,
                             inp_segments,
                             is_leaf=lambda x: x is None)
        update = eqx.tree_at(lambda x: x.input_, update, None)
        return update


class CodesVector(eqx.Module):
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    vec: jnp.ndarray
    scheme: AbstractScheme  # Coding scheme for diagnostic codes

    @classmethod
    def empty_like(cls, other: CodesVector):
        return cls(np.zeros_like(other.vec), other.scheme)

    @classmethod
    def empty(cls, scheme: AbstractScheme):
        return cls(np.zeros(len(scheme), dtype=bool), scheme)

    def to_codeset(self):
        index = self.vec.nonzero()[0]
        return set(self.scheme.index2code[i] for i in index)


class InpatientAdmission(AbstractAdmission, eqx.Module):
    admission_id: str
    admission_dates: Tuple[date, date]
    dx_codes: CodesVector
    dx_codes_history: CodesVector
    outcome: CodesVector
    observables: List[InpatientObservables]
    interventions: InpatientInterventions


@jax.jit
def demographic_vector(age, vec):
    return jnp.hstack((age, vec), dtype=jnp.float16)


class InpatientStaticInfo(eqx.Module):
    gender: jnp.ndarray
    date_of_birth: date
    ethnicity: jnp.ndarray
    ethnicity_scheme: AbstractScheme
    constant_vec: jnp.ndarray
    gender_dict: ClassVar[Dict[str, bool]] = {'M': bool(1), 'F': bool(0)}

    def __init__(self, gender: str, date_of_birth: date, ethnicity: str,
                 ethnicity_scheme: AbstractScheme):
        super().__init__()
        self.gender = np.array(self.gender_dict[gender])
        self.date_of_birth = date_of_birth
        self.ethnicity = ethnicity
        self.ethnicity_scheme = ethnicity_scheme
        self.constant_vec = np.hstack((self.ethnicity, self.gender))

    def age(self, current_date: date):
        return (current_date - self.date_of_birth).days / 365.25

    def demographic_vector(self, current_date: date):
        return demographic_vector(self.age(current_date), self.constant_vec)


class Inpatient(eqx.Module):
    subject_id: str
    static_info: InpatientStaticInfo
    admissions: List[InpatientAdmission]

    def outcome_frequency_vec(self):
        return sum(a.outcome.vec for a in self.admissions)

    @classmethod
    def from_dataset(cls, dataset: "lib.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()
