from __future__ import annotations
from datetime import date
from collections import namedtuple, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Set, Callable, Optional, Union, Dict, ClassVar
import numpy as np
import jax.numpy as jnp

import equinox as eqx
from .concept import AbstractAdmission
from .coding_scheme import (AbstractScheme, NullScheme,
                            AbstractGroupedProcedures)


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
            subset_index = sorted(source_scheme.index[c] for c in groups[g])
            agg_cls = agg_map[aggs[g]]
            self.aggregators[g] = agg_cls(jnp.array(subset_index))

    def __call__(self, inpatient_input: jnp.ndarray):
        return jnp.hstack(
            [agg(inpatient_input) for agg in self.aggregators.values()])


class InpatientInput(eqx.Module):
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

class InpatientSegmentedInput(eqx.Module):
    time_segments: List[Tuple[float, float]]
    input_segments: List[jnp.ndarray]

    @classmethod
    def from_input(cls, inpatient_input: InpatientInput,
                   agg: AggregateRepresentation, start_time: float,
                   end_time: float):
        ii = inpatient_input
        st = set(np.asarray(np.clip(ii.starttime, start_time, end_time)))
        et = set(np.asarray(np.clip(ii.endtime, start_time, end_time)))
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


class Codes(eqx.Module):
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    codes: Set[str]  # Set of diagnostic codes
    vec: jnp.ndarray
    scheme: AbstractScheme  # Coding scheme for diagnostic codes


class InpatientAdmission(AbstractAdmission, eqx.Module):
    dx_codes: Codes
    dx_codes_history: Codes
    outcome: Codes
    procedures: InpatientSegmentedInput
    inputs: InpatientInput
    observables: InpatientObservables


class StaticInfo(eqx.Module):
    gender: str
    date_of_birth: date
    ethnicity: jnp.ndarray
    ethnicity_scheme: AbstractScheme
    constant_vec: jnp.ndarray
    gender_dict: ClassVar[Dict[str, float]] = {'M': 1.0, 'F': 0.0}

    def __init__(self, gender: str, date_of_birth: date, ethnicity: str,
                 ethnicity_scheme: AbstractScheme):
        self.gender = gender
        self.date_of_birth = date_of_birth
        self.ethnicity = ethnicity
        self.ethnicity_scheme = ethnicity_scheme
        self.constant_vec = jnp.hstack(
            (self.ethnicity, self.gender_dict[self.gender]))

    def age(self, current_date: date):
        return (current_date - self.date_of_birth).days / 365.25

    def demographic_vector(self, current_date):
        return jnp.hstack((self.age(current_date), self.constant_vec))

class Inpatient(eqx.Module):
    subject_id: str
    static_info: StaticInfo
    admissions: List[InpatientAdmission]

    def outcome_frequency_vec(self):
        return sum(a.outcome.vec for a in self.admissions)

    @classmethod
    def from_dataset(cls, dataset: "lib.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()


