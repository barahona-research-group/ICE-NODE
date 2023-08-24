"""Data Model for Subjects in MIMIC-III and MIMIC-IV"""

from __future__ import annotations
from datetime import date
from typing import (List, Tuple, Optional, Union, Dict, ClassVar, Union, Any)
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
from .coding_scheme import (AbstractScheme, AbstractGroupedProcedures,
                            CodesVector)


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

    @staticmethod
    def concat(observables: List[InpatientObservables]):
        if len(observables) == 0:
            return InpatientObservables.empty(0)
        if isinstance(observables[0].time, jnp.ndarray):
            _np = jnp
        else:
            _np = np

        time = _np.hstack([o.time for o in observables])
        value = _np.vstack([o.value for o in observables])
        mask = _np.vstack([o.mask for o in observables])

        return InpatientObservables(time, value, mask)

    def __len__(self):
        return len(self.time)


class MaskedAggregator(eqx.Module):
    mask: jnp.array = eqx.static_field()

    def __init__(self,
                 subsets: List[np.ndarray],
                 input_size: int,
                 backend: str = 'jax'):
        super().__init__()
        mask = np.zeros((len(subsets), input_size), dtype=bool)
        for i, s in enumerate(subsets):
            mask[i, s] = True
        if backend == 'jax':
            self.mask = jnp.array(mask)
        else:
            self.mask = mask

    def __call__(self, x):
        raise NotImplementedError


class MaskedPerceptron(MaskedAggregator):
    linear: eqx.Linear

    def __init__(self,
                 subsets: jnp.ndarray,
                 input_size: int,
                 key: "jax.random.PRNGKey",
                 backend: str = 'jax'):
        super().__init__(subsets, input_size, backend)
        self.linear = eqx.nn.Linear(input_size,
                                    len(subsets),
                                    use_bias=False,
                                    key=key)

    @property
    def weight(self):
        return self.linear.weight

    @eqx.filter_jit
    def __call__(self, x):
        return (self.weight * self.mask) @ x


class MaskedSum(MaskedAggregator):

    def __call__(self, x):
        return self.mask @ x


class MaskedOr(MaskedAggregator):

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return np.any(self.mask & (x != 0), axis=1)
        else:
            return jnp.any(self.mask & (x != 0), axis=1)


class AggregateRepresentation(eqx.Module):
    aggregators: List[MaskedAggregator]
    splits: Tuple[int] = eqx.static_field()

    def __init__(self,
                 source_scheme: AbstractScheme,
                 target_scheme: AbstractGroupedProcedures,
                 key: "jax.random.PRNGKey" = None,
                 backend: str = 'numpy'):
        super().__init__()
        self.aggregators = []
        aggs = target_scheme.aggregation
        agg_grps = target_scheme.aggregation_groups
        grps = target_scheme.groups
        splits = []

        def is_contagious(x):
            return x.max() - x.min() == len(x) - 1 and len(set(x)) == len(x)

        for agg in aggs:
            selectors = []
            agg_offset = len(source_scheme)
            for grp in agg_grps[agg]:
                input_codes = grps[grp]
                input_index = sorted(source_scheme.index[c]
                                     for c in input_codes)
                input_index = np.array(input_index, dtype=np.int32)
                assert is_contagious(input_index), (
                    f"Selectors must be contiguous, but {input_index} is not. Codes: {input_codes}. Group: {grp}"
                )
                agg_offset = min(input_index.min(), agg_offset)
                selectors.append(input_index)
            selectors = [s - agg_offset for s in selectors]
            agg_input_size = sum(len(s) for s in selectors)
            max_index = max(s.max() for s in selectors)
            assert max_index == agg_input_size - 1, (
                f"Selectors must be contiguous, max index is {max_index} but size is {agg_input_size}"
            )
            splits.append(agg_input_size)

            if agg == 'w_sum':
                self.aggregators.append(
                    MaskedPerceptron(selectors, agg_input_size, key, backend))
                (key, ) = jrandom.split(key, 1)
            elif agg == 'sum':
                self.aggregators.append(
                    MaskedSum(selectors, agg_input_size, backend))
            elif agg == 'or':
                self.aggregators.append(
                    MaskedOr(selectors, agg_input_size, backend))
            else:
                raise ValueError(f"Aggregation {agg} not supported")
        splits = np.cumsum([0] + splits)[1:-1]
        self.splits = tuple(splits.tolist())

    @eqx.filter_jit
    def __call__(self, inpatient_input: Union[jnp.ndarray, np.ndarray]):
        if isinstance(inpatient_input, np.ndarray):
            splitted = np.hsplit(inpatient_input, self.splits)
            return np.hstack(
                [agg(x) for x, agg in zip(splitted, self.aggregators)])

        splitted = jnp.hsplit(inpatient_input, self.splits)
        return jnp.hstack(
            [agg(x) for x, agg in zip(splitted, self.aggregators)])


class InpatientInput(eqx.Module):
    index: jnp.ndarray
    rate: jnp.ndarray
    starttime: jnp.ndarray
    endtime: jnp.ndarray
    size: int

    def __init__(self, index: np.ndarray, rate: np.ndarray,
                 starttime: np.ndarray, endtime: np.ndarray, size: int):
        super().__init__()
        self.index = index
        self.rate = rate
        self.starttime = starttime
        self.endtime = endtime
        self.size = size

    def __call__(self, t):
        mask = (self.starttime <= t) & (t < self.endtime)
        if isinstance(self.index, np.ndarray):
            index = self.index[mask]
            rate = self.rate[mask]
            adm_input = np.zeros(self.size, dtype=rate.dtype)
            adm_input[index] += rate
            return adm_input
        else:
            index = jnp.where(mask, self.index, 0)
            rate = jnp.where(mask, self.rate, 0.0)
            adm_input = jnp.zeros(self.size, dtype=rate.dtype)
            return adm_input.at[index].add(rate)

    @classmethod
    def empty(cls, size: int):
        zvec = np.zeros(0, dtype=bool)
        return cls(zvec.astype(int), zvec, zvec, zvec, size)


class InpatientInterventions(eqx.Module):
    proc: Optional[InpatientInput]
    input_: Optional[InpatientInput]

    time: jnp.ndarray
    segmented_input: Optional[np.ndarray]
    segmented_proc: Optional[np.ndarray]

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
        time = np.unique(
            np.hstack(time + [0.0, adm_interval], dtype=np.float32))
        time = self.pad_array(time, value=np.nan)
        self.time = time

    @staticmethod
    def pad_array(array: np.ndarray,
                  maximum_padding: int = 100,
                  value: float = 0.0):
        """
        Pad array to be a multiple of maximum_padding. This is efficient to
        avoid jit-compiling a different function for each array shape.
        """

        n = len(array)
        n_pad = maximum_padding - (n % maximum_padding)
        if n_pad == maximum_padding:
            return array

        if isinstance(array, np.ndarray):
            _np = np
        else:
            _np = jnp

        return _np.pad(array,
                       pad_width=(0, n_pad),
                       mode='constant',
                       constant_values=value)

    @property
    def _np(self):
        if isinstance(self.time, np.ndarray):
            return np
        else:
            return jnp

    @property
    def t0_padded(self):
        return self.time[:-1]

    @property
    def t0(self):
        """Start times for segmenting the interventions"""
        t = self.time
        return t[~self._np.isnan(t)][:-1]

    @property
    def t1_padded(self):
        """End times for segmenting the interventions"""
        return self.time[1:]

    @property
    def t1(self):
        """End times for segmenting the interventions"""
        t = self.time
        return t[~self._np.isnan(t)][1:]

    @property
    def t_sep(self):
        """Separation times for segmenting the interventions"""
        t = self.time
        return t[~self._np.isnan(t)][1:-1]

    @property
    def interval(self):
        """Length of the admission interval"""
        return jnp.nanmax(self.time) - jnp.nanmin(self.time)


#     @eqx.filter_jit
#     def _jax_segment_proc(self, proc_repr: Optional[AggregateRepresentation]):
#         if proc_repr is None:
#             return eqx.filter_vmap(self.proc)(self.t0)

#         return eqx.filter_vmap(lambda t: proc_repr(self.proc(t)))(self.t0)

    def _np_segment_proc(self, proc_repr: Optional[AggregateRepresentation]):
        t = self.t0_padded[~np.isnan(self.t0_padded)]
        t_nan = self.t0_padded[np.isnan(self.t0_padded)]

        if proc_repr is None:
            out = np.vstack([self.proc(ti) for ti in t])
        else:
            out = np.vstack([proc_repr(self.proc(ti)) for ti in t])
        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])

    # @eqx.filter_jit
    # def _jax_segment_input(self,
    #                        input_repr: Optional[AggregateRepresentation]):
    #     if input_repr is None:
    #         return eqx.filter_vmap(self.input_)(self.t0)
    #     return eqx.filter_vmap(lambda t: input_repr(self.input_(t)))(self.t0)

    def _np_segment_input(self, input_repr: Optional[AggregateRepresentation]):
        t = self.t0_padded[~np.isnan(self.t0_padded)]
        t_nan = self.t0_padded[np.isnan(self.t0_padded)]

        if input_repr is None:
            out = np.vstack([self.input_(ti) for ti in t])
        else:
            out = np.vstack([input_repr(self.input_(ti)) for ti in t])

        pad = np.zeros((len(t_nan), out[0].shape[0]), dtype=out.dtype)
        return np.vstack([out, pad])

    def segment_proc(self,
                     proc_repr: Optional[AggregateRepresentation] = None):
        proc_segments = self._np_segment_proc(proc_repr)
        update = eqx.tree_at(lambda x: x.segmented_proc,
                             self,
                             proc_segments,
                             is_leaf=lambda x: x is None)
        update = eqx.tree_at(lambda x: x.proc, update, None)
        return update

    def segment_input(self,
                      input_repr: Optional[AggregateRepresentation] = None):
        inp_segments = self._np_segment_input(input_repr)
        update = eqx.tree_at(lambda x: x.segmented_input,
                             self,
                             inp_segments,
                             is_leaf=lambda x: x is None)
        update = eqx.tree_at(lambda x: x.input_, update, None)
        return update


class Admission(eqx.Module):
    admission_id: int  # Unique ID for each admission
    admission_dates: Tuple[date, date]
    dx_codes: CodesVector
    dx_codes_history: CodesVector
    outcome: CodesVector
    observables: Optional[Union[InpatientObservables,
                                List[InpatientObservables]]]
    interventions: Optional[InpatientInterventions]

    @property
    def interval_hours(self):
        return (self.admission_dates[1] -
                self.admission_dates[0]).total_seconds() / 3600

    @property
    def interval_days(self):
        return self.interval_hours / 24

    def days_since(self, date: date):
        d1 = (self.admission_dates[0] - date).total_seconds() / 3600 / 24
        d2 = (self.admission_dates[1] - date).total_seconds() / 3600 / 24
        return d1, d2


class DemographicVectorConfig(eqx.Module):
    gender: bool = False
    age: bool = False
    ethnicity: bool = False


class CPRDDemographicVectorConfig(DemographicVectorConfig):
    imd: bool = False


class StaticInfo(eqx.Module):
    demographic_vector_config: DemographicVectorConfig
    gender: Optional[CodesVector] = None
    ethnicity: Optional[CodesVector] = None
    date_of_birth: Optional[date] = None
    constant_vec: Optional[jnp.ndarray] = eqx.static_field(init=False)

    def __post_init__(self):
        attrs_vec = []
        if self.demographic_vector_config.gender:
            assert self.gender is not None and len(
                self.gender) > 0, "Gender is not extracted from the dataset"
            attrs_vec.append(self.gender.vec)
        if self.demographic_vector_config.ethnicity:
            assert self.ethnicity is not None, \
                "Ethnicity is not extracted from the dataset"
            attrs_vec.append(self.ethnicity.vec)

        if len(attrs_vec) == 0:
            self.constant_vec = np.array([], dtype=jnp.float16)
        else:
            self.constant_vec = np.hstack(attrs_vec)

    def age(self, current_date: date):
        return (current_date - self.date_of_birth).days / 365.25

    def demographic_vector(self, current_date: date):
        if self.demographic_vector_config.age:
            return self._concat(self.age(current_date), self.constant_vec)
        return self.constant_vec

    @staticmethod
    def _concat(age, vec):
        return jnp.hstack((age, vec), dtype=jnp.float16)

class CPRDStaticInfo(StaticInfo):
    imd: Optional[CodesVector] = None

    def __post_init__(self):
        attrs_vec = []
        if self.demographic_vector_config.gender:
            assert self.gender is not None and len(
                self.gender) > 0, "Gender is not extracted from the dataset"
            attrs_vec.append(self.gender.vec)
        if self.demographic_vector_config.ethnicity:
            assert self.ethnicity is not None, \
                "Ethnicity is not extracted from the dataset"
            attrs_vec.append(self.ethnicity.vec)

        if self.demographic_vector_config.imd:
            assert self.imd is not None, \
                "IMD is not extracted from the dataset"
            attrs_vec.append(self.imd.vec)

        if len(attrs_vec) == 0:
            self.constant_vec = np.array([], dtype=jnp.float16)
        else:
            self.constant_vec = np.hstack(attrs_vec)



class Patient(eqx.Module):
    subject_id: int
    static_info: StaticInfo
    admissions: List[Admission]

    @property
    def d2d_interval_days(self):
        d1 = self.admissions[0].admission_dates[1]
        d2 = self.admissions[-1].admission_dates[1]
        return (d2 - d1).total_seconds() / 3600 / 24

    def outcome_frequency_vec(self):
        return sum(a.outcome.vec for a in self.admissions)

    @classmethod
    def from_dataset(cls, dataset: "lib.ehr.dataset.AbstractEHRDataset"):
        return dataset.to_subjects()