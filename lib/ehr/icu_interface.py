from __future__ import annotations
from datetime import date, datetime
from collections import namedtuple, OrderedDict, defaultdict
from typing import List, Tuple, Set, Callable, Optional, Union, Dict, ClassVar
import random
import logging
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import pandas as pd
import equinox as eqx

from .dataset import MIMIC4ICUDataset
from .icu_concepts import (InpatientAdmission, Inpatient, InpatientObservables)
from .coding_scheme import (AbstractScheme, AbstractGroupedProcedures)


class InpatientPrediction(eqx.Module):
    outcome_vec: jnp.ndarray
    state_trajectory: InpatientObservables
    observables: InpatientObservables


class InpatientPredictedRisk(eqx.Module):

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

    def subject_prediction_loss(self, subject_id, outcome_loss, obs_loss):
        outcome_true, outcome_pred, obs_true, obs_pred, obs_mask = [], [], [], [], []
        for r in self[subject_id].values():
            outcome_true.append(r.admission.outcome.vec)
            outcome_pred.append(r.prediction.outcome_vec)
            obs_true.append(r.admission.observables.value)
            obs_pred.append(r.prediction.observables.value)
            obs_mask.append(r.admission.observables.mask)

        outcome_true = jnp.vstack(outcome_true)
        outcome_pred = jnp.vstack(outcome_pred)
        obs_true = jnp.vstack(obs_true)
        obs_pred = jnp.vstack(obs_pred)
        obs_mask = jnp.vstack(obs_mask)

        return outcome_loss(outcome_true, outcome_pred) + obs_loss(
            obs_true, obs_pred, obs_mask)

    def prediction_loss(self, outcome_loss, obs_loss):
        loss = [
            self.subject_prediction_loss(subject_id, outcome_loss, obs_loss)
            for subject_id in self.keys()
        ]
        return jnp.nanmean(jnp.array(loss))


class Inpatients(eqx.Module):
    dataset: MIMIC4ICUDataset
    subjects: Dict[int, Inpatient]

    def __init__(self,
                 dataset: MIMIC4ICUDataset,
                 subject_ids: List[int],
                 max_workers: int = 1):
        self.dataset = dataset
        logging.debug('Loading subjects..')
        subjects_list = dataset.to_subjects(subject_ids,
                                            max_workers=max_workers)
        logging.debug('[DONE] Loading subjects')
        self.subjects = {s.subject_id: s for s in subjects_list}

    @property
    def size_in_bytes(self):
        is_arr = eqx.filter(self.subjects, eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects, is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def subject_size_in_bytes(self, subject_id):
        is_arr = eqx.filter(self.subjects[subject_id], eqx.is_array)
        arr_size = jtu.tree_map(
            lambda a, m: a.size * a.itemsize
            if m is not None else 0, self.subjects[subject_id], is_arr)
        return sum(jtu.tree_leaves(arr_size))

    def to_jax_arrays(self):
        arrs, others = eqx.partition(self.subjects, eqx.is_array)
        arrs = jtu.tree_map(lambda a: jnp.array(a), arrs)
        subjects = eqx.combine(arrs, others)
        return eqx.tree_at(lambda o: o.subjects, self, subjects)


    def outcome_frequency_vec(self, subjects: List[int]):
        return sum(self.subjects[i].outcome_frequency_vec() for i in subjects)

    def outcome_frequency_partitions(self, percentile_range,
                                     subjects: List[int]):
        frequency_vec = self.outcome_frequency_vec(subjects)

        sections = list(range(0, 100, percentile_range)) + [100]
        sections[0] = -1

        frequency_df = pd.DataFrame({
            'code': range(len(frequency_vec)),
            'frequency': frequency_vec
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


if __name__ == '__main__':
    pass
    # from .dataset import load_dataset
    # logging.root.level = logging.DEBUG
    # m4inpatient_dataset = load_dataset('M4ICU')
    # inpatients = Inpatients(m4inpatient_dataset)
