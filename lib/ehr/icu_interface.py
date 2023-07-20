from __future__ import annotations
from datetime import date, datetime
import os
from collections import namedtuple, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Set, Callable, Optional, Union, Dict, ClassVar
import random
from absl import logging
import numpy as np
import jax.numpy as jnp
import pandas as pd

import equinox as eqx
from .dataset import MIMIC4ICUDataset
from .concept import AbstractAdmission, StaticInfoFlags
from .coding_scheme import (AbstractScheme, NullScheme,
                            AbstractGroupedProcedures)


class Inpatients:
    """
    JAX storage and interface for subject information.
    It prepares EHRs information to predictive models.
    NOTE: admissions with overlapping admission dates for the same patietn
    are merged. Hence, in case patients end up with one admission, they
    are discarded.
    """

    def __init__(self, dataset: MIMIC4ICUDataset):
        self._dataset = dataset
        self._subjects = dataset.to_subjects()

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

    def outcome_frequency_vec(self, subjects: List[int]):
        return sum(self._subjects[i].outcome_frequency_vec() for i in subjects)

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
