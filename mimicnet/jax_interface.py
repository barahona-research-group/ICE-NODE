from __future__ import annotations
import logging
import math
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import jax.numpy as jnp

from .concept import Subject, SubjectPoint, Test
from .dag import CCSDAG

jax_interface_logger = logging.getLogger("jax_interface")


class SubjectJAXInterface:
    def __init__(self, subjects: List[Subject], test_id_set: Set[int],
                 dag: CCSDAG):
        self.subjects = dict(
            zip(map(lambda s: s.subject_id, subjects), subjects))
        self.dag = dag
        self.static_features, self.static_idx = self.__static2vec()
        self.test_idx = dict(zip(test_id_set, range(len(test_id_set))))
        self.diag_idx = dict(
            zip(dag.diag_icd_codes, range(len(dag.diag_icd_codes))))
        self.proc_idx = dict(
            zip(dag.proc_icd_codes, range(len(dag.proc_icd_codes))))
        self.nth_points = self.__nth_points()

    def __static2vec(self):
        genders = list(set(map(lambda s: s.gender, self.subjects.values())))
        ethnics = list(
            set(map(lambda s: s.ethnic_group, self.subjects.values())))
        static_columns = genders + ethnics
        static_idx = dict(zip(static_columns, range(len(static_columns))))

        static_features = {}
        n_cols = len(static_idx)

        for subject_id, subject in self.subjects.items():
            subject_features = np.zeros(n_cols)
            ones_indices = map(static_idx.get,
                               [subject.gender, subject.ethnic_group])
            subject_features[list(ones_indices)] = 1
            static_features[subject_id] = jnp.array(subject_features)

        return static_features, static_idx

    def __tests2vec(self,
                    tests: List[Test]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if len(tests) == 0:
            return None

        n_cols = len(self.test_idx)
        vals = np.zeros(n_cols)
        mask = np.zeros(n_cols)
        for test in tests:
            idx = self.test_idx[test.item_id]
            vals[idx] = test.value
            mask[idx] = 1

        return jnp.array(vals), jnp.array(mask)

    def __icd9_diag_to_vec(self, icd9_diag_codes):
        if len(icd9_diag_codes) == 0:
            return None

        n_cols = len(self.diag_idx)
        mask = np.zeros(n_cols)
        for c in icd9_diag_codes:
            mask[self.diag_idx[c]] = 1
        return jnp.array(mask)

    def __icd9_proc_to_vec(self, icd9_proc_codes):
        if len(icd9_proc_codes) == 0:
            return None

        n_cols = len(self.proc_idx)
        mask = np.zeros(n_cols)
        for c in icd9_proc_codes:
            mask[self.proc_idx[c]] = 1
        return jnp.array(mask)

    def __nth_points(self):
        nth_points = defaultdict(dict)

        for subject_id, subject in self.subjects.items():
            for n, point in enumerate(SubjectPoint.subject_to_points(subject)):
                nth_points[n][subject_id] = self.__jaxify_subject_point(point)

        return nth_points

    def __jaxify_subject_point(self, point):
        return {
            'age': point.age,
            'days_ahead': point.days_ahead,
            'icd9_diag_codes': self.__icd9_diag_to_vec(point.icd9_diag_codes),
            'icd9_proc_codes': self.__icd9_proc_to_vec(point.icd9_proc_codes),
            'tests': self.__tests2vec(point.tests)
        }

    def nth_points_batch(self, n: int, batch: List[int]):
        return {k: v for k, v in self.nth_points[n].items() if k in batch}

    def subject_static(self, subject_id):
        return self.static_features[subject_id]
