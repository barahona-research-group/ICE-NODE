import concurrent.futures
import logging
import math
from collections import defaultdict
from copy import deepcopy
from datetime import date
from itertools import chain
from random import shuffle
from typing import AbstractSet, Any, List, Dict, Mapping, Tuple, Optional, Sequence, Union, Callable, Set

import numpy as np
import pandas as pd

from .dag import CCSDAG
from .concept import Subject, HospitalAdmission
from .jax_interface import AbstractSubjectJAXInterface

glove_logger = logging.getLogger("glove")
"""
This implementation follows the following technical blog:
http://www.foldl.me/2014/glove-python/
"""

CooccurrencesType = Dict[Tuple[int, int], int]


def build_coocur(subjects: List[Subject],
                 code2idx: Mapping[str, int],
                 adm_ccs_codes: Callable[[HospitalAdmission], Set[str]],
                 ccs_dag: CCSDAG,
                 window_size_days: int = 1) -> Tuple[CooccurrencesType]:
    """
    Build cooccurrence matrix from timestamped CCS codes.

    Args:
        subjects: a list of subjects (patients) from which diagnosis codes are extracted for GloVe initialization.
        diag_idx: a mapping between CCS diagnosis codes as strings to integer indices.
        proc_idx: a mapping between CCS procedure codes as strings to integer indices.
        ccs_dag: CCSDAG object that supports DAG operations on ICD9 codes in their hierarchical organization within CCS.
        window_size_days: the moving time window of the context.
    """
    # Filter and augment all the codes, i.e. by adding the parent codes in the CCS hierarchy.
    # As described in the paper of GRAM, ancestors duplications are allowed and informative.
    subject_augmented_admissions = {}
    for subject in subjects:
        augmented_admissions = []
        for adm in subject.admissions:
            augmented_codes = []
            ccs_codes = adm_ccs_codes(adm)

            for c in ccs_codes:
                augmented_codes.append(c)
                parents = ccs_dag.get_ccs_parents(c)
                augmented_codes.extend(parents)

            augmented_admissions.append(
                (adm.admission_dates[0], augmented_codes))

        subject_augmented_admissions[subject.subject_id] = augmented_admissions

    cooccurrences: CooccurrencesType = defaultdict(int)
    for subject_id, admissions in subject_augmented_admissions.items():
        glove_logger.debug(f'admissions({len(admissions)}): {admissions}')
        for adm_date, adm in admissions:

            def is_context(other_adm):
                _adm_date, _adm = other_adm
                delta_days = Subject.days(adm_date, _adm_date)
                glove_logger.debug(f'delta_days: {delta_days}')
                # Symmetric context (left+right)
                return abs(delta_days) <= window_size_days

            context_admissions = filter(is_context, admissions)

            glove_logger.debug(f'{list(context_admissions)}')
            concepts = [c for a in context_admissions for c in a]

            concept_count: Dict[int, int] = defaultdict(int)

            for c in concepts:
                if c in code2idx:
                    concept_count[code2idx[c]] += 1
                else:
                    glove_logger.debug(f'CCS code {c} not found')

            for i, count_i in concept_count.items():
                for j, count_j in concept_count.items():
                    cooccurrences[(i, j)] += count_i * count_j
                    cooccurrences[(j, i)] += count_i * count_j
    return cooccurrences


def run_iter(data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is a pre-fetched data / weights list where each element is of
    the form
        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)
    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.
    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.
    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.
    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max)**alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context) + b_main[0] + b_context[0] -
                      math.log(cooccurrence))

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner**2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context /
                      np.sqrt(gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main**2
        gradsq_b_context += grad_bias_context**2

    return global_cost


def train_glove(code2idx: Mapping[str, int],
                cooccurrences: Mapping[Tuple[int, int], int],
                vector_size=100,
                iterations=25,
                **kwargs) -> Mapping[str, np.ndarray]:
    """
    Train GloVe vectors on the given `cooccurrences`, where
    each element is of the form ((word_i_id, word_j_id): x_ij)
    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.
    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.
    Keyword arguments are passed on to the iteration step function
    `run_iter`.
    Returns the computed word vector dictionary (code: vector).
    """

    size = len(code2idx)

    W = (np.random.rand(size * 2, vector_size) - 0.5) / float(vector_size + 1)
    biases = (np.random.rand(size * 2) - 0.5) / float(vector_size + 1)
    gradient_squared = np.ones((size * 2, vector_size), dtype=np.float64)
    gradient_squared_biases = np.ones(size * 2, dtype=np.float64)
    data = [(W[i_main], W[i_context + size], biases[i_main:i_main + 1],
             biases[i_context + size:i_context + size + 1],
             gradient_squared[i_main], gradient_squared[i_context + size],
             gradient_squared_biases[i_main:i_main + 1],
             gradient_squared_biases[i_context + size:i_context + size + 1],
             cooccurrence)
            for (i_main, i_context), cooccurrence in cooccurrences.items()]

    for i in range(iterations):
        glove_logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(data, **kwargs)

        glove_logger.info("\t\tDone (cost %f)", cost)

    code_vector = {}
    for code, idx in code2idx.items():
        code_vector[code] = W[idx, :]

    return code_vector


def glove_representation(category: str,
                         patient_interface: AbstractSubjectJAXInterface,
                         train_ids: List[Subject],
                         vector_size: int = 80,
                         iterations=25,
                         window_size_days=45,
                         **kwargs) -> Mapping[str, Mapping[str, np.ndarray]]:
    """
    Trains a GloVe vector representation for a set medical concepts in a given
    EHR data.
    Args:
        diag_idx: a mapping between CCS diagnosis codes as strings to integer indices.
        proc_idx: a mapping between CCS procedure codes as strings to integer indices.
        ccs_dag: CCSDAG object that supports DAG operations on ICD9 codes in their hierarchical organization within CCS.
        subjects: a list of subjects (patients) from which diagnosis codes are extracted for GloVe initialization.
        iter_callback: a callback to be invoked for every GloVe iteration.
        progress: a callback to be invoked after a set of codes finished
                processing.
        vector_size: vector size for each medical concept after training.
        iterations: maximum number of iterations.
        window_size_days: the moving time window of the context.
    Returns:
        Two dictionaries (tuple) for the GloVe vector representation for diagnoses codes and procedure codes, respectively.
    """

    if category == 'diag':
        code2idx = patient_interface.diag_multi_ccs_idx
        adm_ccs_codes = lambda adm: set(
            map(ccs_dag.diag_multi_icd2ccs.get, adm.icd9_diag_codes))

    elif category == 'proc':
        code2idx = patient_interface.proc_multi_ccs_idx
        adm_ccs_codes = lambda adm: set(
            map(ccs_dag.proc_multi_icd2ccs.get, adm.icd9_proc_codes))

    ccs_dag = patient_interface.dag
    glove_logger.setLevel(logging.WARNING)

    cooc = build_coocur(
        subjects=[patient_interface.subjects[i] for i in train_ids],
        code2idx=code2idx,
        adm_ccs_codes=adm_ccs_codes,
        ccs_dag=ccs_dag,
        window_size_days=window_size_days)
    return train_glove(code2idx=code2idx,
                       cooccurrences=cooc,
                       vector_size=vector_size,
                       iterations=iterations,
                       **kwargs)
