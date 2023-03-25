"""."""

from typing import Set, List, Callable, Dict, Tuple
from dataclasses import dataclass
import os
from collections import defaultdict
import jax
import jax.numpy as jnp
import numpy as np

from ..utils import load_config

from . import coding_scheme as C
from .concept import Admission, Subject

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_OUTCOME_DIR = os.path.join(_RSC_DIR, 'outcome_filters')

outcome_conf_files = {
    'dx_cprd_ltc212': 'dx_cprd_ltc212_v1.json',
    'dx_cprd_ltc9809': 'dx_cprd_ltc9809_v1.json',
    'dx_flatccs_mlhc_groups': 'dx_flatccs_mlhc_groups.json',
    'dx_flatccs_filter_v1': 'dx_flatccs_v1.json',
    'dx_icd9_filter_v1': 'dx_icd9_v1.json',
    'dx_icd9_filter_v2_groups': 'dx_icd9_v2_groups.json',
    'dx_icd9_filter_v3_groups': 'dx_icd9_v3_groups.json'
}

Outcome = jnp.ndarray
Mask = jnp.ndarray
SingleOutcome = Tuple[Outcome, Mask]
MaskedOutcome = Tuple[SingleOutcome, SingleOutcome]


class OutcomeExtractor(C.AbstractScheme):

    def __init__(self, outcome_space='dx_flatccs_filter_v1'):
        conf = self.conf_from_json(outcome_conf_files[outcome_space])

        self._t_scheme = eval(f"C.{conf['code_scheme']}")()
        codes = [
            c for c in sorted(self.t_scheme.index)
            if c not in conf['exclude_codes']
        ]

        index = dict(zip(codes, range(len(codes))))
        desc = {c: self.t_scheme.desc[c] for c in codes}
        super().__init__(codes=codes, index=index, desc=desc, name=conf)

    @property
    def t_scheme(self):
        return self._t_scheme

    @property
    def outcome_dim(self):
        return len(self.index)

    def map_codeset(self, codeset: Set[str], s_scheme: C.AbstractScheme):
        m = s_scheme.mapper_to(self._t_scheme)
        codeset = m.map_codeset(codeset)

        if m.t_dag_space:
            codeset &= set(m.t_scheme.dag2code)
            codeset = set(m.t_scheme.dag2code[c] for c in codeset)

        return codeset & set(self.codes)

    def codeset2vec(self, codeset: Set[str], s_scheme: C.AbstractScheme):
        vec = np.zeros(len(self.index), dtype=bool)
        for c in self.map_codeset(codeset, s_scheme):
            vec[self.index[c]] = True
        return jnp.array(vec)

    def subject_outcome(self, subject: Subject):
        mask = jnp.ones((self.outcome_dim, ))
        return [(self.codeset2vec(adm.dx_codes, adm.dx_scheme), mask)
                for adm in subject.admissions]

    def outcome_frequency_vec(self, subjects: List[Subject]):
        outcomes = [
            outcome[0] for subj in subjects
            for outcome in self.subject_outcome(subj)
        ]
        return sum(outcomes)

    def outcome_history(self,
                        admissions: List[Admission],
                        absolute_dates=False):
        history = defaultdict(list)
        first_adm_date = admissions[0].admission_dates[0]
        for adm in admissions:
            for code in self.map_codeset(adm.dx_codes, adm.dx_scheme):
                if absolute_dates:
                    history[code].append(adm.admission_dates)
                else:
                    history[code].append(adm.admission_day(first_adm_date),
                                         adm.discharge_day(first_adm_date))
        return history

    @staticmethod
    def conf_from_json(json_file: str):
        json_file = os.path.join(_OUTCOME_DIR, json_file)
        conf = load_config(json_file)

        if 'exclude_branches' in conf:
            # TODO
            return None
        elif 'select_branches' in conf:
            # TODO
            return None
        elif 'selected_codes' in conf:
            t_scheme = eval(f"C.{conf['code_scheme']}")()
            conf['exclude_codes'] = [
                c for c in t_scheme.codes if c not in conf['selected_codes']
            ]
            return conf
        elif 'exclude_codes' in conf:
            return conf


class SurvivalOutcomeExtractor(OutcomeExtractor):

    def subject_outcome(self, subject: Subject):
        # The mask elements are ones for each outcome coordinate until
        # the outcome appears at one admission, the mask will have zero values
        # for later admissions for that particular coordinate
        mask = jnp.ones((self.outcome_dim, ), dtype=bool)
        last_outcome = jnp.zeros((self.outcome_dim, ), dtype=bool)
        outcomes = []
        for adm in subject.admissions:
            outcome = self.codeset2vec(adm.dx_codes, adm.dx_scheme)
            outcome = outcome | last_outcome
            outcomes.append((outcome, mask))
            last_outcome = outcome
            # set mask[i] to zero if already zero or outcome[i] == 1
            mask = (mask & ~outcome)

        return outcomes

    def outcome_frequency_vec(self, subjects: List[Subject]):
        return sum(outcome[0] * outcome[1] for subj in subjects
                   for outcome in self.subject_outcome(subj))
