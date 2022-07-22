from typing import List
import os

import numpy as np

from ..utils import load_config, LazyDict

from .coding_scheme import AbstractScheme, code_scheme
from .concept import Admission

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_OUTCOME_DIR = os.path.join(_RSC_DIR, 'outcome_filters')


class DxCodeOutcomeFilter:

    def __init__(self, outcome_scheme: str, exclude_codes: List[str],
                 **kwargs):
        self.outcome_scheme = outcome_scheme
        base_codes = code_scheme[outcome_scheme].codes
        # Aleady sorted as base_scheme.codes is sorted.
        self.codes = [c for c in base_codes if c not in exclude_codes]
        self.index = dict(zip(self.codes, range(len(self.codes))))

    def __call__(self, adm: Admission):
        codeset, _ = AbstractScheme.map_codeset(adm.dx_codes, adm.dx_scheme,
                                                self.outcome_scheme)
        if self.outcome_scheme.hierarchical():
            dag2code = self.outcome_scheme.dag2code
            codeset = {dag2code[c] for c in codeset if c in dag2code}

        vec = np.zeros(len(self.index))
        for c in codeset & self.codes:
            vec[self.index[c]] = 1
        return vec

    @staticmethod
    def from_json(json_file: str):
        json_file = os.path.join(_OUTCOME_DIR, json_file)
        kwargs = load_config(json_file)

        if 'exclude_branches' in kwargs:
            # TODO
            return None
        elif 'select_branches' in kwargs:
            # TODO
            return None
        elif 'exclude_codes' in kwargs:
            return DxCodeOutcomeFilter(**kwargs)


dx_outcome_filter = LazyDict({
    'dx_flatccs_v1':
    lambda: DxCodeOutcomeFilter.from_json('dx_flatccs_v1.json')
})
