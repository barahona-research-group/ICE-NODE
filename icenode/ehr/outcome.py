from typing import Set
import os

import numpy as np

from ..utils import load_config

from .coding_scheme import AbstractScheme, CodeMapper, code_scheme as C

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_OUTCOME_DIR = os.path.join(_RSC_DIR, 'outcome_filters')

outcome_conf_files = {
    'dx_flatccs_filter_v1': 'dx_flatccs_v1.json',
    'dx_icd9_filter_v1': 'dx_icd9_v1.json',
    'dx_icd9_filter_v2_groups': 'dx_icd9_v2_groups.json',
    'dx_icd9_filter_v3_groups': 'dx_icd9_v3_groups.json'
}


class OutcomeExtractor(AbstractScheme):

    def __init__(self, outcome_space='dx_flatccs_filter_v1'):
        conf = self.conf_from_json(outcome_conf_files[outcome_space])
        self._t_scheme = conf['code_scheme']
        codes = [
            c for c in sorted(self.t_scheme.index)
            if c not in conf['exclude_codes']
        ]

        index = dict(zip(codes, range(len(codes))))
        desc = {c: self.t_scheme.desc[c] for c in codes}
        super().__init__(codes=codes, index=index, desc=desc, name=conf)

    @property
    def t_scheme(self):
        return C[self._t_scheme]

    def map_codeset(self, codeset: Set[str], s_scheme: str):
        m = CodeMapper.get_mapper(s_scheme, self._t_scheme)
        codeset = m.map_codeset(codeset)

        if m.t_dag_space:
            codeset &= set(m.t_scheme.dag2code)
            codeset = set(m.t_scheme.dag2code[c] for c in codeset)

        return codeset & set(self.codes)

    def codeset2vec(self, codeset: Set[str], s_scheme: str):
        vec = np.zeros(len(self.index), dtype=bool)
        for c in self.map_codeset(codeset, s_scheme):
            vec[self.index[c]] = True
        return vec

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
            t_scheme = C[conf['code_scheme']]
            conf['exclude_codes'] = [
                c for c in t_scheme.codes if c not in conf['selected_codes']
            ]
            return conf
        elif 'exclude_codes' in conf:
            return conf
