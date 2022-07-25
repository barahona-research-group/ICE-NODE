from typing import List
import os

import numpy as np

from ..utils import load_config, LazyDict

from .coding_scheme import AbstractScheme, CodeMapper
from .concept import Admission

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_OUTCOME_DIR = os.path.join(_RSC_DIR, 'outcome_filters')

conf_files = {'dx_flatccs_v1': 'dx_flatccs_v1.json'}


class DxCodeOutcomeFilter:

    def __init__(self, s_dx_scheme: str, conf='dx_flatccs_v1'):
        conf = self.conf_from_json(conf_files[conf])
        self._mapper = CodeMapper.get_mapper(s_dx_scheme, conf['code_scheme'])

        base_codes = sorted(self.mapper.t_index)
        self._codes = [c for c in base_codes if c not in conf['exclude_codes']]
        self._index = dict(zip(self.codes, range(len(self.codes))))

    @property
    def mapper(self):
        return self._mapper

    @property
    def codes(self):
        return self._codes

    @property
    def index(self):
        return self._index

    def adm2vec(self, adm: Admission):
        codeset = self.mapper.map_codeset(adm.dx_codes)
        vec = np.zeros(len(self.index))
        for c in codeset & set(self.codes):
            vec[self.index[c]] = 1
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
        elif 'exclude_codes' in conf:
            return conf
