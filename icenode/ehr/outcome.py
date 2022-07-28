from typing import Set
import os

import numpy as np

from ..utils import load_config

from .coding_scheme import AbstractScheme, CodeMapper

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_OUTCOME_DIR = os.path.join(_RSC_DIR, 'outcome_filters')

conf_files = {'dx_flatccs_filter_v1': 'dx_flatccs_v1.json'}


class DxOutcome(AbstractScheme):

    def __init__(self, input_dx_scheme: str, conf='dx_flatccs_filter_v1'):
        conf = self.conf_from_json(conf_files[conf])
        self._mapper = CodeMapper.get_mapper(input_dx_scheme,
                                             conf['code_scheme'])

        assert self._mapper is not None, (
            f'None mapper: {input_dx_scheme}->{conf["code_scheme"]}')

        base_codes = sorted(self.mapper.t_index)

        codes = [c for c in base_codes if c not in conf['exclude_codes']]
        index = dict(zip(codes, range(len(codes))))
        desc = {c: self._mapper.t_desc[c] for c in codes}
        super().__init__(codes=codes, index=index, desc=desc, name=conf)

    @property
    def mapper(self):
        return self._mapper

    def map_codeset(self, codeset: Set[str]):
        codeset = self.mapper.map_codeset(codeset)
        return codeset & set(self.codes)

    def codeset2vec(self, codeset: Set[str]):
        vec = np.zeros(len(self.index))
        for c in self.map_codeset(codeset):
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
