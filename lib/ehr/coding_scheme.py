"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from __future__ import annotations

import logging
import os
import re
from abc import abstractmethod
from collections import defaultdict, OrderedDict
from threading import Lock
from typing import Set, Dict, Optional, List, Union, ClassVar, Callable, Tuple

import numpy as np
import pandas as pd

from ..base import Config, Module, Data
from ..utils import load_config

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, "resources")


class CodingSchemeConfig(Config):
    name: str


class CodingScheme(Module):
    config: CodingSchemeConfig
    # Possible Schemes, Lazy-loaded schemes.
    _load_schemes: ClassVar[Dict[str, Callable]] = {}
    _schemes: ClassVar[Dict[str, 'CodingScheme']] = {}

    def __init__(self, config: CodingSchemeConfig):
        super().__init__(config=config)
        # Register the scheme by name
        self._schemes[config.name] = self

        # Register the identity map
        CodeMap.register_map(self.name, self.name, IdentityCodeMap(self.name))

    @classmethod
    def from_name(cls, name):
        if name in cls._schemes:
            return cls._schemes[name]

        if name in cls._load_schemes:
            cls._load_schemes[name]()

        return cls._schemes[name]

    @classmethod
    def register_scheme(cls, scheme: CodingScheme):
        cls._schemes[scheme.name] = scheme

    @classmethod
    def register_scheme_loader(cls, name: str, loader: Callable):
        cls._load_schemes[name] = loader

    @property
    @abstractmethod
    def codes(self):
        pass

    @property
    @abstractmethod
    def index(self):
        pass

    @property
    @abstractmethod
    def desc(self):
        pass

    @property
    def name(self):
        return self.config.name

    @property
    @abstractmethod
    def index2code(self):
        pass

    @property
    @abstractmethod
    def index2desc(self):
        pass

    def __len__(self):
        return len(self.codes)

    def __bool__(self):
        return len(self.codes) > 0

    def __str__(self):
        return self.name

    def __contains__(self, code):
        """Returns True if `code` is contained in the current scheme."""
        return code in self.codes

    def search_regex(self, query, regex_flags=re.I):
        """
        a regex-supported search of codes by a `query` string. the search is \
            applied on the code description.\
            for example, you can use it to return all codes related to cancer \
            by setting the `query = 'cancer'` \
            and `regex_flags = re.i` (for case-insensitive search).
        """
        return set(
            filter(lambda c: re.match(query, self.desc[c], flags=regex_flags),
                   self.codes))

    def mapper_to(self, target_scheme: str):
        return CodeMap.get_mapper(self.name, target_scheme)

    def codeset2vec(self, codeset: Set[str]) -> CodesVector:
        vec = np.zeros(len(self), dtype=bool)
        try:
            for c in codeset:
                vec[self.index[c]] = True
        except KeyError as missing:
            logging.error(f'Code {missing} is missing.'
                          f'Accepted keys: {self.index.keys()}')

        return CodesVector(vec, self.name)

    def empty_vector(self) -> CodesVector:
        return CodesVector.empty(self.name)

    @property
    def supported_targets(self):
        return tuple(t for s, t in CodeMap._load_maps.keys() if s == self.name)

    def as_dataframe(self):
        index = sorted(self.index.values())
        return pd.DataFrame(
            {
                'code': self.index2code,
                'desc': self.index2desc,
            },
            index=index
        )


class FlatScheme(CodingScheme):
    config: CodingSchemeConfig
    _codes: List[str]
    _index: Dict[str, int]
    _desc: Dict[str, str]
    _index2code: Dict[int, str]
    _index2desc: Dict[int, str]

    def __init__(self, config: CodingSchemeConfig, codes: List[str], index: Dict[str, int], desc: Dict[str, str]):
        super().__init__(config=config)

        logging.debug(f'Constructing {config.name} ({type(self)}) scheme')
        self._codes = codes
        self._index = index
        self._desc = desc

        self._index2code = {idx: code for code, idx in index.items()}
        self._index2desc = {index[code]: desc for code, desc in desc.items()}

        self._check_types()

    def _check_types(self):
        for collection in [self.codes, self.index, self.desc]:
            assert all(
                type(c) == str
                for c in collection), f"{self}: All name types should be str."

    @property
    def codes(self):
        return self._codes

    @property
    def index(self):
        return self._index

    @property
    def desc(self):
        return self._desc

    @property
    def index2code(self):
        return self._index2code

    @property
    def index2desc(self):
        return self._index2desc


class BinaryScheme(FlatScheme):

    def __init__(self, codes: List[str], index: Dict[str, int], desc: Dict[str, str], name: str):
        assert all(len(c) == 2 for c in (codes, index, desc)), \
            f"{self}: Codes should be of length 2."
        super().__init__(codes, index, desc, name)

    def codeset2vec(self, code: str):
        return BinaryCodesVector(np.array(self.index[code], dtype=bool), self)

    def __len__(self):
        return 1


class SchemeWithMissing(FlatScheme):
    _missing_code: str

    def __init__(self, config: CodingSchemeConfig,
                 codes: List[str], index: Dict[str, int], desc: Dict[str, str], name: str, missing_code: str):
        super().__init__(config, codes, index, desc)
        self._missing_code = missing_code

    def __len__(self):
        return len(self.codes) - 1

    @property
    def missing_code(self):
        return self._missing_code

    def codeset2vec(self, codeset: Set[str]) -> CodesVectorWithMissing:
        vec = np.zeros(len(self), dtype=bool)
        try:
            for c in codeset:
                idx = self.index[c]
                if idx >= 0:
                    vec[idx] = True
        except KeyError as missing:
            logging.error(f'Code {missing} is missing.'
                          f'Accepted keys: {self.index.keys()}')

        return CodesVectorWithMissing(vec, self.name)

    def empty_vector(self) -> CodesVectorWithMissing:
        return CodesVectorWithMissing.empty(self.name)


class CodeMapConfig(Config):
    source_scheme: str
    target_scheme: str
    mapped_to_dag_space: bool = False


class CodeMap(Module):
    config: CodeMapConfig
    _source_scheme: Union[CodingScheme, HierarchicalScheme]
    _target_scheme: Union[CodingScheme, HierarchicalScheme]
    _data: Dict[str, Set[str]]

    _maps: ClassVar[Dict[Tuple[str, str], CodeMap]] = {}
    # Possible Mappings, Lazy-loaded maps.
    _load_maps: ClassVar[Dict[Tuple[str, str], Callable]] = {}
    _maps_lock: ClassVar[Dict[Tuple[str, str], Lock]] = defaultdict(Lock)

    def __init__(self, config: CodeMapConfig, data: Dict[str, Set[str]]):
        super().__init__(config=config)
        self._data = data
        self._source_scheme = CodingScheme.from_name(config.source_scheme)
        self._target_scheme = CodingScheme.from_name(config.target_scheme)

        self._source_index = self._source_scheme.index
        self._target_index = self._target_scheme.index
        self._target_desc = self._target_scheme.desc

        if config.source_scheme != config.target_scheme and config.mapped_to_dag_space:
            self._t_index = self._target_scheme.dag_index
            self._t_desc = self._target_scheme.dag_desc

        self.register_map(config.source_scheme, config.target_scheme, self)

    @classmethod
    def register_map(cls, source_scheme: str, target_scheme: str, mapper: CodeMap):
        cls._maps[(source_scheme, target_scheme)] = mapper

    @classmethod
    def register_chained_map(cls, s_scheme: str, inter_scheme: str, t_scheme: str):
        map1 = CodeMap.get_mapper(s_scheme, inter_scheme)
        map2 = CodeMap.get_mapper(inter_scheme, t_scheme)
        assert map1.config.mapped_to_dag_space == False
        assert map2.config.mapped_to_dag_space == False

        bridge = lambda x: set.union(*[map2[c] for c in map1[x]])
        s_scheme_object = CodingScheme.from_name(s_scheme)

        # Supported codes in the new map are the intersection of the source codes and the source codes of the first map
        new_source_codes = set(s_scheme_object.codes) & set(map1.keys())
        config = CodeMapConfig(s_scheme, t_scheme, False)
        cls.register_map(s_scheme, t_scheme, CodeMap(config=config, data={c: bridge(c) for c in new_source_codes}))

    @classmethod
    def register_chained_map_loader(cls, s_scheme: str, inter_scheme: str, t_scheme: str):
        cls._load_maps[(s_scheme, t_scheme)] = lambda: cls.register_chained_map(s_scheme, inter_scheme, t_scheme)

    @classmethod
    def register_map_loader(cls, source_scheme: str, target_scheme: str, loader: Callable):
        cls._load_maps[(source_scheme, target_scheme)] = loader

    def __str__(self):
        return f'{self.source_scheme.name}->{self.target_scheme.name}'

    def __hash__(self):
        return hash(str(self))

    def __bool__(self):
        return len(self) > 0

    @property
    def target_index(self):
        return self._target_index

    @property
    def target_desc(self):
        return self._target_desc

    @property
    def source_index(self):
        return self._source_index

    @property
    def source_scheme(self):
        return self._source_scheme

    @property
    def target_scheme(self):
        return self._target_scheme

    @property
    def mapped_to_dag_space(self):
        return self.config.mapped_to_dag_space

    @classmethod
    def has_mapper(cls, source_scheme: str, target_scheme: str):
        key = (source_scheme, target_scheme)
        return key in cls._maps or key[0] == key[1] or key in cls._load_maps

    @classmethod
    def get_mapper(cls, source_scheme: str, target_scheme: str) -> CodeMap:
        if not cls.has_mapper(source_scheme, target_scheme):
            logging.warning(f'Mapping {source_scheme}->{target_scheme} is not available')
            return NullCodeMap()

        key = (source_scheme, target_scheme)
        with cls._maps_lock[key]:
            if key in cls._maps:
                return cls._maps[key]

            if key[0] == key[1]:
                return IdentityCodeMap(source_scheme)

            if key in cls._load_maps:
                cls._load_maps[key]()

            return cls._maps[key]

    def __getitem__(self, item):
        return self._data[item]

    def keys(self):
        return self._data.keys()

    def map_codeset(self, codeset: Set[str]):
        return set().union(*[self[c] for c in codeset])

    def target_code_ancestors(self, t_code: str, include_itself=True):
        if self.config.mapped_to_dag_space == False:
            t_code = self.target_scheme.code2dag[t_code]
        return self.target_scheme.code_ancestors_bfs(t_code,
                                                     include_itself=include_itself)

    def codeset2vec(self, codeset: Set[str]):
        index = self.target_index
        vec = np.zeros(len(index), dtype=bool)
        try:
            for c in codeset:
                vec[index[c]] = True
        except KeyError as missing:
            logging.error(
                f'Code {missing} is missing. Accepted keys: {index.keys()}')

        return CodesVector(vec, self.target_scheme)

    def codeset2dagset(self, codeset: Set[str]):
        if self.config.mapped_to_dag_space == False:
            return set(self.target_scheme.code2dag[c] for c in codeset)
        else:
            return codeset

    def codeset2dagvec(self, codeset: Set[str]):
        if self.config.mapped_to_dag_space == False:
            codeset = set(self.target_scheme.code2dag[c] for c in codeset)
            index = self.target_scheme.dag_index
        else:
            index = self.target_scheme
        vec = np.zeros(len(index), dtype=bool)
        try:
            for c in codeset:
                vec[index[c]] = True
        except KeyError as missing:
            logging.error(
                f'Code {missing} is missing. Accepted keys: {index.keys()}')

        return vec


#     def log_unrecognised_range(self, json_fname):
#         if self._unrecognised_range:
#             write_config(
#                 {
#                     'code_scheme': self._t_scheme.name,
#                     'conv_file': self._conv_file,
#                     'n': len(self._unrecognised_range),
#                     'codes': sorted(self._unrecognised_range)
#                 }, json_fname)

#     def log_unrecognised_domain(self, json_fname):
#         if self._unrecognised_domain:
#             write_config(
#                 {
#                     'code_scheme': self._s_scheme.name,
#                     'conv_file': self._conv_file,
#                     'n': len(self._unrecognised_domain),
#                     'codes': sorted(self._unrecognised_domain)
#                 }, json_fname)

#     def log_uncovered_source_codes(self, json_fname):
#         res = self.report_source_discrepancy()
#         uncovered = res["fwd_diff"]
#         if len(uncovered) > 0:
#             write_config(
#                 {
#                     'code_scheme': self._s_scheme.name,
#                     'conv_file': self._conv_file,
#                     'n': len(uncovered),
#                     'p': len(uncovered) / len(self._s_scheme.index),
#                     'codes': sorted(uncovered),
#                     'desc': {
#                         c: self._s_scheme.desc[c]
#                         for c in uncovered
#                     }
#                 }, json_fname)

#     def report_discrepancy(self):
#         assert all(type(c) == str
#                    for c in self), f"All M_domain({self}) types should be str"
#         assert all(type(c) == str for c in set().union(
#             *self.values())), f"All M_range({self}) types should be str"
#         try:
#             s_discrepancy = self.report_source_discrepancy()
#             t_discrepancy = self.report_target_discrepancy()
#         except TypeError as e:
#             logging.error(f'{self}: {e}')

#         if s_discrepancy['fwd_p'] > 0:
#             logging.debug('Source discrepancy')
#             logging.debug(s_discrepancy['msg'])

#         if t_discrepancy['fwd_p'] > 0:
#             logging.debug('Target discrepancy')
#             logging.debug(t_discrepancy['msg'])

#     def report_target_discrepancy(self):
#         """
#         S={S-Space}  ---M={S:T MAPPER}---> T={T-Space}
#         M-domain = M.keys()
#         M-range = set().union(*M.values())
#         """
#         M_range = set().union(*self.values())
#         T = set(self.t_index)
#         fwd_diff = M_range - T
#         bwd_diff = T - M_range
#         fwd_p = len(fwd_diff) / len(M_range)
#         bwd_p = len(bwd_diff) / len(T)
#         msg = f"""M: {self} \n
# Mapping converts to codes that are not supported by the target scheme.
# |M-range - T|={len(fwd_diff)}; \
# |M-range - T|/|M-range|={fwd_p:0.2f}); \
# first5(M-range - T)={sorted(fwd_diff)[:5]}\n
# Target codes that not covered by the mapping. \
# |T - M-range|={len(bwd_diff)}; \
# |T - M-range|/|T|={bwd_p:0.2f}; \
# first5(T - M-range)={sorted(bwd_diff)[:5]}\n
# |M-range|={len(M_range)}; \
# first5(M-range) {sorted(M_range)[:5]}.\n
# |T|={len(T)}; first5(T)={sorted(T)[:5]}
#         """
#         return dict(fwd_diff=fwd_diff,
#                     bwd_diff=bwd_diff,
#                     fwd_p=fwd_p,
#                     bwd_p=bwd_p,
#                     msg=msg)

#     def report_source_discrepancy(self):
#         """
#         S={S-Space}  ---M={S:T MAPPER}---> T={T-Space}
#         M-domain = M.keys()
#         M-range = set().union(*M.values())
#         """
#         M_domain = set(self.keys())
#         S = set(self.s_index)
#         fwd_diff = S - M_domain
#         bwd_diff = M_domain - S
#         fwd_p = len(fwd_diff) / len(S)
#         bwd_p = len(bwd_diff) / len(M_domain)
#         msg = f"""M: {self} \n
# Mapping converts codes that are not supported by the source scheme.\
# |M-domain - S|={len(bwd_diff)}; \
# |M-domain - S|/|M-domain|={bwd_p:0.2f}); \
# first5(M-domain - S)={sorted(bwd_diff)[:5]}\n
# Source codes that not covered by the mapping. \
# |S - M-domain|={len(fwd_diff)}; \
# |S - M-domain|/|S|={fwd_p:0.2f}; \
# first5(S - M-domain)={sorted(fwd_diff)[:5]}\n
# |M-domain|={len(M_domain)}; \
# first5(M-domain) {sorted(M_domain)[:5]}.\n
# |S|={len(S)}; first5(S)={sorted(S)[:5]}
#         """
#         return dict(fwd_diff=fwd_diff,
#                     bwd_diff=bwd_diff,
#                     fwd_p=fwd_p,
#                     bwd_p=bwd_p,
#                     msg=msg)
class IdentityCodeMap(CodeMap):

    def __init__(self, scheme: str):
        config = CodeMapConfig(source_scheme=scheme,
                               target_scheme=scheme,
                               mapped_to_dag_space=False)
        scheme = CodingScheme.from_name(scheme)
        data = {c: {c} for c in scheme.codes}
        super().__init__(config=config, data=data)

    def map_codeset(self, codeset):
        return codeset


class NullCodeMap(CodeMap):

    def __init__(self):
        config = CodeMapConfig(source_scheme='null',
                               target_scheme='null',
                               mapped_to_dag_space=False)
        super().__init__(config=config, data={})

    def map_codeset(self, codeset):
        return None

    def codeset2vec(self, codeset):
        return None

    def __bool__(self):
        return False


class CodesVector(Data):
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    vec: np.ndarray
    scheme: str  # Coding scheme for diagnostic codes

    @property
    def scheme_object(self):
        return CodingScheme.from_name(self.scheme)

    @classmethod
    def empty_like(cls, other: CodesVector) -> CodesVector:
        return cls(np.zeros_like(other.vec), other.scheme)

    @classmethod
    def empty(cls, scheme: str) -> CodesVector:
        return cls(np.zeros(len(CodingScheme.from_name(scheme)), dtype=bool), scheme)

    def to_codeset(self):
        index = self.vec.nonzero()[0]
        scheme = self.scheme_object
        return set(scheme.index2code[i] for i in index)

    def union(self, other):
        return CodesVector(self.vec | other.vec, self.scheme)

    def __len__(self):
        return len(self.vec)


class CodesVectorWithMissing(CodesVector):

    def to_codeset(self):
        index = self.vec.nonzero()[0]
        if len(index) == 0:
            return {self.scheme_object.missing_code}


class BinaryCodesVector(CodesVector):

    @classmethod
    def empty(cls, scheme: str):
        return cls(np.zeros(1, dtype=bool), scheme)

    def to_codeset(self):
        return {self.scheme_object.index2code[self.vec[0]]}

    def __len__(self):
        return 1


class NullScheme(FlatScheme):

    def __init__(self):
        super().__init__(CodingSchemeConfig('null'), [], {}, {})


class HierarchicalScheme(FlatScheme):

    def __init__(self,
                 config: CodingSchemeConfig,
                 codes: Optional[List[str]] = None,
                 index: Optional[Dict[str, int]] = None,
                 desc: Optional[Dict[str, str]] = None,
                 dag_codes: Optional[List[str]] = None,
                 dag_index: Optional[Dict[str, int]] = None,
                 dag_desc: Optional[Dict[str, str]] = None,
                 code2dag: Optional[Dict[str, str]] = None,
                 pt2ch: Optional[Dict[str, Set[str]]] = None,
                 ch2pt: Optional[Dict[str, Set[str]]] = None):
        super().__init__(config, codes, index, desc)

        self._dag_codes = dag_codes or codes
        self._dag_index = dag_index or index
        self._dag_desc = dag_desc or desc
        self._code2dag = code2dag or {c: c for c in codes}
        self._dag2code = {d: c for c, d in self._code2dag.items()}

        assert pt2ch or ch2pt, (
            "Should provide ch2pt or pt2ch connection dictionary")
        self._pt2ch = pt2ch or self.reverse_connection(ch2pt)
        self._ch2pt = ch2pt or self.reverse_connection(pt2ch)
        self._check_types()

    def _check_types(self):
        for collection in [
            self._dag_codes, self._dag_index, self._dag_desc,
            self._code2dag, self._dag2code, self._pt2ch, self._ch2pt
        ]:
            assert all(
                type(c) == str
                for c in collection), f"{self}: All name types should be str."

    def make_ancestors_mat(self, include_itself=True) -> np.ndarray:
        ancestors_mat = np.zeros((len(self.dag_index), len(self.dag_index)),
                                 dtype=bool)
        for code_i, i in self.dag_index.items():
            for ancestor_j in self.code_ancestors_bfs(code_i, include_itself):
                j = self._dag_index[ancestor_j]
                ancestors_mat[i, j] = 1

        return ancestors_mat

    @property
    def dag_index(self):
        return self._dag_index

    @property
    def dag_codes(self):
        return self._dag_codes

    @property
    def dag_desc(self):
        return self._dag_desc

    @property
    def code2dag(self):
        return self._code2dag

    @property
    def dag2code(self):
        return self._dag2code

    def __contains__(self, code):
        """Returns True if `code` is contained in the current hierarchy."""
        return code in self.dag_codes or code in self.codes

    @staticmethod
    def reverse_connection(connection):
        rev_connection = defaultdict(set)
        for node, conns in connection.items():
            for conn in conns:
                rev_connection[conn].add(node)
        return rev_connection

    @staticmethod
    def _bfs_traversal(connection, code, include_itself):
        result = OrderedDict()
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_code = q.pop(0)
            current_connections = connection.get(current_code, [])
            q.extend([c for c in current_connections if c not in result])
            if current_code not in result:
                result[current_code] = 1

        if not include_itself:
            del result[code]
        return list(result.keys())

    @staticmethod
    def _dfs_traversal(connection, code, include_itself):
        result = {code} if include_itself else set()

        def _traversal(_node):
            for conn in connection.get(_node, []):
                result.add(conn)
                _traversal(conn)

        _traversal(code)

        return list(result)

    @staticmethod
    def _dfs_edges(connection, code):
        result = set()

        def _edges(_node):
            for conn in connection.get(_node, []):
                result.add((_node, conn))
                _edges(conn)

        _edges(code)
        return result

    def code_ancestors_bfs(self, code, include_itself=True):
        return self._bfs_traversal(self._ch2pt, code, include_itself)

    def code_ancestors_dfs(self, code, include_itself=True):
        return self._dfs_traversal(self._ch2pt, code, include_itself)

    def code_successors_bfs(self, code, include_itself=True):
        return self._bfs_traversal(self._pt2ch, code, include_itself)

    def code_successors_dfs(self, code, include_itself=True):
        return self._dfs_traversal(self._pt2ch, code, include_itself)

    def ancestors_edges_dfs(self, code):
        return self._dfs_edges(self._ch2pt, code)

    def successors_edges_dfs(self, code):
        return self._dfs_edges(self._pt2ch, code)

    def least_common_ancestor(self, codes):
        while len(codes) > 1:
            a, b = codes[:2]
            a_ancestors = self.code_ancestors_bfs(a, True)
            b_ancestors = self.code_ancestors_bfs(b, True)
            last_len = len(codes)
            for ancestor in a_ancestors:
                if ancestor in b_ancestors:
                    codes = [ancestor] + codes[2:]
            if len(codes) == last_len:
                raise RuntimeError('Common Ancestor not Found!')
        return codes[0]

    def search_regex(self, query, regex_flags=re.I):
        """
        A regex-based search of codes by a `query` string. the search is \
            applied on the code descriptions. for example, you can use it \
            to return all codes related to cancer by setting the \
            `query = 'cancer'` and `regex_flags = re.i` \
            (for case-insensitive search). For all found codes, \
            their successor codes are also returned in the resutls.
        """

        codes = filter(
            lambda c: re.match(query, self._desc[c], flags=regex_flags),
            self.codes)

        dag_codes = filter(
            lambda c: re.match(query, self._dag_desc[c], flags=regex_flags),
            self.dag_codes)

        all_codes = set(map(self._code2dag.get, codes)) | set(dag_codes)

        for c in list(all_codes):
            all_codes.update(self.code_successors_dfs(c))

        return all_codes


class Ethnicity(FlatScheme):
    pass


class AbstractGroupedProcedures(CodingScheme):

    def __init__(self, groups, aggregation, aggregation_groups, **init_kwargs):
        super().__init__(**init_kwargs)
        self._groups = groups
        self._aggregation = aggregation
        self._aggregation_groups = aggregation_groups

    @property
    def groups(self):
        return self._groups

    @property
    def aggregation(self):
        return self._aggregation

    @property
    def aggregation_groups(self):
        return self._aggregation_groups


def register_gender_scheme():
    CodingScheme.register_scheme(BinaryScheme(CodingSchemeConfig('gender'),
                                              codes=['M', 'F'], index={'M': 0, 'F': 1},
                                              desc={'M': 'male', 'F': 'female'}))


_OUTCOME_DIR = os.path.join(_RSC_DIR, 'outcome_filters')


class OutcomeExtractorConfig(CodingSchemeConfig):
    name: str
    spec_file: str


class OutcomeExtractor(FlatScheme):
    config: OutcomeExtractorConfig
    _base_scheme: CodingScheme

    _spec_files: ClassVar[Dict[str, str]] = {}

    def __init__(self, config: OutcomeExtractorConfig):

        specs = self.spec_from_json(config.spec_file)

        self._base_scheme = CodingScheme.from_name(specs['code_scheme'])
        codes = [
            c for c in sorted(self.base_scheme.index)
            if c not in specs['exclude_codes']
        ]

        index = dict(zip(codes, range(len(codes))))
        desc = {c: self.base_scheme.desc[c] for c in codes}
        super().__init__(config=config,
                         codes=codes,
                         index=index,
                         desc=desc)

    @classmethod
    def register_outcome_extractor_loader(cls, name: str, spec_file: str):
        def load():
            config = OutcomeExtractorConfig(name=name, spec_file=spec_file)
            cls.register_scheme(OutcomeExtractor(config))

        cls._spec_files[name] = spec_file
        cls.register_scheme_loader(name, load)

    @classmethod
    def from_name(cls, name):
        outcome_extractor = super().from_name(name)
        assert isinstance(outcome_extractor,
                          OutcomeExtractor), f'OutcomeExtractor expected, got {type(outcome_extractor)}'
        return outcome_extractor

    @property
    def base_scheme(self):
        return self._base_scheme

    @property
    def outcome_dim(self):
        return len(self.index)

    def _map_codeset(self, codeset: Set[str], base_scheme: str):
        m = CodeMap.get_mapper(base_scheme, self._base_scheme.name)
        codeset = m.map_codeset(codeset)

        if m.config.mapped_to_dag_space:
            codeset &= set(m.target_scheme.dag2code)
            codeset = set(m.target_scheme.dag2code[c] for c in codeset)

        return codeset & set(self.codes)

    def mapcodevector(self, codes: CodesVector):
        vec = np.zeros(len(self.index), dtype=bool)
        for c in self._map_codeset(codes.to_codeset(), codes.scheme):
            vec[self.index[c]] = True
        return CodesVector(np.array(vec), self.name)

    @classmethod
    def supported_outcomes(cls, base_scheme: str):
        outcome_base = {
            k: load_config(v, relative_to=_OUTCOME_DIR)['code_scheme']
            for k, v in cls._spec_files.items()
        }
        return tuple(k for k, v in outcome_base.items()
                     if v == base_scheme or v in CodingScheme.from_name(base_scheme).supported_targets)

    @staticmethod
    def spec_from_json(json_file: str):
        conf = load_config(json_file, relative_to=_OUTCOME_DIR)

        if 'exclude_branches' in conf:
            # TODO
            return None
        elif 'select_branches' in conf:
            # TODO
            return None
        elif 'selected_codes' in conf:
            t_scheme = CodingScheme.from_name(conf['code_scheme'])
            conf['exclude_codes'] = [
                c for c in t_scheme.codes if c not in conf['selected_codes']
            ]
            return conf
        elif 'exclude_codes' in conf:
            return conf
