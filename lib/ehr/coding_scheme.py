"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from __future__ import annotations
from abc import ABC, abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from typing import Set, Optional
from threading import Lock
import re
import os
import gzip
import xml.etree.ElementTree as ET
import logging

import numpy as np
import pandas as pd
import equinox as eqx
from ..utils import write_config, load_config

_DIR = os.path.dirname(__file__)
_RSC_DIR = os.path.join(_DIR, 'resources')
_CCS_DIR = os.path.join(_RSC_DIR, 'CCS')
singleton_lock = Lock()
maps_lock = Lock()


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        with singleton_lock:
            if cls._instances.get(cls, None) is None:
                cls._instances[cls] = super(Singleton,
                                            cls).__new__(cls, *args, **kwargs)

        return Singleton._instances[cls]


class _CodeMapper(defaultdict):
    maps = {}

    def __init__(self, s_scheme, t_scheme, t_dag_space, *args, **kwargs):
        """
        Constructor of _CodeMapper object.
        Args:
            s_scheme: source scheme object.
            t_scheme: target scheme object.
            t_dag_space: flag to enforce mapping to Directed Acyclic Graph
            (DAG) space instead of the leaf node set (i.e. flat space).
        """
        super(_CodeMapper, self).__init__(*(args or (set, )))

        self._s_scheme = s_scheme
        self._t_scheme = t_scheme
        self._s_index = s_scheme.index
        self._t_dag_space = t_dag_space
        if s_scheme != t_scheme and t_dag_space:
            self._t_index = t_scheme.dag_index
            self._t_desc = t_scheme.dag_desc
        else:
            self._t_index = t_scheme.index
            self._t_desc = t_scheme.desc

        self._unrecognised_range = kwargs.get('unrecognised_target', set())
        self._unrecognised_domain = kwargs.get('unrecognised_source', set())
        self._conv_file = kwargs.get('conv_file', '')
        _CodeMapper.maps[(type(s_scheme), type(t_scheme))] = self

    def __str__(self):
        return f'{self.s_scheme.name}->{self.t_scheme.name}'

    def __hash__(self):
        return hash(str(self))

    def log_unrecognised_range(self, json_fname):
        if self._unrecognised_range:
            write_config(
                {
                    'code_scheme': self._t_scheme.name,
                    'conv_file': self._conv_file,
                    'n': len(self._unrecognised_range),
                    'codes': sorted(self._unrecognised_range)
                }, json_fname)

    def log_unrecognised_domain(self, json_fname):
        if self._unrecognised_domain:
            write_config(
                {
                    'code_scheme': self._s_scheme.name,
                    'conv_file': self._conv_file,
                    'n': len(self._unrecognised_domain),
                    'codes': sorted(self._unrecognised_domain)
                }, json_fname)

    def log_uncovered_source_codes(self, json_fname):
        res = self.report_source_discrepancy()
        uncovered = res["fwd_diff"]
        if len(uncovered) > 0:
            write_config(
                {
                    'code_scheme': self._s_scheme.name,
                    'conv_file': self._conv_file,
                    'n': len(uncovered),
                    'p': len(uncovered) / len(self._s_scheme.index),
                    'codes': sorted(uncovered),
                    'desc': {
                        c: self._s_scheme.desc[c]
                        for c in uncovered
                    }
                }, json_fname)

    def report_discrepancy(self):
        assert all(type(c) == str
                   for c in self), f"All M_domain({self}) types should be str"
        assert all(type(c) == str for c in set().union(
            *self.values())), f"All M_range({self}) types should be str"
        try:
            s_discrepancy = self.report_source_discrepancy()
            t_discrepancy = self.report_target_discrepancy()
        except TypeError as e:
            logging.error(f'{self}: {e}')

        if s_discrepancy['fwd_p'] > 0:
            logging.warning('Source discrepancy')
            logging.warning(s_discrepancy['msg'])

        if t_discrepancy['fwd_p'] > 0:
            logging.warning('Target discrepancy')
            logging.warning(t_discrepancy['msg'])

    def report_target_discrepancy(self):
        """
        S={S-Space}  ---M={S:T MAPPER}---> T={T-Space}
        M-domain = M.keys()
        M-range = set().union(*M.values())
        """
        M_range = set().union(*self.values())
        T = set(self.t_index)
        fwd_diff = M_range - T
        bwd_diff = T - M_range
        fwd_p = len(fwd_diff) / len(M_range)
        bwd_p = len(bwd_diff) / len(T)
        return dict(fwd_diff=fwd_diff,
                    bwd_diff=bwd_diff,
                    fwd_p=fwd_p,
                    bwd_p=bwd_p,
                    msg=f"""M: {self} \n
                            M_range - T ({len(fwd_diff)}, p={fwd_p}):
                            {sorted(fwd_diff)[:5]}...\n
                            T - M_range ({len(bwd_diff)}, p={bwd_p}):
                            {sorted(bwd_diff)[:5]}...\n
                            M_range ({len(M_range)}):
                            {sorted(M_range)[:5]}...\n
                            T ({len(T)}): {sorted(T)[:5]}...""")

    def report_source_discrepancy(self):
        """
        S={S-Space}  ---M={S:T MAPPER}---> T={T-Space}
        M-domain = M.keys()
        M-range = set().union(*M.values())
        """
        M_domain = set(self.keys())
        S = set(self.s_index)
        fwd_diff = S - M_domain
        bwd_diff = M_domain - S
        fwd_p = len(fwd_diff) / len(S)
        bwd_p = len(bwd_diff) / len(M_domain)
        return dict(fwd_diff=fwd_diff,
                    bwd_diff=bwd_diff,
                    fwd_p=fwd_p,
                    bwd_p=bwd_p,
                    msg=f"""M: {self} \n
                            S - M_domain ({len(fwd_diff)}, p={fwd_p}):
                            {sorted(fwd_diff)[:5]}...\n
                            M_domain - S ({len(bwd_diff)}, p={bwd_p}):
                            {sorted(bwd_diff)[:5]}...\n
                            M_domain ({len(M_domain)}):
                            {sorted(M_domain)[:5]}...\n
                            S ({len(S)}): {sorted(S)[:5]}...""")

    @property
    def t_index(self):
        return self._t_index

    @property
    def t_desc(self):
        return self._t_desc

    @property
    def s_index(self):
        return self._s_index

    @property
    def s_scheme(self):
        return self._s_scheme

    @property
    def t_scheme(self):
        return self._t_scheme

    @property
    def t_dag_space(self):
        return self._t_dag_space

    @classmethod
    def get_mapper(cls, s_scheme, t_scheme):
        with maps_lock:
            if any(isinstance(s, NullScheme) for s in (s_scheme, t_scheme)):
                return _NullCodeMapper()
            key = (type(s_scheme), type(t_scheme))

            if key in _CodeMapper.maps:
                return _CodeMapper.maps[key]

            if key[0] == key[1]:
                return _IdentityCodeMapper(s_scheme)

            if key in load_maps:
                load_maps[key]()

            mapper = _CodeMapper.maps.get(key)
            if mapper:
                mapper.report_discrepancy()
            else:
                logging.warning(f'Mapping {key} is not available')

            return mapper

    def map_codeset(self, codeset: Set[str]):
        return set().union(*[self[c] for c in codeset])

    def t_code_ancestors(self, t_code: str, include_itself=True):
        if self._t_dag_space == False:
            t_code = self.t_scheme.code2dag[t_code]
        return self.t_scheme.code_ancestors_bfs(t_code,
                                                include_itself=include_itself)

    def codeset2vec(self, codeset: Set[str]):
        index = self.t_index
        vec = np.zeros(len(self.t_index), dtype=bool)
        try:
            for c in codeset:
                vec[index[c]] = True
        except KeyError as missing:
            logging.error(
                f'Code {missing} is missing. Accepted keys: {index.keys()}')

        return CodesVector(vec, self._t_scheme)

    def codeset2dagset(self, codeset: Set[str]):
        if self._t_dag_space == False:
            return set(self.t_scheme.code2dag[c] for c in codeset)
        else:
            return codeset

    def codeset2dagvec(self, codeset: Set[str]):
        if self._t_dag_space == False:
            codeset = set(self.t_scheme.code2dag[c] for c in codeset)
            index = self.t_scheme.dag_index
        else:
            index = self.t_index
        vec = np.zeros(len(index), dtype=bool)
        try:
            for c in codeset:
                vec[index[c]] = True
        except KeyError as missing:
            logging.error(
                f'Code {missing} is missing. Accepted keys: {index.keys()}')

        return vec


class _IdentityCodeMapper(_CodeMapper):

    def __init__(self, scheme, *args):
        super().__init__(s_scheme=scheme,
                         t_scheme=scheme,
                         t_dag_space=False,
                         *args)
        self.update({c: {c} for c in scheme.codes})

    def map_codeset(self, codeset):
        return codeset


class _NullCodeMapper(_CodeMapper):

    def __init__(self, *args):
        super().__init__(s_scheme=NullScheme(),
                         t_scheme=NullScheme(),
                         t_dag_space=False,
                         *args)

    def map_codeset(self, codeset):
        return None

    def codeset2vec(self, codeset):
        return None

    def __bool__(self):
        return False


class AbstractScheme:

    def __init__(self, codes, index, desc, name):
        logging.debug(f'Constructing {name} ({type(self)}) scheme')
        self._codes = codes
        self._index = index
        self._desc = desc
        self._name = name
        self._index2code = {idx: code for code, idx in index.items()}
        self._index2desc = {index[code]: desc for code, desc in desc.items()}

        for collection in [codes, index, desc]:
            assert all(
                type(c) == str
                for c in collection), f"{self}: All name types should be str."

        _IdentityCodeMapper(self)

    def __len__(self):
        return len(self.codes)

    def __bool__(self):
        return len(self.codes) > 0

    def __str__(self):
        return self.name

    def __contains__(self, code):
        """Returns True if `code` is contained in the current scheme."""
        return code in self._codes

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
    def name(self):
        return self._name

    @property
    def index2code(self):
        return self._index2code

    @property
    def index2desc(self):
        return self._index2desc

    def search_regex(self, query, regex_flags=re.I):
        """
        a regex-supported search of codes by a `query` string. the search is \
            applied on the code description.\
            for example, you can use it to return all codes related to cancer \
            by setting the `query = 'cancer'` \
            and `regex_flags = re.i` (for case-insensitive search).
        """
        return set(
            filter(lambda c: re.match(query, self._desc[c], flags=regex_flags),
                   self.codes))

    def mapper_to(self, target_scheme):
        return _CodeMapper.get_mapper(self, target_scheme)

    def codeset2vec(self, codeset: Set[str]):
        vec = np.zeros(len(self), dtype=bool)
        try:
            for c in codeset:
                vec[self.index[c]] = True
        except KeyError as missing:
            logging.error(f'Code {missing} is missing.'
                          f'Accepted keys: {self.index.keys()}')

        return CodesVector(vec, self)

    def empty_vector(self):
        return CodesVector.empty(self)


class CodesVector(eqx.Module):
    """
    Admission class encapsulates the patient EHRs diagnostic/procedure codes.
    """
    vec: np.ndarray
    scheme: AbstractScheme  # Coding scheme for diagnostic codes

    @classmethod
    def empty_like(cls, other: CodesVector):
        return cls(np.zeros_like(other.vec), other.scheme)

    @classmethod
    def empty(cls, scheme: AbstractScheme):
        return cls(np.zeros(len(scheme), dtype=bool), scheme)

    def to_codeset(self):
        index = self.vec.nonzero()[0]
        return set(self.scheme.index2code[i] for i in index)

    def union(self, other):
        return CodesVector(self.vec | other.vec, self.scheme)

    def __len__(self):
        return len(self.vec)


class BinaryCodesVector(CodesVector):

    @classmethod
    def empty(cls, scheme: BinaryScheme):
        return cls(np.zeros(1, dtype=bool), scheme)

    def to_codeset(self):
        return set(self.scheme.index2code[self.vec[0]])

    def __len__(self):
        return 1


class BinaryScheme(AbstractScheme):

    def __init__(self, codes, index, desc, name):
        assert all(len(c) == 2 for c in (codes, index, desc)), \
            f"{self}: Codes should be of length 2."
        super().__init__(codes, index, desc, name)

    def codeset2vec(self, code: str):
        return BinaryCodesVector(np.array(self.index[code], dtype=bool), self)

    def __len__(self):
        return 1


class NullScheme(Singleton, AbstractScheme):

    def __init__(self):
        super().__init__([], {}, {}, 'none')


class HierarchicalScheme(AbstractScheme):

    def __init__(self,
                 dag_codes=None,
                 dag_index=None,
                 dag_desc=None,
                 code2dag=None,
                 pt2ch=None,
                 ch2pt=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._dag_codes = dag_codes or kwargs['codes']
        self._dag_index = dag_index or kwargs['index']
        self._dag_desc = dag_desc or kwargs['desc']

        self._code2dag = None
        if code2dag is not None:
            self._code2dag = code2dag
        else:
            # Identity
            self._code2dag = {c: c for c in kwargs['codes']}

        self._dag2code = {d: c for c, d in self._code2dag.items()}

        assert pt2ch or ch2pt, "Should provide ch2pt or pt2ch connection dictionary"
        if ch2pt and pt2ch:
            self._ch2pt = ch2pt
            self._pt2ch = pt2ch
        if pt2ch is not None:
            # Direct parent-children relationship
            self._pt2ch = pt2ch
            # Direct children-parents relashionship
            self._ch2pt = self.reverse_connection(pt2ch)
        elif ch2pt is not None:
            self._ch2pt = ch2pt
            self._pt2ch = self.reverse_connection(ch2pt)

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
        return code in self._dag_codes or code in self._code2dag

    @staticmethod
    def reverse_connection(connection):
        rconnection = defaultdict(set)
        for node, conns in connection.items():
            for conn in conns:
                rconnection[conn].add(node)
        return rconnection

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
        A regex-based search of codes by a `query` string. the search is
        applied on the code descriptions.
        for example, you can use it to return all codes related to cancer
        by setting the `query = 'cancer'` and `regex_flags = re.i` (for case-insensitive search).
        For all found codes, their successor codes are also returned in the resutls.
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

    def to_digraph(self, discard_set=set(), node_attrs={}):
        """
        Generate a networkx.DiGraph (Directed Graph) representing the hierarchy.
        Filters can be applied through `discard_set`. Additional attributes can be added to the nodes through the dictionary `node_attrs`.
        """
        dag = nx.DiGraph()

        def _populate_dag(ch):
            for pt in self._ch2pt.get(ch, []):
                dag.add_edge(ch, pt)
                _populate_dag(pt)

        for c in set(self._dag_codes) - set(discard_set):
            _populate_dag(c)

        for attr_name, attr_dict in node_attrs.items():
            for node in dag.nodes:
                dag.nodes[node][attr_name] = attr_dict.get(node, '')
        return dag


class ICDCommons(HierarchicalScheme, metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def add_dots(code):
        pass

    @staticmethod
    def _deselect_subtree(pt2ch, sub_root):
        to_del = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        pt2ch = pt2ch.copy()
        to_del = set(to_del) & set(pt2ch.keys())
        for node_idx in to_del:
            del pt2ch[node_idx]
        return pt2ch

    @staticmethod
    def _select_subtree(pt2ch, sub_root):
        to_keep = HierarchicalScheme._bfs_traversal(pt2ch, sub_root, True)
        to_keep = set(to_keep) & set(pt2ch.keys())
        return {idx: pt2ch[idx] for idx in to_keep}

    @staticmethod
    def load_conv_table(s_scheme, t_scheme, conv_fname):
        df = pd.read_csv(os.path.join(_RSC_DIR, conv_fname),
                         sep='\s+',
                         dtype=str,
                         names=['source', 'target', 'meta'])
        df['approximate'] = df['meta'].apply(lambda s: s[0])
        df['no_map'] = df['meta'].apply(lambda s: s[1])
        df['combination'] = df['meta'].apply(lambda s: s[2])
        df['scenario'] = df['meta'].apply(lambda s: s[3])
        df['choice_list'] = df['meta'].apply(lambda s: s[4])
        df['source'] = df['source'].map(s_scheme.add_dots)
        df['target'] = df['target'].map(t_scheme.add_dots)

        valid_target = df['target'].isin(t_scheme.index)
        unrecognised_target = set(df[~valid_target]["target"])
        if len(unrecognised_target) > 0:
            logging.warning(f"""
                            {s_scheme}->{t_scheme} Unrecognised t_codes
                            ({len(unrecognised_target)}):
                            {sorted(unrecognised_target)[:20]}...""")

        valid_source = df['source'].isin(s_scheme.index)
        unrecognised_source = set(df[~valid_source]["source"])
        if len(unrecognised_source) > 0:
            logging.warning(f"""
                            {s_scheme}->{t_scheme} Unrecognised s_codes
                            ({len(unrecognised_source)}):
                            {sorted(unrecognised_source)[:20]}...""")

        df = df[valid_target & valid_source]
        # df['target'] = df['target'].map(t_scheme.code2dag)

        return {
            "df": df,
            "conv_file": conv_fname,
            "unrecognised_target": unrecognised_target,
            "unrecognised_source": unrecognised_source
        }

    @staticmethod
    def analyse_conversions(s_scheme, t_scheme, conv_fname):
        df = ICDCommons.load_conv_table(s_scheme, t_scheme, conv_fname)["df"]
        codes = list(df['source'][df['no_map'] == '1'])
        status = ['no_map' for _ in codes]
        for code, source_df in df[df['no_map'] == '0'].groupby('source'):
            codes.append(code)
            if len(source_df) == 1:
                status.append('11_map')
            elif len(set(source_df['scenario'])) > 1:
                status.append('ambiguous')
            elif len(set(source_df['choice_list'])) < len(source_df):
                status.append('1n_map(resolved)')
            else:
                status.append('1n_map')

        status = pd.DataFrame({'code': codes, 'status': status})
        return status

    @staticmethod
    def register_mappings(s_scheme, t_scheme, conv_fname):
        # For choice_list, represent each group by there common ancestor
        # def _resolve_choice_list(df):
        #     represent = set()
        #     for _, choice_list_df in df.groupby('choice_list'):
        #         choice_list = list(choice_list_df['target'])
        #         if len(choice_list) > 1:
        #             lca = t_scheme.least_common_ancestor(choice_list)
        #             represent.add(lca)
        #         elif len(choice_list) == 1:
        #             represent.add(choice_list[0])
        #     return represent

        res = ICDCommons.load_conv_table(s_scheme, t_scheme, conv_fname)
        conv_df = res["df"]
        status_df = ICDCommons.analyse_conversions(s_scheme, t_scheme,
                                                   conv_fname)
        map_kind = dict(zip(status_df['code'], status_df['status']))

        mapping = _CodeMapper(
            s_scheme,
            t_scheme,
            # t_dag_space=True,
            t_dag_space=False,
            unrecognised_source=res["unrecognised_source"],
            unrecognised_target=res["unrecognised_target"],
            conv_file=res["conv_file"])

        for code, df in conv_df.groupby('source'):
            kind = map_kind[code]
            if kind == 'no_map':
                continue
            mapping[code] = set(df['target'])
            # elif kind == '11_map' or kind == '1n_map':
            #     mapping[code] = set(df['target'])
            # elif kind == '1n_map(resolved)':
            #     mapping[code] = _resolve_choice_list(df)
            # elif kind == 'ambiguous':
            #     represent = set()
            #     for _, scenario_df in df.groupby('scenario'):
            #         represent.update(_resolve_choice_list(scenario_df))
            #     mapping[code] = represent


class DxICD10(Singleton, ICDCommons):
    """
    NOTE: for prediction targets, remember to exclude the following chapters:
        - 'chapter:19': 'Injury, poisoning and certain other consequences of external causes (S00-T88)',
        - 'chapter:20': 'External causes of morbidity (V00-Y99)',
        - 'chapter:21': 'Factors influencing health status and contact with health services (Z00-Z99)',
        - 'chapter:22': 'Codes for special purposes (U00-U85)'
    """

    @staticmethod
    def add_dots(code):
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 3:
            return code[:3] + '.' + code[3:]
        else:
            return code

    @staticmethod
    def distill_icd10_xml(filename):
        # https://www.cdc.gov/nchs/icd/Comprehensive-Listing-of-ICD-10-CM-Files.htm
        _ICD10_FILE = os.path.join(_RSC_DIR, filename)
        with gzip.open(_ICD10_FILE, 'r') as f:
            tree = ET.parse(f)
        root = tree.getroot()
        pt2ch = defaultdict(set)
        root_node = f'root:{root.tag}'
        desc = {root_node: 'root'}
        chapters = [ch for ch in root if ch.tag == 'chapter']

        def _traverse_diag_dfs(parent_name, dx_element):
            dx_name = next(e for e in dx_element if e.tag == 'name').text
            dx_desc = next(e for e in dx_element if e.tag == 'desc').text
            dx_name = f'dx:{dx_name}'
            desc[dx_name] = dx_desc
            pt2ch[parent_name].add(dx_name)

            diags = [dx for dx in dx_element if dx.tag == 'diag']
            for dx in diags:
                _traverse_diag_dfs(dx_name, dx)

        for chapter in chapters:
            ch_name = next(e for e in chapter if e.tag == 'name').text
            ch_desc = next(e for e in chapter if e.tag == 'desc').text
            ch_name = f'chapter:{ch_name}'
            pt2ch[root_node].add(ch_name)
            desc[ch_name] = ch_desc

            sections = [sec for sec in chapter if sec.tag == 'section']
            for section in sections:
                sec_name = section.attrib['id']
                sec_desc = next(e for e in section if e.tag == 'desc').text
                sec_name = f'section:{sec_name}'

                pt2ch[ch_name].add(sec_name)
                desc[sec_name] = sec_desc

                diags = [dx for dx in section if dx.tag == 'diag']
                for dx in diags:
                    _traverse_diag_dfs(sec_name, dx)

        icd_codes = sorted(c.split(':')[1] for c in desc if 'dx:' in c)
        icd_index = dict(zip(icd_codes, range(len(icd_codes))))
        icd_desc = {c: desc[f'dx:{c}'] for c in icd_codes}
        icd2dag = {c: f'dx:{c}' for c in icd_codes}
        dag_codes = [f'dx:{c}' for c in icd_codes] + sorted(
            c for c in set(desc) - set(icd2dag.values()))
        dag_index = dict(zip(dag_codes, range(len(dag_codes))))

        return {
            'codes': icd_codes,
            'index': icd_index,
            'desc': icd_desc,
            'code2dag': icd2dag,
            'dag_codes': dag_codes,
            'dag_index': dag_index,
            'dag_desc': desc,
            'pt2ch': pt2ch
        }

    def __init__(self):
        # https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2023/
        super().__init__(**self.distill_icd10_xml(
            'icd10cm_tabular_2023.xml.gz'),
                         name='dx_icd10')


class PrICD10(Singleton, ICDCommons):

    @staticmethod
    def add_dots(code):
        # No decimal point in ICD10-PCS
        return code

    @staticmethod
    def distill_icd10_xml(filename):
        _ICD10_FILE = os.path.join(_RSC_DIR, filename)

        with gzip.open(_ICD10_FILE, 'rt') as f:
            desc = {
                code: desc
                for code, desc in map(lambda line: line.strip().split(' ', 1),
                                      f.readlines())
            }
        codes = sorted(desc)
        index = dict(zip(codes, range(len(codes))))

        dag_desc = {'_': 'root'}
        code2dag = {}

        pt2ch = {}
        pos = list(f'p{i}' for i in range(7))
        df = pd.DataFrame(list(list(pos for pos in code) for code in desc),
                          columns=pos)

        def _distill_connections(branch_df, ancestors, ancestors_str,
                                 next_positions):
            if len(next_positions) == 0:
                leaf_code = ''.join(ancestors[1:])
                code2dag[leaf_code] = ancestors_str
            else:
                children = set()
                for branch, _df in branch_df.groupby(next_positions[0]):
                    child = ancestors + [branch]
                    child_str = ':'.join(child)
                    children.add(child_str)
                    _distill_connections(_df, child, child_str,
                                         next_positions[1:])
                dag_desc[ancestors_str] = ancestors_str
                pt2ch[ancestors_str] = children

        _distill_connections(df, ['_'], '_', pos)

        dag_codes = list(map(code2dag.get, codes))
        dag_codes.extend(sorted(pt2ch))
        dag_index = dict(zip(dag_codes, range(len(dag_codes))))

        return {
            'codes': codes,
            'index': index,
            'desc': desc,
            'code2dag': code2dag,
            'dag_codes': dag_codes,
            'dag_index': dag_index,
            'dag_desc': desc,
            'pt2ch': pt2ch
        }

    def __init__(self):
        super().__init__(**self.distill_icd10_xml(
            'icd10pcs_codes_2023.txt.gz'),
                         name='pr_icd10')


class DxICD9(Singleton, ICDCommons):
    _PR_ROOT_CLASS_ID = 'MM_CLASS_2'
    _DX_ROOT_CLASS_ID = 'MM_CLASS_21'
    _DX_DUMMY_ROOT_CLASS_ID = 'owl#Thing'

    @staticmethod
    def add_dots(code):
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if code[0] == 'E':
            if len(code) > 4:
                return code[:4] + '.' + code[4:]
            else:
                return code
        else:
            if len(code) > 3:
                return code[:3] + '.' + code[3:]
            else:
                return code

    @staticmethod
    def icd9_columns():
        # https://bioportal.bioontology.org/ontologies/HOM-ICD9
        ICD9CM_FILE = os.path.join(_RSC_DIR, 'HOM-ICD9.csv.gz')
        df = pd.read_csv(ICD9CM_FILE, dtype=str)
        df = df.fillna('')

        def retain_suffix(cell):
            if 'http' in cell:
                return cell.split('/')[-1]
            else:
                return cell

        df = df.applymap(retain_suffix)
        df.columns = list(map(retain_suffix, df.columns))

        df['level'] = 0
        for j in range(1, 7):
            level_rows = df[f'ICD9_LEVEL{j}'] != ''
            df.loc[level_rows, 'level'] = j

        return {
            'ICD9': list(df['C_BASECODE'].apply(lambda c: c.split(':')[-1])),
            'NODE_IDX': list(df['Class ID']),
            'PARENT_IDX': list(df['Parents']),
            'LABEL': list(df['Preferred Label']),
            'LEVEL': list(df['level'])
        }

    @staticmethod
    def parent_child_mappings(df):
        pt2ch = {}
        for pt, ch_df in df.groupby('PARENT_IDX'):
            pt2ch[pt] = set(ch_df['NODE_IDX'])

        # Remove dummy parent of diagnoses.
        del pt2ch[DxICD9._DX_DUMMY_ROOT_CLASS_ID]
        return pt2ch

    @staticmethod
    def generate_dictionaries(df):
        # df version for leaf nodes only (with non-empty ICD9 codes)
        df_leaves = df[df['ICD9'] != '']

        icd2dag = dict(zip(df_leaves['ICD9'], df_leaves['NODE_IDX']))

        # df version for internal nodes only (with empty ICD9 codes)
        df_internal = df[(df['ICD9'] == '') | df['ICD9'].isnull()]

        icd_codes = sorted(df_leaves['ICD9'])
        icd_index = dict(zip(icd_codes, range(len(icd_codes))))
        icd_desc = dict(zip(df_leaves['ICD9'], df_leaves['LABEL']))

        dag_codes = list(map(icd2dag.get, icd_codes)) + sorted(
            df_internal['NODE_IDX'])
        dag_index = dict(zip(dag_codes, range(len(dag_codes))))
        dag_desc = dict(zip(df['NODE_IDX'], df['LABEL']))

        return {
            'icd_codes': icd_codes,
            'icd_index': icd_index,
            'icd_desc': icd_desc,
            'icd2dag': icd2dag,
            'dag_codes': dag_codes,
            'dag_index': dag_index,
            'dag_desc': dag_desc
        }

    def __init__(self):
        df = pd.DataFrame(self.icd9_columns())
        pt2ch = self.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = self._deselect_subtree(pt2ch, self._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = self.generate_dictionaries(df)

        super().__init__(dag_codes=d['dag_codes'],
                         dag_index=d['dag_index'],
                         dag_desc=d['dag_desc'],
                         code2dag=d['icd2dag'],
                         pt2ch=pt2ch,
                         codes=d['icd_codes'],
                         index=d['icd_index'],
                         desc=d['icd_desc'],
                         name='dx_icd9')


class PrICD9(Singleton, ICDCommons):

    @staticmethod
    def add_dots(code):
        if '.' in code:
            # logging.debug(f'Code {code} already is in decimal format')
            return code
        if len(code) > 2:
            return code[:2] + '.' + code[2:]
        else:
            return code

    def __init__(self):
        df = pd.DataFrame(DxICD9.icd9_columns())
        pt2ch = DxICD9.parent_child_mappings(df)

        # Remove the procedure codes.
        pt2ch = self._select_subtree(pt2ch, DxICD9._PR_ROOT_CLASS_ID)

        # Remaining node indices in one set.
        nodes = set().union(set(pt2ch), *pt2ch.values())

        # Filter out the procedure code from the df.
        df = df[df['NODE_IDX'].isin(nodes)]

        d = DxICD9.generate_dictionaries(df)

        super().__init__(dag_codes=d['dag_codes'],
                         dag_index=d['dag_index'],
                         dag_desc=d['dag_desc'],
                         code2dag=d['icd2dag'],
                         pt2ch=pt2ch,
                         codes=d['icd_codes'],
                         index=d['icd_index'],
                         desc=d['icd_desc'],
                         name='pr_icd9')


class CCSCommons(HierarchicalScheme):
    _SCHEME_FILE = None
    _N_LEVELS = None

    @classmethod
    def ccs_columns(cls, icd9_scheme):
        df = pd.read_csv(os.path.join(_CCS_DIR, cls._SCHEME_FILE), dtype=str)
        icd_cname = '\'ICD-9-CM CODE\''

        df[icd_cname] = df[icd_cname].apply(lambda l: l.strip('\'').strip())
        df[icd_cname] = df[icd_cname].map(icd9_scheme.add_dots)
        valid_icd = df[icd_cname].isin(icd9_scheme.index)
        unrecognised_icd9 = set(df[~valid_icd][icd_cname])
        df = df[valid_icd]

        cols = {}
        for i in range(1, cls._N_LEVELS + 1):
            cols[f'I{i}'] = list(
                df[f'\'CCS LVL {i}\''].apply(lambda l: l.strip('\'').strip()))
            cols[f'L{i}'] = list(df[f'\'CCS LVL {i} LABEL\''].apply(
                lambda l: l.strip('\'').strip()))
        cols['ICD'] = list(df[icd_cname])

        return {
            "cols": cols,
            "unrecognised_icd9": unrecognised_icd9,
            "conv_file": cls._SCHEME_FILE
        }

    @staticmethod
    def register_mappings(ccs_scheme, icd9_scheme):
        res = ccs_scheme.ccs_columns(icd9_scheme)

        icd92ccs = _CodeMapper(icd9_scheme,
                               ccs_scheme,
                               t_dag_space=False,
                               unrecognised_source=res["unrecognised_icd9"],
                               conv_file=res["conv_file"])
        ccs2icd9 = _CodeMapper(ccs_scheme,
                               icd9_scheme,
                               t_dag_space=False,
                               unrecognised_target=res["unrecognised_icd9"],
                               conv_file=res["conv_file"])
        cols = res["cols"]
        n_rows = len(cols['ICD'])
        for i in range(n_rows):
            last_index = None
            for j in range(1, ccs_scheme._N_LEVELS + 1):
                level = cols[f'I{j}'][i]
                if level != '':
                    last_index = level
            if last_index != None:
                icode = cols['ICD'][i]
                icd92ccs[icode].add(last_index)
                ccs2icd9[last_index].add(icode)

    @classmethod
    def parent_child_mappings(cls, df):
        """Make dictionary for parent-child connections."""
        pt2ch = {'root': set(df['I1'])}
        levels = list(map(lambda i: f'I{i}', range(1, cls._N_LEVELS + 1)))

        for pt_col, ch_col in zip(levels[:-1], levels[1:]):
            df_ = df[(df[pt_col] != '') & (df[ch_col] != '')]
            df_ = df_[[pt_col, ch_col]].drop_duplicates()
            for parent_code, ch_df in df_.groupby(pt_col):
                pt2ch[parent_code] = set(ch_df[ch_col])
        return pt2ch

    @classmethod
    def desc_mappings(cls, df):
        """Make a dictionary for CCS labels."""
        desc = {'root': 'root'}
        levels = list(map(lambda i: f'I{i}', range(1, cls._N_LEVELS + 1)))
        descs = list(map(lambda i: f'L{i}', range(1, cls._N_LEVELS + 1)))
        for code_col, desc_col in zip(levels, descs):
            df_ = df[df[code_col] != '']
            df_ = df_[[code_col, desc_col]].drop_duplicates()
            code_desc = dict(zip(df_[code_col], df_[desc_col]))
            desc.update(code_desc)
        return desc

    @staticmethod
    def _code_ancestors_dots(code, include_itself=True):

        ancestors = {code} if include_itself else set()
        if code == 'root':
            return ancestors
        else:
            ancestors.add('root')

        indices = code.split('.')
        for i in reversed(range(1, len(indices))):
            parent = '.'.join(indices[0:i])
            ancestors.add(parent)
        return ancestors

    @classmethod
    def code_ancestors(cls, code, include_itself):
        return cls._code_ancestors_dots(code, include_itself)


class DxCCS(Singleton, CCSCommons):
    _SCHEME_FILE = 'ccs_multi_dx_tool_2015.csv.gz'
    _N_LEVELS = 4

    def __init__(self):
        cols = self.ccs_columns(DxICD9())["cols"]
        df = pd.DataFrame(cols)
        pt2ch = self.parent_child_mappings(df)
        desc = self.desc_mappings(df)
        codes = sorted(desc.keys())

        super().__init__(pt2ch=pt2ch,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='dx_ccs')


class PrCCS(Singleton, CCSCommons):
    _SCHEME_FILE = 'ccs_multi_pr_tool_2015.csv.gz'
    _N_LEVELS = 3

    def __init__(self):
        cols = self.ccs_columns(PrICD9())["cols"]
        df = pd.DataFrame(cols)
        pt2ch = self.parent_child_mappings(df)
        desc = self.desc_mappings(df)
        codes = sorted(desc.keys())

        super().__init__(pt2ch=pt2ch,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='pr_ccs')


class FlatCCSCommons(AbstractScheme):
    _SCHEME_FILE = None

    @classmethod
    def flatccs_columns(cls, icd9_scheme):
        filepath = os.path.join(_CCS_DIR, cls._SCHEME_FILE)
        df = pd.read_csv(filepath, skiprows=[0, 2], dtype=str)
        icd9_cname = '\'ICD-9-CM CODE\''
        cat_cname = '\'CCS CATEGORY\''
        desc_cname = '\'CCS CATEGORY DESCRIPTION\''
        df[icd9_cname] = df[icd9_cname].map(lambda c: c.strip('\'').strip())
        df[icd9_cname] = df[icd9_cname].map(icd9_scheme.add_dots)

        valid_icd9 = df[icd9_cname].isin(icd9_scheme.index)

        unrecognised_icd9 = set(df[~valid_icd9][icd9_cname])
        df = df[valid_icd9]

        code_col = list(df[cat_cname].map(lambda c: c.strip('\'').strip()))
        icd9_col = list(df[icd9_cname])
        desc_col = list(df[desc_cname].map(lambda d: d.strip('\'').strip()))

        return {
            'code': code_col,
            'icd9': icd9_col,
            'desc': desc_col,
            'unrecognised_icd9': unrecognised_icd9,
            'conv_file': cls._SCHEME_FILE
        }

    @staticmethod
    def register_mappings(flatccs_scheme, icd9_scheme):
        res = flatccs_scheme.flatccs_columns(icd9_scheme)

        flatccs2icd9 = _CodeMapper(
            flatccs_scheme,
            icd9_scheme,
            t_dag_space=False,
            unrecognised_target=res["unrecognised_icd9"],
            conv_file=res["conv_file"])
        icd92flatccs = _CodeMapper(
            icd9_scheme,
            flatccs_scheme,
            t_dag_space=False,
            unrecognised_source=res["unrecognised_icd9"],
            conv_file=res["conv_file"])

        map_n1 = dict(zip(res['icd9'], res['code']))
        assert len(map_n1) == len(res['icd9']), "1toN mapping expected"

        for icode, ccode in map_n1.items():
            flatccs2icd9[ccode].add(icode)
            icd92flatccs[icode].add(ccode)


class DxFlatCCS(Singleton, FlatCCSCommons):

    _SCHEME_FILE = '$dxref 2015.csv.gz'

    def __init__(self):
        cols = self.flatccs_columns(DxICD9())
        codes = sorted(set(cols['code']))
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=dict(zip(cols['code'], cols['desc'])),
                         name='dx_flatccs')


class PrFlatCCS(Singleton, FlatCCSCommons):
    _SCHEME_FILE = '$prref 2015.csv.gz'

    def __init__(self):
        cols = self.flatccs_columns(PrICD9())
        codes = sorted(set(cols['code']))
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=dict(zip(cols['code'], cols['desc'])),
                         name='pr_flatccs')


class DxLTC212FlatCodes(Singleton, AbstractScheme):
    _SCHEME_FILE = os.path.join(_RSC_DIR, 'CPRD_212_LTC_ALL.csv.gz')

    def __init__(self):
        df = pd.read_csv(self._SCHEME_FILE, dtype=str)

        medcode_cname = 'medcodeid'
        disease_num_cname = 'disease_num'
        disease_cname = 'disease'

        system_cname = 'system'
        system_num_cname = 'system_num'

        desc = dict()
        system = dict()
        medcodes = dict()
        for disease_num, disease_df in df.groupby(disease_num_cname):
            disease_set = set(disease_df[disease_cname])
            assert len(disease_set) == 1, "Disease name should be unique"
            (disease_name, ) = disease_set

            system_set = set(disease_df[system_cname])
            system_num_set = set(disease_df[system_num_cname])
            assert len(system_set) == 1, "System name should be unique"
            assert len(system_num_set) == 1, "System num should be unique"

            (system_name, ) = system_set
            (system_num, ) = system_num_set

            medcodes_list = sorted(set(disease_df[medcode_cname]))

            desc[disease_num] = disease_name
            system[disease_num] = system_num
            medcodes[disease_num] = medcodes_list

        codes = sorted(set(df[disease_num_cname]))

        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='dx_cprd_ltc212')
        self._system = system
        self._medcodes = medcodes

    @property
    def medcodes(self):
        return self._medcodes

    @property
    def system(self):
        return self._system


class DxLTC9809FlatMedcodes(Singleton, AbstractScheme):
    _SCHEME_FILE = 'CPRD_212_LTC_ALL.csv.gz'

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, self._SCHEME_FILE)
        df = pd.read_csv(filepath, dtype=str)

        medcode_cname = 'medcodeid'
        disease_num_cname = 'disease_num'
        disease_cname = 'disease'
        desc_cname = 'descr'

        system_cname = 'system'
        system_num_cname = 'system_num'

        desc = dict()
        systems = dict()
        diseases = dict()
        for medcodeid, medcode_df in df.groupby(medcode_cname):
            disease_num_list = sorted(set(medcode_df[disease_num_cname]))
            system_num_list = sorted(set(medcode_df[system_num_cname]))

            disease_list = sorted(set(medcode_df[disease_cname]))
            system_list = sorted(set(medcode_df[system_cname]))

            desc[medcodeid] = str(medcode_df[desc_cname])
            systems[medcodeid] = system_num_list
            diseases[medcodeid] = disease_num_list

        codes = sorted(set(df[medcode_cname]))

        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='dx_cprd_ltc9809')

        self._systems = systems
        self._diseases = diseases

    @property
    def systems(self):
        return self._systems

    @property
    def diseases(self):
        return self._diseases


class EthCPRD(AbstractScheme):
    _SCHEME_FILE = 'cprd_eth.csv'
    NAME = None
    ETH_CODE_CNAME = None
    ETH_DESC_CNAME = None

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, self._SCHEME_FILE)
        df = pd.read_csv(filepath, dtype=str)
        desc = dict()
        for eth_code, eth_df in df.groupby(self.ETH_CODE_CNAME):
            eth_set = set(eth_df[self.ETH_DESC_CNAME])
            assert len(eth_set) == 1, "Ethnicity description should be unique"
            (eth_desc, ) = eth_set
            desc[eth_code] = eth_desc

        codes = sorted(set(df[self.ETH_CODE_CNAME]))

        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name=self.NAME)


class EthCPRD16(Singleton, EthCPRD):
    NAME = 'eth_cprd_16'
    ETH_CODE_CNAME = 'eth16'
    ETH_DESC_CNAME = 'eth16_desc'


class EthCPRD5(Singleton, EthCPRD):
    NAME = 'eth_cprd_5'
    ETH_CODE_CNAME = 'eth5'
    ETH_DESC_CNAME = 'eth5_desc'


class MIMICEth(AbstractScheme):
    _SCHEME_FILE = 'mimic4_race_grouper.csv.gz'
    NAME = 'mimic4_eth32'
    ETH_CNAME = 'eth32'

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, self._SCHEME_FILE)
        df = pd.read_csv(filepath, dtype=str)
        desc = dict()
        for eth_code, eth_df in df.groupby(self.ETH_CNAME):
            eth_set = set(eth_df[self.ETH_CNAME])
            assert len(eth_set) == 1, "Ethnicity description should be unique"
            (eth_desc, ) = eth_set
            desc[eth_code] = eth_code

        codes = sorted(set(df[self.ETH_CNAME]))

        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name=self.NAME)


class MIMICEth32(Singleton, MIMICEth):
    NAME = 'mimic4_eth32'
    ETH_CNAME = 'eth32'


class MIMICEth5(Singleton, MIMICEth):
    NAME = 'mimic4_eth5'
    ETH_CNAME = 'eth5'


def register_mimic4_eth_mapping(s_scheme: MIMICEth32, t_scheme: MIMICEth5):
    filepath = os.path.join(_RSC_DIR, s_scheme._SCHEME_FILE)
    df = pd.read_csv(filepath, dtype=str)
    m = _CodeMapper(s_scheme, t_scheme, t_dag_space=False)
    for eth32, eth_df in df.groupby(s_scheme.ETH_CNAME):
        m[eth32] = set(eth_df[t_scheme.ETH_CNAME])


class MIMICProcedures(Singleton, AbstractScheme):

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
        df = pd.read_csv(filepath, dtype=str)
        df = df[df.group != 'exclude']
        df = df.sort_values(['group', 'label'])
        codes = df.code.tolist()
        labels = df.label.tolist()
        desc = dict(zip(codes, labels))
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='int_mimic4_proc')


class AbstractGroupedProcedures(AbstractScheme):

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


class MIMICProcedureGroups(Singleton, AbstractGroupedProcedures):

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
        df = pd.read_csv(filepath, dtype=str)
        df = df[df.group != 'exclude']
        df = df.sort_values(['group', 'label'])
        codes = df.group.unique().tolist()
        desc = dict(zip(codes, codes))

        groups = {
            group: set(group_df['code'])
            for group, group_df in df.groupby('group')
        }
        aggregation_groups = {'or': set(codes)}

        super().__init__(groups=groups,
                         aggregation=['or'],
                         aggregation_groups=aggregation_groups,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='int_mimic4_grouped_proc')


class MIMICInputGroups(Singleton, AbstractGroupedProcedures):
    """
    InterventionGroup class encapsulates the similar interventions.
    """

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
        df = pd.read_csv(filepath, dtype=str)
        df = df[df.group_decision != 'E']
        df = df.sort_values(by=['group_decision', 'group', 'label'])
        codes = df.group.unique().tolist()
        desc = dict(zip(codes, codes))

        aggs = df.group_decision.unique().tolist()

        self._dose_impact = dict()
        aggregation_groups = dict()
        groups = dict()
        for agg, agg_df in df.groupby('group_decision'):
            aggregation_groups[agg] = set(agg_df['group'])
            for group, group_df in agg_df.groupby('group'):
                assert group not in groups, "Group should be unique"
                groups[group] = set(group_df['label'])
                self._dose_impact[group] = group_df['dose_impact'].iloc[0]

        super().__init__(groups=groups,
                         aggregation=aggs,
                         aggregation_groups=aggregation_groups,
                         codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='int_mimic4_input_group')

    @property
    def dose_impact(self):
        return self._dose_impact


class MIMICInput(Singleton, AbstractScheme):
    """
    InterventionGroup class encapsulates the similar interventions.
    """

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
        df = pd.read_csv(filepath, dtype=str)
        df = df[df.group_decision != 'E']
        df = df.sort_values(by=['group_decision', 'group', 'label'])
        codes = df.label.unique().tolist()
        desc = dict(zip(codes, codes))

        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='int_mimic4_input')


class MIMICObservables(Singleton, AbstractScheme):

    def __init__(self):
        filepath = os.path.join(_RSC_DIR, 'mimic4_obs_codes.csv.gz')
        df = pd.read_csv(filepath, dtype=str)
        codes = df.code.tolist()
        desc = dict(zip(codes, df.label.tolist()))
        self._groups = dict(zip(codes, df.group.tolist()))
        super().__init__(codes=codes,
                         index=dict(zip(codes, range(len(codes)))),
                         desc=desc,
                         name='int_mimic4_obs')

    @property
    def groups(self):
        return self._groups


class Gender(Singleton, BinaryScheme):

    def __init__(self):
        codes = ['M', 'F']
        index = {'M': 0, 'F': 1}
        desc = {'M': 'male', 'F': 'female'}
        name = 'gender'
        super().__init__(codes=codes, index=index, desc=desc, name=name)


def register_mimic4proc_mapping(s_scheme: MIMICProcedures,
                                t_scheme: MIMICProcedureGroups):
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_proc.csv.gz')
    df = pd.read_csv(filepath, dtype=str)

    m = _CodeMapper(s_scheme, t_scheme, t_dag_space=False)
    for group, group_df in df.groupby('group'):
        m.update({c: group for c in group_df.code})


def register_mimic4input_mapping(s_scheme: MIMICInput,
                                 t_schame: MIMICInputGroups):
    filepath = os.path.join(_RSC_DIR, 'mimic4_int_grouper_input.csv.gz')
    df = pd.read_csv(filepath, dtype=str)

    m = _CodeMapper(s_scheme, t_schame, t_dag_space=False)
    for group, group_df in df.groupby('group'):
        m.update({c: group for c in group_df.label})


def register_cprd_eth_mapping(s_scheme: EthCPRD16, t_scheme: EthCPRD5):
    filepath = os.path.join(_RSC_DIR, s_scheme._SCHEME_FILE)
    df = pd.read_csv(filepath, dtype=str)
    m = _CodeMapper(s_scheme, t_scheme, t_dag_space=False)
    for eth16, eth_df in df.groupby(s_scheme.ETH_CODE_CNAME):
        m[eth16] = set(eth_df[t_scheme.ETH_CODE_CNAME])


def register_medcode_mapping(s_scheme: DxLTC9809FlatMedcodes,
                             t_scheme: DxLTC212FlatCodes):
    m = _CodeMapper(s_scheme, t_scheme, t_dag_space=False)
    for medcodeid, disease_nums in s_scheme.diseases.items():
        m[medcodeid] = set(disease_nums)


def register_chained_map(s_scheme, t_scheme, inter_scheme):
    inter_mapping = _CodeMapper.get_mapper(s_scheme, inter_scheme)
    assert inter_mapping.t_dag_space == False
    s_codes = set(s_scheme.codes) & set(inter_mapping)
    m = _CodeMapper(s_scheme, t_scheme, t_dag_space=False)
    m.update({c: inter_mapping[c] for c in s_codes})


def reg_dx_icd9_chained_map(s_scheme, t_scheme):
    return register_chained_map(s_scheme, t_scheme, DxICD9())


def reg_pr_icd9_chained_map(s_scheme, t_scheme):
    return register_chained_map(s_scheme, t_scheme, PrICD9())


# Possible Mappings, Lazy-loaded maps.
load_maps = {}

# ICD9 <-> ICD10s
load_maps.update({
    (DxICD10, DxICD9):
    lambda: ICDCommons.register_mappings(DxICD10(), DxICD9(),
                                         '2018_gem_cm_I10I9.txt.gz'),
    (DxICD9, DxICD10):
    lambda: ICDCommons.register_mappings(DxICD9(), DxICD10(),
                                         '2018_gem_cm_I9I10.txt.gz'),
    (PrICD10, PrICD9):
    lambda: ICDCommons.register_mappings(PrICD10(), PrICD9(),
                                         '2018_gem_pcs_I10I9.txt.gz'),
    (PrICD9, PrICD10):
    lambda: ICDCommons.register_mappings(PrICD9(), PrICD10(),
                                         '2018_gem_pcs_I9I10.txt.gz')
})

# ICD9 <-> CCS
load_maps.update({
    (DxCCS, DxICD9):
    lambda: CCSCommons.register_mappings(DxCCS(), DxICD9()),
    (DxICD9, DxCCS):
    lambda: CCSCommons.register_mappings(DxCCS(), DxICD9()),
    (PrCCS, PrICD9):
    lambda: CCSCommons.register_mappings(PrCCS(), PrICD9()),
    (PrICD9, PrCCS):
    lambda: CCSCommons.register_mappings(PrCCS(), PrICD9()),
    (DxFlatCCS, DxICD9):
    lambda: FlatCCSCommons.register_mappings(DxFlatCCS(), DxICD9()),
    (DxICD9, DxFlatCCS):
    lambda: FlatCCSCommons.register_mappings(DxFlatCCS(), DxICD9()),
    (PrFlatCCS, PrICD9):
    lambda: FlatCCSCommons.register_mappings(PrFlatCCS(), PrICD9()),
    (PrICD9, PrFlatCCS):
    lambda: FlatCCSCommons.register_mappings(PrFlatCCS(), PrICD9()),
})

# ICD10 <-> CCS (Through ICD9 as an intermediate scheme)
load_maps.update({
    (DxCCS, DxICD10):
    lambda: reg_dx_icd9_chained_map(DxCCS(), DxICD10()),
    (DxICD10, DxCCS):
    lambda: reg_dx_icd9_chained_map(DxICD10(), DxCCS()),
    (PrCCS, PrICD10):
    lambda: reg_pr_icd9_chained_map(PrCCS(), PrICD10()),
    (PrICD10, PrCCS):
    lambda: reg_pr_icd9_chained_map(PrICD10(), PrCCS()),
    (DxFlatCCS, DxICD10):
    lambda: reg_dx_icd9_chained_map(DxFlatCCS(), DxICD10()),
    (DxICD10, DxFlatCCS):
    lambda: reg_dx_icd9_chained_map(DxICD10(), DxFlatCCS()),
    (PrFlatCCS, PrICD10):
    lambda: reg_pr_icd9_chained_map(PrFlatCCS(), PrICD10()),
    (PrICD10, PrFlatCCS):
    lambda: reg_pr_icd9_chained_map(PrICD10(), PrFlatCCS())
})

# CPRD conversions
# LTC9809 -> LTC212
# Eth16 -> Eth5

load_maps.update({
    (DxLTC9809FlatMedcodes, DxLTC212FlatCodes):
    lambda: register_medcode_mapping(DxLTC9809FlatMedcodes(),
                                     DxLTC212FlatCodes()),
    (EthCPRD16, EthCPRD5):
    lambda: register_cprd_eth_mapping(EthCPRD16(), EthCPRD5())
})

# MIMIC Inpatient conversions
load_maps.update({
    (MIMICProcedures, MIMICProcedureGroups):
    lambda: register_mimic4proc_mapping(MIMICProcedures(),
                                        MIMICProcedureGroups()),
    (MIMICInput, MIMICInputGroups):
    lambda: register_mimic4input_mapping(MIMICInput(), MIMICInputGroups()),
    (MIMICEth32, MIMICEth5):
    lambda: register_mimic4_eth_mapping(MIMICEth32(), MIMICEth5())
})

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


class OutcomeExtractor(AbstractScheme):

    def __init__(self, outcome_space='dx_flatccs_filter_v1'):
        conf = self.conf_from_json(outcome_conf_files[outcome_space])

        self._t_scheme = scheme_from_classname(conf['code_scheme'])
        codes = [
            c for c in sorted(self.t_scheme.index)
            if c not in conf['exclude_codes']
        ]

        index = dict(zip(codes, range(len(codes))))
        desc = {c: self.t_scheme.desc[c] for c in codes}
        super().__init__(codes=codes,
                         index=index,
                         desc=desc,
                         name=outcome_space)

    @property
    def t_scheme(self):
        return self._t_scheme

    @property
    def outcome_dim(self):
        return len(self.index)

    def map_codeset(self, codeset: Set[str], s_scheme: AbstractScheme):
        m = s_scheme.mapper_to(self._t_scheme)
        codeset = m.map_codeset(codeset)

        if m.t_dag_space:
            codeset &= set(m.t_scheme.dag2code)
            codeset = set(m.t_scheme.dag2code[c] for c in codeset)

        return codeset & set(self.codes)

    def codeset2vec(self, codeset: Set[str], s_scheme: AbstractScheme):
        vec = np.zeros(len(self.index), dtype=bool)
        for c in self.map_codeset(codeset, s_scheme):
            vec[self.index[c]] = True
        return CodesVector(np.array(vec), self)

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
            t_scheme = scheme_from_classname(conf['code_scheme'])
            conf['exclude_codes'] = [
                c for c in t_scheme.codes if c not in conf['selected_codes']
            ]
            return conf
        elif 'exclude_codes' in conf:
            return conf


def scheme_from_classname(classname):
    return eval(classname)()
