"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from __future__ import annotations

import logging
import math
import os
import re
from abc import abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from dataclasses import field
from functools import cached_property
from types import MappingProxyType
from typing import Set, Dict, Type, Optional, List, Union, ClassVar, Tuple, Any, Literal, ItemsView, Iterator, \
    Iterable, Mapping, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import tables as tb

from ..base import VxData, VxDataView
from ..utils import load_config

NumericalTypeHint = Literal['B', 'N', 'O', 'C']  # Binary, Numerical, Ordinal, Categorical


def resources_dir(*subdir) -> str:
    return os.path.join(os.path.dirname(__file__), "resources", *subdir)


class FrozenDict11(VxData):
    data: MappingProxyType[str, Union[str, int]]

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "FrozenDict11":
        return FrozenDict11(MappingProxyType(d))

    def equals(self, other: "FrozenDict11") -> bool:
        return self.data == other.data

    def __getitem__(self, key: str) -> Union[str, int]:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: str = None) -> Union[str, int]:
        return self.data.get(key, default)

    def items(self) -> ItemsView[str, Union[str, int]]:
        return self.data.items()

    def keys(self) -> Iterable[str]:
        return self.data.keys()

    def values(self) -> Iterable[Union[str, int]]:
        return self.data.values()

    def to_dataframe(self):
        return pd.DataFrame(self.data.items(), columns=['key', 'value'])

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "FrozenDict11":
        return FrozenDict11.from_dict(df.set_index('key')['value'].to_dict())

    def to_hdf_group(self, group: tb.Group):
        h5file = group._v_file
        h5file.create_array(group, 'classname', obj=self.__class__.__name__.encode('utf-8'))
        df = self.to_dataframe()
        df.to_hdf(group._v_file.filename, key=f'{group._v_pathname}/data')

    @classmethod
    def _from_hdf_group(cls, group: tb.Group) -> 'VxData':
        df = pd.read_hdf(group._v_file.filename, key=f'{group._v_pathname}/data')
        return cls.from_dataframe(df)


class FrozenDict1N(FrozenDict11):
    data: MappingProxyType[str, Set[str]]

    @staticmethod
    def from_dict(d: Dict[str, Set[str]]) -> "FrozenDict1N":
        return FrozenDict1N(MappingProxyType(d))

    def __getitem__(self, key: str) -> Set[str]:
        return self.data[key]

    def items(self) -> ItemsView[str, Set[str]]:
        return self.data.items()

    def get(self, key: str, default: Set[str] = None) -> Set[str]:
        return self.data.get(key, default)

    def to_dataframe(self):
        return pd.DataFrame([(k, item) for k, v in self.data.items() for item in v], columns=['key', 'value'])

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "FrozenDict1N":
        return FrozenDict1N.from_dict(df.groupby('key')['value'].apply(set).to_dict())


class CodesVector(VxData):
    """
    Represents a multi-hot vector encoding of codes using a specific coding scheme.

    Attributes:
        vec (np.ndarray): the vector of codes.
        scheme (str): the coding scheme.
    """

    vec: npt.NDArray[bool] | np.ndarray[np.bool_]
    scheme: str

    @classmethod
    def empty_like(cls, other: CodesVector) -> CodesVector:
        """
        Creates an empty CodesVector with the same shape as the given CodesVector.

        Args:
            other (CodesVector): the CodesVector to mimic.

        Returns:
            CodesVector: the empty CodesVector.
        """
        return cls(np.zeros_like(other.vec), other.scheme)

    def to_codeset(self, manager: Union[CodingSchemesManager, SchemeManagerView]) -> Set[str]:
        """
        Converts the binary codes vector to a set of one code.

        Returns:
            set: a set containing one code.
        """
        scheme = manager.scheme[self.scheme]
        return {scheme.index2code[i] for i in np.flatnonzero(self.vec)}

    def __len__(self) -> int:
        """
        Returns the length of the vector.

        Returns:
            int: the length of the vector.
        """
        return len(self.vec)

    def union(self, other: CodesVector) -> CodesVector:
        """
        Returns the union of the current CodesVector with another CodesVector.

        Args:
            other (CodesVector): the other CodesVector to union with.

        Returns:
            CodesVector: the union of the two CodesVectors.
        """
        assert self.scheme == other.scheme, "Schemes should be the same."
        assert self.vec.dtype == bool and other.vec.dtype == bool, "Vector types should be the same."
        return CodesVector(self.vec | other.vec, self.scheme)


class CodingScheme(VxData):
    name: str = field(kw_only=True)
    codes: Tuple[str, ...] = field(kw_only=True)
    desc: FrozenDict11 = field(kw_only=True)

    # Possible Schemes, Lazy-loaded schemes.
    # _load_schemes: ClassVar[Dict[str, Callable]] = {}
    # _schemes: ClassVar[Dict[str, Union["CodingScheme", Any]]] = {}
    # vector representation class
    vector_cls: ClassVar[Type[CodesVector]] = CodesVector

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, codes({len(self.codes)}), desc({len(self.desc)}))"

    def __post_init__(self):
        self._check_types()

        # Check types.
        assert isinstance(self.name, str), "Scheme name must be a string."
        assert isinstance(self.codes, tuple), "Scheme codes must be a tuple."
        assert isinstance(self.desc, FrozenDict11), "Scheme description must be a FrozenDict11."

        assert tuple(sorted(self.codes)) == self.codes, "Scheme codes must be sorted."

        # Check for uniqueness of codes.
        assert len(self.codes) == len(set(self.codes)), f"{self}: Codes should be unique."

        # Check sizes.
        assert len(self.codes) == len(self.desc), f"{self}: Codes and descriptions should have the same size."

    def _check_types(self):
        for collection in [self.codes, self.desc]:
            assert all(
                isinstance(c, str) for c in collection
            ), f"{self}: All name types should be str."

        assert all(
            isinstance(desc, str)
            for desc in self.desc.values()

        ), f"{self}: All desc types should be str."

    @cached_property
    def index(self) -> Dict[str, int]:
        return {code: idx for idx, code in enumerate(self.codes)}

    @cached_property
    def index2code(self) -> Dict[int, str]:
        return {idx: code for code, idx in self.index.items()}

    @cached_property
    def index2desc(self) -> Dict[int, str]:
        return {self.index[code]: _desc for code, _desc in self.desc.items()}

    def __eq__(self, other):
        """
        Check if the current scheme is equal to another scheme.
        """
        return self.equals(other)

    def __len__(self) -> int:
        """
        Returns the number of codes in the current scheme.
        """
        return len(self.codes)

    def __bool__(self) -> bool:
        """
        Returns True if the current scheme is not empty.
        """
        return len(self.codes) > 0

    def __str__(self) -> str:
        """
        Returns the name of the current scheme.
        """
        return self.name

    def __contains__(self, code: str) -> bool:
        """Returns True if `code` is contained in the current scheme."""
        return code in self.codes

    def search_regex(self, query: str, regex_flags: int = re.I) -> Set[str]:
        """
        A regex-supported search of codes by a `query` string. the search is applied on the code description.
        For example, you can use it to return all codes related to cancer by setting the
        `query = 'cancer'` and `regex_flags = re.i` (for case-insensitive search).

        Args:
            query (str): the query string.
            regex_flags (int): the regex flags.

        Returns:
            Set[str]: the set of codes matching the query.
        """
        return set(filter(lambda c: re.findall(query, self.desc[c], flags=regex_flags), self.codes))

    def wrap_vector(self, vec: np.ndarray) -> "CodingScheme.vector_cls":
        """
        Wrap a numpy array as a vector representation of the current scheme.
        Args:
            vec (np.ndarray): the numpy array to wrap.
        Returns:
            CodingScheme.vector_cls: a vector representation of the current scheme.
        """
        assert len(vec) == len(self), f"Vector length should be {len(self)}."
        assert vec.ndim == 1, f"Vector should be 1-dimensional."

        return CodingScheme.vector_cls(vec, self.name)

    def codeset2vec(self, codeset: Set[str]) -> "CodingScheme.vector_cls":
        """
        Convert a codeset to a vector representation.
        Args:
            codeset (Set[str]): the codeset to convert.
        Returns:
            CodingScheme.vector_cls: a vector representation of the current scheme.
        """
        vec = np.zeros(len(self), dtype=bool)
        try:
            for c in codeset:
                vec[self.index[c]] = True
        except KeyError as missing:
            logging.error(
                f"Code {missing} is missing." f"Accepted keys: {self.index.keys()}"
            )

        return CodingScheme.vector_cls(vec, self.name)

    def supported_targets(self, scheme_manager: CodingSchemesManager) -> Tuple[str, ...]:
        return tuple(t for s, t in scheme_manager.map.keys() if s == self.name)

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns the scheme as a Pandas DataFrame.
        The DataFrame contains the following columns:
            - code: the code string
            - desc: the code description
        """

        index = list(range(len(self)))
        return pd.DataFrame(
            {
                "code": self.index2code,
                "desc": self.index2desc,
            },
            index=index,
        )


class NumericScheme(CodingScheme):
    """
    NumericScheme is a subclass of FlatScheme that represents a numerical coding scheme.
    Additional to `FlatScheme` attributes, it contains the following attributes to represent the coding scheme:
    - type_hint: Dict mapping codes to their type hint (B: binary, N: numerical, O: ordinal, C: categorical)
    """

    type_hint: Optional[Dict[str, NumericalTypeHint]] = None
    default_type_hint: Literal['N', 'C', 'B', 'O'] = 'N'

    def __post_init__(self):
        super().__post_init__()
        if self.type_hint is None:
            self.type_hint = {code: self.default_type_hint for code in self.codes}
        assert set(self.codes) == set(self.type_hint.keys()), \
            f"The set of codes ({self.codes}) does not match the set of type hints ({self.type_hint.keys()})."
        assert set(self.type_hint.values()) <= {'B', 'N', 'O', 'C'}, \
            f"The set of type hints ({self.type_hint.values()}) contains invalid values."

    @cached_property
    def type_array(self) -> npt.NDArray[NumericalTypeHint]:
        """
        Returns the type hint of the codes in the scheme as a numpy array.
        """
        assert set(self.index[c] for c in self.codes) == set(range(len(self))), \
            f"The order of codes ({self.codes}) does not match the order of type hints ({self.type_hint.keys()})."
        return np.array([self.type_hint[code] for code in self.codes])


class CodesVectorWithMissing(CodesVector):
    """
    A subclass of CodesVector that represents a vector of codes with the ability to handle missing values.
    """

    def to_codeset(self, manager: Union[CodingSchemesManager, SchemeManagerView]) -> Set[str]:
        """
        Convert the codes vector to a set of codes.

        Returns:
            set: a set of codes represented by the non-zero elements in the vector.
        """
        index = self.vec.nonzero()[0]
        if len(index) == 0:
            scheme: SchemeWithMissing = manager.scheme[self.scheme]
            return {scheme.missing_code}


class SchemeWithMissing(CodingScheme):
    """
    A coding scheme that represents categorical schemes and supports missing/unkown values.

    This class extends the `FlatScheme` class and adds support for a missing code.
    It provides methods to convert a set of codes to a multi-hot vector representation,
    where each element in the vector represents the presence or absence of a code.

    Attributes:
        _missing_code (str): the code that represents a missing value in the coding scheme.
    """

    missing_code: str = field(kw_only=True)
    vector_cls: ClassVar[Type[CodesVectorWithMissing]] = CodesVectorWithMissing

    def __post_init__(self):
        super().__post_init__()
        assert self.missing_code not in self.codes, f"{self}: Missing code should not be in the list of codes."
        self.codes = self.codes + (self.missing_code,)
        self.desc = FrozenDict11.from_dict({k: v for k, v in self.desc.items()} | {self.missing_code: "Missing"})
        self._check_index_integrity()

    def __len__(self) -> int:
        return len(self.codes) - 1

    @cached_property
    def index(self) -> Dict[str, int]:
        return {code: idx for idx, code in enumerate(self.codes)} | {self.missing_code: -1}

    def _check_index_integrity(self):
        for code, idx in self.index.items():
            if code == self.missing_code:
                continue
            assert idx == self.codes.index(
                code), f"{self}: Index of {code} is not consistent with its position in the list."


class HierarchicalScheme(CodingScheme):
    """
    A class representing a hierarchical coding scheme.

    This class extends the functionality of the FlatScheme class and provides
    additional methods for working with hierarchical coding schemes.
    """
    ch2pt: FrozenDict1N = field(kw_only=True)
    dag_codes: Optional[Tuple[str, ...]] = None
    dag_desc: Optional[FrozenDict11] = None
    code2dag: Optional[FrozenDict11] = None

    def __post_init__(self):
        super().__post_init__()

        self.dag_codes = self.dag_codes or self.codes
        self.dag_desc = self.dag_desc or self.desc
        self.code2dag = self.code2dag or FrozenDict11.from_dict({c: c for c in self.codes})

        # Check types
        assert isinstance(self.dag_codes, tuple), f"{self}: codes should be a list."
        assert isinstance(self.dag_desc, FrozenDict11), f"{self}: desc should be a dict."
        assert isinstance(self.code2dag, FrozenDict11), f"{self}: code2dag should be a dict."
        assert isinstance(self.ch2pt, FrozenDict1N), f"{self}: ch2pt should be a dict."
        for collection in [self.dag_codes, self.dag_desc.values(), self.dag_desc.keys(), self.code2dag.keys(),
                           self.dag_desc.values(), self.code2dag.values(), self.ch2pt.keys(),
                           set.union(*self.ch2pt.values())]:
            assert all(
                isinstance(c, str) for c in collection
            ), f"{self}: All name types should be str."

        # Check sizes
        # TODO: note in the documentation that dag2code size can be less than the dag_codes since some dag_codes are internal nodes that themselves are not are not complete clinical concepts.
        for collection in [self.dag_codes, self.dag_desc]:
            assert len(collection) == len(self.dag_codes), f"{self}: All collections should have the same size."

    def make_ancestors_mat(self, include_itself: bool = True) -> npt.NDArray[np.bool_]:
        """
        Creates a matrix representing the ancestors of each code in the hierarchy.

        Args:
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            np.ndarray: a boolean matrix where each element (i, j) is True if code i is an ancestor of code j, and False otherwise.
        """
        ancestors_mat = np.zeros((len(self.dag_index), len(self.dag_index)),
                                 dtype=bool)
        for code_i, i in self.dag_index.items():
            for ancestor_j in self.code_ancestors_bfs(code_i, include_itself):
                j = self.dag_index[ancestor_j]
                ancestors_mat[i, j] = 1

        return ancestors_mat

    @cached_property
    def dag_index(self) -> Dict[str, int]:
        """
        Dict[str, int]: a dictionary mapping codes to their indices in the hierarchy.
        """
        return {c: i for i, c in enumerate(self.dag_codes)}

    @cached_property
    def dag2code(self) -> Dict[str, str]:
        """
        Dict[str, str]: a dictionary mapping codes in the hierarchy to their corresponding codes.
        """
        return {d: c for c, d in self.code2dag.items()}

    @cached_property
    def pt2ch(self) -> FrozenDict1N:
        return self.reverse_connection(self.ch2pt)

    def __contains__(self, code: str) -> bool:
        """
        Checks if a code is contained in the current hierarchy.

        Args:
            code (str): the code to check.

        Returns:
            bool: true if the code is contained in the hierarchy, False otherwise.
        """
        return code in self.dag_codes or code in self.codes

    @staticmethod
    def reverse_connection(connection: Mapping[str, Set[str]]) -> FrozenDict1N:
        """
        Reverses a connection dictionary.

        Args:
            connection (Dict[str, Set[str]]): the connection dictionary to reverse.

        Returns:
            Dict[str, Set[str]]: the reversed connection dictionary.
        """
        rev_connection = defaultdict(set)
        for node, conns in connection.items():
            for conn in conns:
                rev_connection[conn].add(node)
        return FrozenDict1N.from_dict(rev_connection)

    @staticmethod
    def _bfs_traversal(connection: FrozenDict1N, code: str, include_itself: bool) -> List[str]:
        """
        Performs a breadth-first traversal of the hierarchy.

        Args:
            connection (Dict[str, Set[str]]): the connection dictionary representing the hierarchy.
            code (str): the starting code for the traversal.
            include_itself (bool): whether to include the starting code in the traversal.

        Returns:
            List[str]: a list of codes visited during the traversal.
        """
        result = OrderedDict()
        q = [code]

        while len(q) != 0:
            # remove the first element from the stack
            current_code = q.pop(0)
            current_connections = connection.get(current_code, set())
            q.extend([c for c in current_connections if c not in result])
            if current_code not in result:
                result[current_code] = 1

        if not include_itself:
            del result[code]
        return list(result.keys())

    @staticmethod
    def _dfs_traversal(connection: FrozenDict1N, code: str, include_itself: bool) -> List[str]:
        """
        Performs a depth-first traversal of the hierarchy.

        Args:
            connection (Dict[str, Set[str]]): the connection dictionary representing the hierarchy.
            code (str): the starting code for the traversal.
            include_itself (bool): whether to include the starting code in the traversal.

        Returns:
            List[str]: A list of codes visited during the traversal.
        """
        result = {code} if include_itself else set()

        def _traversal(_node):
            for conn in connection.get(_node, set()):
                result.add(conn)
                _traversal(conn)

        _traversal(code)

        return list(result)

    @staticmethod
    def _dfs_edges(connection: FrozenDict1N, code: str) -> Set[Tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal.

        Args:
            connection (Dict[str, Set[str]]): the connection dictionary representing the hierarchy.
            code (str): the starting code for the traversal.

        Returns:
            Set[Tuple[str, str]]: a set of edges in the hierarchy.
        """
        result = set()

        def _edges(_node):
            for conn in connection.get(_node, set()):
                result.add((_node, conn))
                _edges(conn)

        _edges(code)
        return result

    def code_ancestors_bfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the ancestors of a code in the hierarchy using breadth-first traversal.

        Args:
            code (str): the code for which to find the ancestors.
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            List[str]: A list of ancestor codes.
        """
        return self._bfs_traversal(self.ch2pt, code, include_itself)

    def code_ancestors_dfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the ancestors of a code in the hierarchy using depth-first traversal.

        Args:
            code (str): the code for which to find the ancestors.
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            List[str]: a list of ancestor codes.
        """
        return self._dfs_traversal(self.ch2pt, code, include_itself)

    def code_successors_bfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the successors of a code in the hierarchy using breadth-first traversal.

        Args:
            code (str): the code for which to find the successors.
            include_itself (bool): whether to include the code itself as its own successor. Defaults to True.

        Returns:
            List[str]: A list of successor codes.
        """
        return self._bfs_traversal(self.pt2ch, code, include_itself)

    def code_successors_dfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the successors of a code in the hierarchy using depth-first traversal.

        Args:
            code (str): the code for which to find the successors.
            include_itself (bool): whether to include the code itself as its own successor. Defaults to True.

        Returns:
            List[str]: a list of successor codes.
        """
        return self._dfs_traversal(self.pt2ch, code, include_itself)

    def ancestors_edges_dfs(self, code: str) -> Set[Tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal of ancestors.

        Args:
            code (str): the code for which to find the ancestor edges.

        Returns:
            Set[Tuple[str, str]]: a set of edges in the hierarchy.
        """
        return self._dfs_edges(self.ch2pt, code)

    def successors_edges_dfs(self, code: str) -> Set[Tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal of successors.

        Args:
            code (str): the code for which to find the successor edges.

        Returns:
            Set[Tuple[str, str]]: a set of edges in the hierarchy.
        """
        return self._dfs_edges(self.pt2ch, code)

    def least_common_ancestor(self, codes: List[str]) -> str:
        """
        Finds the least common ancestor of a list of codes in the hierarchy.

        Args:
            codes (List[str]): the list of codes for which to find the least common ancestor.

        Returns:
            str: the least common ancestor code.
        
        Raises:
            RuntimeError: if a common ancestor is not found.
        """
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

    def search_regex(self, query: str, regex_flags: int = re.I) -> Set[str]:
        """
        A regex-based search of codes by a `query` string. the search is \
            applied on the code descriptions. for example, you can use it \
            to return all codes related to cancer by setting the \
            `query = 'cancer'` and `regex_flags = re.i` \
            (for case-insensitive search). For all found codes, \
            their successor codes are also returned in the resutls.

        Args:
            query (str): The regex query string.
            regex_flags (int): The flags to use for the regex search. Defaults to re.I (case-insensitive).

        Returns:
            Set[str]: A set of codes that match the regex query, including their successor codes.
        """
        codes = filter(
            lambda c: re.findall(query, self.desc[c], flags=regex_flags),
            self.codes)

        dag_codes = filter(
            lambda c: re.findall(query, self.dag_desc[c], flags=regex_flags),
            self.dag_codes)

        all_codes = set(map(self.code2dag.get, codes)) | set(dag_codes)

        for c in list(all_codes):
            all_codes.update(self.code_successors_dfs(c, include_itself=False))

        return all_codes


class Ethnicity(CodingScheme):
    pass


class UnsupportedMapping(ValueError):
    pass


class CodeMap(VxData):
    source_name: str = field(kw_only=True)
    target_name: str = field(kw_only=True)
    data: FrozenDict1N = field(kw_only=True)

    def __post_init__(self):
        assert isinstance(self.data, FrozenDict1N), "Data should be a FrozenDict1N."

    def mapped_to_dag_space(self, scheme_manager: CodingSchemesManager) -> bool:
        """
        Returns True if the CodeMap is mapped to DAG space, False otherwise.

        Returns:
            bool: True if the CodeMap is mapped to DAG space, False otherwise.
        """
        scheme = scheme_manager.scheme[self.target_name]
        if not isinstance(scheme, HierarchicalScheme) or scheme.dag_codes is scheme.codes:
            return False
        map_target_codes = set.union(*self.data.values())
        target_codes = set(scheme.codes)
        target_dag_codes = set(scheme.dag_codes)
        is_code_subset = map_target_codes.issubset(target_codes)
        is_dag_subset = map_target_codes.issubset(target_dag_codes)
        assert is_code_subset != is_dag_subset, "The target codes are not a subset " \
                                                "of the target codes or the DAG codes."
        return is_dag_subset

    def support_ratio(self, scheme_manager: CodingSchemesManager) -> float:
        """
        Returns the ratio between the source scheme codes covered by the mapping and the total source scheme codes.

        Returns:
            float: the support ratio of the CodeMap.
        """
        source_scheme = scheme_manager.scheme[self.source_name]
        return len(set(self.data.keys()) & set(source_scheme.codes)) / len(source_scheme.codes)

    def range_ratio(self, scheme_manager: CodingSchemesManager) -> float:
        """
        Returns the ratio between the target scheme codes covered by the mapping and the total target scheme codes.

        Returns:
            float: the range ratio of the CodeMap.
        """
        target_scheme = scheme_manager.scheme[self.target_name]
        return len(set.union(*self.data.values()) & set(target_scheme.codes)) / len(target_scheme.codes)

    def log_ratios(self, scheme_manager: CodingSchemesManager) -> float:
        """
        Returns the score of mapping between schemes as the logarithm of support ratio multiplied by range ratio.
        """
        return math.log(self.support_ratio(scheme_manager)) + math.log(self.range_ratio(scheme_manager))

    def __str__(self):
        """
        Returns a string representation of the CodeMap.

        Returns:
            str: the string representation of the CodeMap.
        """
        return f'{self.source_name}->{self.target_name}'

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source_name}->{self.target_name}, {len(self.data)} mappings)"

    def __hash__(self):
        """
        Returns the hash value of the CodeMap.

        Returns:
            int: the hash value of the CodeMap.
        """
        return hash(str(self))

    def __bool__(self):
        """
        Returns True if the CodeMap is not empty, False otherwise.

        Returns:
            bool: true if the CodeMap is not empty, False otherwise.
        """
        return len(self) > 0

    def __len__(self):
        """
        Returns the number of supported codes in the CodeMap.

        Returns:
            int: the number of supported codes in the CodeMap.
        """
        return len(self.data)

    def target_index(self, scheme_manager: CodingSchemesManager):
        """
        Returns the target coding scheme index.

        Returns:
            dict: the target coding scheme index.
        """
        target_scheme = scheme_manager.scheme[self.target_name]
        if self.mapped_to_dag_space(scheme_manager) and self.source_name != self.target_name and hasattr(target_scheme,
                                                                                                         'dag_index'):
            return target_scheme.dag_index
        return target_scheme.index

    def target_desc(self, scheme_manager: CodingSchemesManager):
        """
        Returns the target coding scheme description.

        Returns:
            dict: the target coding scheme description.
        """
        target_scheme = scheme_manager.scheme[self.target_name]
        if self.mapped_to_dag_space(scheme_manager) and self.source_name != self.target_name and hasattr(target_scheme,
                                                                                                         'dag_desc'):
            return target_scheme.dag_desc
        return target_scheme.desc

    def source_index(self, scheme_manager: CodingSchemesManager) -> dict:
        """
        Returns the source coding scheme index.

        Returns:
            dict: the source coding scheme index.
        """
        return scheme_manager.scheme[self.source_name].index

    def __getitem__(self, item):
        """
        Returns the mapped codes for the given item.

        Args:
            item: the item to retrieve the mapped codes for.

        Returns:
            Set[str]: the mapped codes for the given item.
        """
        return self.data[item]

    def __contains__(self, item):
        """
        Checks if an item is in the CodeMap.

        Args:
            item: the item to check.

        Returns:
            bool: True if the item is in the CodeMap, False otherwise.
        """
        return item in self.data

    def map_codeset(self, codeset: Set[str]):
        """
        Maps a codeset to the target coding scheme.

        Args:
            codeset (Set[str]): the codeset to map.

        Returns:
            Set[str]: The mapped codeset.
        """
        return set().union(*[self[c] for c in codeset])

    def map_dataframe(self, df: pd.DataFrame, code_column: str) -> pd.DataFrame:
        df = df.iloc[:, :]
        code2list = {k: list(v) if len(v) > 1 else next(iter(v)) for k, v in self.data.items()}
        df[code_column] = df[code_column].map(code2list)

        if df[code_column].isna().sum() > 0:
            logging.warning(
                f"Some codes are not mapped to the target scheme: {df[code_column].isna().sum()} / {len(df)}")
            # TODO: Add to the report the conversion misses.
            df = df[~df[code_column].isna()]
        return df.explode(code_column)

    def target_code_ancestors(self, scheme_manager: CodingSchemesManager, t_code: str, include_itself=True):
        """
        Returns the ancestors of a target code.

        Args:
            t_code (str): The target code.
            include_itself (bool): Whether to include the target code itself in the ancestors.

        Returns:
            List[str]: The ancestors of the target code.
        """
        target_scheme = scheme_manager.scheme[self.target_name]
        if not self.mapped_to_dag_space(scheme_manager):
            t_code = target_scheme.code2dag[t_code]
        return target_scheme.code_ancestors_bfs(t_code, include_itself=include_itself)


AggregationLiteral = Literal['sum', 'or', 'w_sum']


class GroupingData(VxData):
    permute: np.ndarray
    split: np.ndarray
    size: np.ndarray
    aggregation: Tuple[AggregationLiteral, ...]
    scheme_size: np.ndarray[int, int]

    def __len__(self):
        return 1


class ReducedCodeMapN1(CodeMap):
    set_aggregation: FrozenDict11 = field(kw_only=True)
    reduced_groups: FrozenDict1N = field(kw_only=True)

    @staticmethod
    def from_data(source_name: str, target_name: str, map_data: FrozenDict1N,
                  set_aggregation: FrozenDict11) -> ReducedCodeMapN1:
        assert all(len(t) == 1 for t in map_data.values()), "A code should have one target."

        new_map = defaultdict(set)
        for code, (target,) in map_data.items():
            new_map[target].add(code)

        return ReducedCodeMapN1(source_name=source_name, target_name=target_name, data=map_data,
                                set_aggregation=set_aggregation,
                                reduced_groups=FrozenDict1N.from_dict(new_map))

    def groups(self, scheme_manager: CodingSchemesManager) -> Tuple[Tuple[str, ...], ...]:
        source_index = self.source_index(scheme_manager)
        target_codes = tuple(sorted(self.reduced_groups.keys()))
        return tuple(tuple(sorted(self.reduced_groups[t], key=source_index.get)) for t in target_codes)

    @staticmethod
    def _validate_aggregation(a) -> AggregationLiteral:
        if a in ('sum', 'or', 'w_sum'):
            return a
        else:
            raise ValueError(f"Unrecognised aggregation: {a}")

    @cached_property
    def groups_aggregation(self) -> Tuple[AggregationLiteral, ...]:
        aggregation = tuple(self.set_aggregation[g] for g in sorted(self.reduced_groups.keys()))
        return tuple(self._validate_aggregation(a) for a in aggregation)

    def groups_size(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[int, ...]:
        return tuple(len(g) for g in self.groups(scheme_manager))

    def groups_split(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[int, ...]:
        return tuple(np.cumsum(self.groups_size(scheme_manager)).tolist())

    def groups_permute(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[int, ...]:
        source_index = self.source_index(scheme_manager)
        permutes = sum((tuple(map(source_index.get, g)) for g in self.groups(scheme_manager)), tuple())
        if len(permutes) == len(source_index):
            return permutes
        else:
            return permutes + tuple(set(source_index.keys()) - set(permutes))

    def grouping_data(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> GroupingData:
        source_scheme = scheme_manager.scheme[self.source_name]
        target_scheme = scheme_manager.scheme[self.target_name]
        return GroupingData(permute=np.array(self.groups_permute(scheme_manager), dtype=int),
                            split=np.array(self.groups_split(scheme_manager), dtype=int),
                            size=np.array(self.groups_size(scheme_manager), dtype=int),
                            aggregation=self.groups_aggregation,
                            scheme_size=np.array((len(source_scheme), len(target_scheme))))


class OutcomeExtractor(VxData, metaclass=ABCMeta):
    name: str = field(kw_only=True)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @abstractmethod
    def base_scheme(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> CodingScheme:
        pass

    def codes(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[str, ...]:
        raise NotImplementedError

    def desc(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> FrozenDict11:
        raise NotImplementedError

    def index(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.codes(scheme_manager))}

    def outcome_dim(self, scheme_manager: CodingSchemesManager | SchemeManagerView):
        """
        Gets the dimension of the outcome.

        Returns:
            int: the dimension of the outcome.

        """

        return len(self.index(scheme_manager))

    def codemap(self, scheme_manager: CodingSchemesManager, source_scheme: str) -> CodeMap:
        return scheme_manager.map[(source_scheme, self.base_scheme(scheme_manager).name)]

    def codeset2vec_extractor(self, scheme_manager: CodingSchemesManager, source_scheme: str) -> Callable[
        [Set[str]], CodesVector]:
        codemap = self.codemap(scheme_manager, source_scheme)
        codes = set(self.codes(scheme_manager))
        index = self.index(scheme_manager)

        def _apply(codeset: Set[str]):
            codeset = codemap.map_codeset(codeset) & codes
            vec = np.zeros(len(codes), dtype=bool)
            for c in codeset:
                vec[index[c]] = True
            return CodesVector(vec, self.name)

        return _apply


class ExcludingOutcomeExtractor(OutcomeExtractor):
    exclude_codes: Tuple[str, ...] = field(kw_only=True)
    base_name: str = field(kw_only=True, default=None)

    def codes(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[str, ...]:
        return tuple(c for c in self.base_scheme(scheme_manager).codes if c not in self.exclude_codes)

    def desc(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> FrozenDict11:
        desc = self.base_scheme(scheme_manager).desc
        return FrozenDict11({c: desc[c] for c in self.codes(scheme_manager)})

    def base_scheme(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> CodingScheme:
        return scheme_manager.scheme[self.base_name]


class FileBasedOutcomeExtractor(OutcomeExtractor):
    spec_file: str = field(kw_only=True)
    name: Optional[str] = field(kw_only=True, default=None)

    def __post_init__(self):
        if self.name is None:
            self.name = self.spec_file.split('.')[0]

    def exclude_codes(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[str, ...]:
        return self.specs(scheme_manager)['exclude_codes']

    def codes(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Tuple[str, ...]:
        excluded = self.exclude_codes(scheme_manager)
        base_codes = self.base_scheme(scheme_manager).codes
        return tuple(c for c in sorted(base_codes) if c not in excluded)

    def desc(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> FrozenDict11:
        desc = self.base_scheme(scheme_manager).desc
        return FrozenDict11.from_dict({c: desc[c] for c in self.codes(scheme_manager)})

    def specs(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> Dict[str, Any]:
        return self.spec_from_json(scheme_manager, self.spec_file)

    def base_scheme(self, scheme_manager: CodingSchemesManager | SchemeManagerView) -> CodingScheme:
        """
        Gets the base coding scheme used for outcome extraction.

        Returns:
            CodingScheme: the base coding scheme.

        """
        return scheme_manager.scheme[self.specs(scheme_manager)['code_scheme']]

    @staticmethod
    def spec_from_json(manager: SchemeManagerView | CodingSchemesManager, json_file: str):
        conf = load_config(json_file, relative_to=resources_dir('outcome_filters'))

        if 'exclude_branches' in conf:
            # TODO
            return None
        elif 'select_branches' in conf:
            # TODO
            return None
        elif 'selected_codes' in conf:
            t_scheme = manager.scheme[conf['code_scheme']]
            conf['exclude_codes'] = [
                c for c in t_scheme.codes if c not in conf['selected_codes']
            ]
            return conf
        elif 'exclude_codes' in conf:
            return conf

    @staticmethod
    def from_spec_file(spec_file: str):
        return FileBasedOutcomeExtractor(spec_file=spec_file, name=spec_file.split('.')[0])


class CodingSchemesManager(VxData):
    schemes: Tuple[CodingScheme, ...] = field(default_factory=tuple)
    maps: Tuple[CodeMap, ...] = field(default_factory=tuple)
    outcomes: Tuple[OutcomeExtractor, ...] = field(default_factory=tuple)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.schemes}, {self.maps}, {self.outcomes})"

    def __post_init__(self):
        super().__post_init__()

    def __len__(self):
        return len(self.schemes) + len(self.maps) + len(self.outcomes)

    def view(self) -> SchemeManagerView:
        return SchemeManagerView(_manager=self)

    def add_scheme(self, scheme: CodingScheme) -> CodingSchemesManager:
        assert isinstance(scheme, CodingScheme), f"{scheme} is not a CodingScheme."
        if scheme.name in self.scheme:
            logging.warning(f'Scheme {scheme.name} already exists')
            return self
        return CodingSchemesManager(schemes=self.schemes + (scheme,), maps=self.maps, outcomes=self.outcomes)

    def add_map(self, map: CodeMap) -> CodingSchemesManager:
        assert isinstance(map, CodeMap), f"{map} is not a CodeMap."
        if (map.source_name, map.target_name) in self.map:
            logging.warning(f'Map {map.source_name}->{map.target_name} already exists')
            return self
        return CodingSchemesManager(schemes=self.schemes, maps=self.maps + (map,), outcomes=self.outcomes)

    def add_outcome(self, outcome: OutcomeExtractor) -> CodingSchemesManager:
        assert isinstance(outcome, OutcomeExtractor), f"{outcome} is not an OutcomeExtractor."
        if outcome.name in self.outcome:
            logging.warning(f'Outcome {outcome.name} already exists')
            return self
        return CodingSchemesManager(schemes=self.schemes, maps=self.maps, outcomes=self.outcomes + (outcome,))

    def supported_outcome(self, outcome_name: str, supporting_scheme: str) -> bool:
        return outcome_name in self.outcome and (
            supporting_scheme, self.outcome[outcome_name].base_scheme(self).name) in self.map

    def supported_outcomes(self, supporting_scheme: str) -> Tuple[str, ...]:
        return tuple(o for o in self.outcome if self.supported_outcome(o, supporting_scheme))

    def union(self, other: CodingSchemesManager) -> CodingSchemesManager:
        updated = self
        with logging.captureWarnings(False):
            for s in other.schemes:
                updated = updated.add_scheme(s)
            for m in other.maps:
                updated = updated.add_map(m)
            for o in other.outcomes:
                updated = updated.add_outcome(o)
        return updated

    @cached_property
    def scheme(self) -> Dict[str, CodingScheme]:
        return {s.name: s for s in self.schemes}

    @cached_property
    def identity_maps(self) -> Dict[Tuple[str, str], CodeMap]:
        return {(s, s): CodeMap(source_name=s, target_name=s,
                                data=FrozenDict1N.from_dict({c: {c} for c in self.scheme[s].codes})) for s in
                self.scheme.keys()}

    @cached_property
    def chainable_maps(self) -> Dict[Tuple[str, str, str], CodeMap]:
        raise NotImplementedError

    @cached_property
    def map(self) -> Dict[Tuple[str, str], CodeMap]:
        return {(m.source_name, m.target_name): m for m in self.maps} | self.identity_maps

    @cached_property
    def outcome(self) -> Dict[str, OutcomeExtractor]:
        return {o.name: o for o in self.outcomes}

    def register_chained_map(self, s_scheme: str, inter_scheme: str, t_scheme: str) -> CodingSchemesManager:
        """
        Registers a chained CodeMap. The source and target coding schemes are chained together if there is an intermediate scheme that can act as a bridge between the two.
        There must be registered two CodeMaps, one that maps between the source and intermediate coding schemes and one that maps between the intermediate and target coding schemes.
        Args:
            s_scheme (str): the source coding scheme.
            inter_scheme (str): the intermediate coding scheme.
            t_scheme (str): the target coding scheme.
        """
        assert len({s_scheme, inter_scheme, t_scheme}) == 3, "The schemes should be different."

        map1 = self.map[(s_scheme, inter_scheme)]
        map2 = self.map[(inter_scheme, t_scheme)]
        assert not map1.mapped_to_dag_space(self)
        assert not map2.mapped_to_dag_space(self)

        bridge = lambda x: set.union(*[map2[c] for c in map1[x]])
        s_scheme_object = self.scheme[s_scheme]

        # Supported codes in the new map are the intersection of the source codes and the source codes of the first map
        new_source_codes = set(s_scheme_object.codes) & set(map1.data.keys())
        data = FrozenDict1N.from_dict({c: bridge(c) for c in new_source_codes})
        return self.add_map(CodeMap(source_name=s_scheme, target_name=t_scheme, data=data))

    def register_target_scheme(self, source_name: str, target_name: str,
                               map_table: pd.DataFrame, c_code: str, c_target_code: str,
                               c_target_desc: str) -> CodingSchemesManager:
        """
        Register a target scheme and its mapping.
        # TODO: test me.
        """
        target_codes = tuple(sorted(map_table[c_target_code].drop_duplicates().astype(str).tolist()))
        # drop=False in case c_target_code == c_target_desc.
        target_desc = map_table.set_index(c_target_code, drop=False)[c_target_desc].to_dict()
        target_scheme = CodingScheme(name=target_name, codes=target_codes, desc=FrozenDict11.from_dict(target_desc))
        updated = self.add_scheme(target_scheme)
        source_scheme = self.scheme[source_name]
        map_table = map_table[[c_code, c_target_code]].astype(str)
        map_table = map_table[
            map_table[c_code].isin(source_scheme.codes) & map_table[c_target_code].isin(target_scheme.codes)]
        mapping = map_table.groupby(c_code)[c_target_code].apply(set).to_dict()
        data = FrozenDict1N.from_dict(mapping)
        return updated.add_map(CodeMap(source_name=source_scheme.name, target_name=target_name, data=data))

    def register_scheme_from_selection(self, name: str,
                                       supported_space: pd.DataFrame,
                                       code_selection: Optional[pd.DataFrame],
                                       c_code: str, c_desc: str) -> CodingSchemesManager:
        # TODO: test this method.
        if code_selection is None:
            code_selection = supported_space[c_code].drop_duplicates().astype(str).tolist()
        else:
            code_selection = code_selection[c_code].drop_duplicates().astype(str).tolist()

            assert len(set(code_selection) - set(supported_space[c_code])) == 0, \
                "Some item ids are not supported."
        # drop=False in case c_target_code == c_target_desc.
        desc = supported_space.set_index(c_code, drop=False)[c_desc].to_dict()
        scheme = CodingScheme(name=name,
                              codes=tuple(sorted(code_selection)),
                              desc=FrozenDict11.from_dict({k: v for k, v in desc.items() if k in code_selection}))
        return self.add_scheme(scheme)


class SchemeManagerView(VxDataView):
    _manager: CodingSchemesManager

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @cached_property
    def scheme(self) -> Dict[str, CodingScheme]:
        return self._manager.scheme

    @cached_property
    def map(self) -> Dict[Tuple[str, str], CodeMap]:
        return self._manager.map

    @cached_property
    def outcome(self) -> Dict[str, OutcomeExtractor]:
        return self._manager.outcome

    def supported_outcome(self, outcome_name: str, supporting_scheme: str) -> bool:
        return self._manager.supported_outcome(outcome_name, supporting_scheme)

    def supported_outcomes(self, supporting_scheme: str) -> Tuple[str, ...]:
        return self._manager.supported_outcomes(supporting_scheme)
