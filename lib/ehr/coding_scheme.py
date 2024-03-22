"""Extract diagnostic/procedure information of CCS files into new
data structures to support conversion between CCS and ICD9."""

from __future__ import annotations

import logging
import os
import re
from abc import abstractmethod, ABCMeta
from collections import defaultdict, OrderedDict
from functools import cached_property
from threading import Lock
from types import MappingProxyType
from typing import Set, Dict, Type, Optional, List, Union, ClassVar, Callable, Tuple, Any, Literal, ItemsView, Iterator, \
    Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd
import tables as tb
import tables as tbl

from ..base import Config, Module, VxData
from ..utils import load_config

NumericalTypeHint = Literal['B', 'N', 'O', 'C']  # Binary, Numerical, Ordinal, Categorical


def resources_dir(*subdir) -> str:
    return os.path.join(os.path.dirname(__file__), "resources", *subdir)


class FrozenDict11(VxData):
    data: MappingProxyType[str, Union[str, int]]

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "FrozenDict11":
        return FrozenDict11(MappingProxyType(d))

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

    def to_hdf_group(self, group: tbl.Group):
        df = self.to_dataframe()
        df.to_hdf(group._v_file.filename, key=group._v_pathname)

    @classmethod
    def _from_hdf_group(cls, group: tb.Group) -> 'VxData':
        df = pd.read_hdf(group._v_file.filename, key=group._v_pathname)
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
        return FrozenDict1N.from_dict(df.set_index('key')['value'].groupby('key').apply(set).to_dict())


class CodesVector(VxData):
    """
    Represents a multi-hot vector encoding of codes using a specific coding scheme.

    Attributes:
        vec (np.ndarray): the vector of codes.
        scheme (str): the coding scheme.
    """

    vec: npt.NDArray[bool]
    scheme: str

    @property
    def scheme_object(self):
        """
        Returns the coding scheme object associated with the CodesVector.

        Returns:
            CodingScheme: the coding scheme object.
        """
        return CodingScheme.from_name(self.scheme)

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

    def to_codeset(self):
        """
        Converts the binary codes vector to a set of one code.

        Returns:
            set: a set containing one code.
        """
        return {self.scheme_object.index2code[i] for i in np.flatnonzero(self.vec)}

    def __len__(self) -> int:
        """
        Returns the length of the vector.

        Returns:
            int: the length of the vector.
        """
        return len(self.vec)

    def equals(self, other: CodesVector) -> bool:
        """
        Checks if the current CodesVector is equal to another CodesVector.

        Args:
            other (CodesVector): the other CodesVector to compare with.

        Returns:
            bool: True if the two CodesVectors are equal, False otherwise.
        """
        return (self.scheme == other.scheme) and np.array_equal(self.vec, other.vec)

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
    name: str
    codes: Tuple[str, ...]
    desc: FrozenDict11

    # Possible Schemes, Lazy-loaded schemes.
    _load_schemes: ClassVar[Dict[str, Callable]] = {}
    _schemes: ClassVar[Dict[str, Union["CodingScheme", Any]]] = {}
    # vector representation class
    vector_cls: ClassVar[Type[CodesVector]] = CodesVector

    def __post_init__(self):
        self._check_uniqueness()
        self._check_types()
        self._check_sizes()
        self._check_index_integrity()

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

    @staticmethod
    def from_name(name: str) -> CodingScheme:
        if name in CodingScheme._schemes:
            return CodingScheme._schemes[name]

        if name in CodingScheme._load_schemes:
            CodingScheme._load_schemes[name]()

        return CodingScheme._schemes[name]

    @staticmethod
    def register_scheme(scheme: CodingScheme) -> None:
        """
        Register a scheme in order to be retrieved by its name in `scheme.config.name` using the function `from_name`.
        """
        assert scheme.name not in CodingScheme._schemes or scheme == CodingScheme._schemes[scheme.name], \
            f"Scheme {scheme.name} already registered with mismatched content. Make sure to unregister schemes before" \
            "loading new ones with the same name."

        if scheme.name in CodingScheme._schemes:
            logging.warning(f"Scheme {scheme.name} already registered and matching content. Overwriting.")

        CodingScheme._schemes[scheme.name] = scheme

    @staticmethod
    def register_scheme_loader(name: str, loader: Callable) -> None:
        """
        Register a scheme loader for easy-loading of schemes in order to be retrieved by its name in `name`.
        """

        if name in CodingScheme._load_schemes:
            logging.warning(f"Scheme {name} already registered. Overwriting.")

        CodingScheme._load_schemes[name] = loader

    @staticmethod
    def unregister_schemes():
        """
        Unregister all schemes.
        """
        CodingScheme._schemes = {}

    @staticmethod
    def unregister_scheme_loaders():
        """
        Unregister all scheme loaders.
        """
        CodingScheme._load_schemes = {}

    @staticmethod
    def deregister_scheme(name: str):
        """
        Deregister a scheme by its name.
        """
        if name in CodingScheme._schemes:
            del CodingScheme._schemes[name]

    def register_target_scheme(self,
                               target_name: Optional[str], map_table: pd.DataFrame,
                               c_code: str, c_target_code: str, c_target_desc: str) -> "CodingScheme":
        """
        Register a target scheme and its mapping.
        # TODO: test me.
        """
        target_codes = tuple(sorted(map_table[c_target_code].drop_duplicates().astype(str).tolist()))
        # drop=False in case c_target_code == c_target_desc.
        target_desc = map_table.set_index(c_target_code, drop=False)[c_target_desc].to_dict()
        target_scheme = CodingScheme(name=target_name, codes=target_codes, desc=target_desc)
        self.register_scheme(target_scheme)

        map_table = map_table[[c_code, c_target_code]].astype(str)
        map_table = map_table[map_table[c_code].isin(self.codes) & map_table[c_target_code].isin(target_scheme.codes)]
        mapping = map_table.groupby(c_code)[c_target_code].apply(set).to_dict()
        CodeMap.register_map(CodeMap(CodeMapConfig(self.name, target_scheme.name), mapping))
        return target_scheme

    @staticmethod
    def register_scheme_from_selection(name: str,
                                       supported_space: pd.DataFrame,
                                       code_selection: Optional[pd.DataFrame],
                                       c_code: str, c_desc: str) -> CodingScheme:
        # TODO: test this method.
        if code_selection is None:
            code_selection = supported_space[c_code].drop_duplicates().astype(str).tolist()
        else:
            code_selection = code_selection[c_code].drop_duplicates().astype(str).tolist()

            assert len(set(code_selection) - set(supported_space[c_code])) == 0, \
                "Some item ids are not supported."
        # drop=False in case c_target_code == c_target_desc.
        desc = supported_space.set_index(c_code, drop=False)[c_desc].to_dict()
        desc = {k: v for k, v in desc.items() if k in code_selection}
        scheme = CodingScheme(name=name,
                              codes=tuple(sorted(code_selection)),
                              desc=desc)
        CodingScheme.register_scheme(scheme)
        return scheme

    @staticmethod
    def available_schemes() -> Set[str]:
        """
        Get a list of all registered schemes.
        """
        return set(CodingScheme._schemes.keys()) | set(CodingScheme._load_schemes.keys())

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

    def mapper_to(self, target_scheme: str):
        """
        Returns a mapper between the current scheme and the target scheme.
        """
        return CodeMap.get_mapper(self.name, target_scheme)

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

    @property
    def supported_targets(self):
        return tuple(t for s, t in set(CodeMap._load_maps.keys()) | set(CodeMap._maps.keys()) if s == self.name)

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

    def to_codeset(self):
        """
        Convert the codes vector to a set of codes.

        Returns:
            set: a set of codes represented by the non-zero elements in the vector.
        """
        index = self.vec.nonzero()[0]
        if len(index) == 0:
            return {self.scheme_object.missing_code}


class SchemeWithMissing(CodingScheme):
    """
    A coding scheme that represents categorical schemes and supports missing/unkown values.

    This class extends the `FlatScheme` class and adds support for a missing code.
    It provides methods to convert a set of codes to a multi-hot vector representation,
    where each element in the vector represents the presence or absence of a code.

    Attributes:
        _missing_code (str): the code that represents a missing value in the coding scheme.
    """

    missing_code: str
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
    ch2pt: FrozenDict1N

    dag_codes: Optional[Tuple[str, ...]] = None
    dag_desc: Optional[FrozenDict11] = None
    code2dag: Optional[FrozenDict11] = None

    def __post_init__(self):
        super().__post_init__()

        self.dag_codes = self.dag_codes or self.codes
        self.dag_desc = self.dag_desc or self.desc
        self.code2dag = self.code2dag or {c: c for c in self.codes}

    def _check_types(self):
        """
        Checks the types of the collections in the hierarchical scheme.

        Raises:
            AssertionError: If any of the collections contains a non-string element.
        """
        super()._check_types()

        assert isinstance(self.dag_codes, list), f"{self}: codes should be a list."
        assert isinstance(self.dag_desc, dict), f"{self}: desc should be a dict."
        assert isinstance(self.code2dag, dict), f"{self}: code2dag should be a dict."
        assert isinstance(self.ch2pt, dict), f"{self}: ch2pt should be a dict."

        for collection in [self.dag_codes, self.dag_index, self.dag_desc]:
            assert all(
                isinstance(c, str) for c in collection
            ), f"{self}: All name types should be str."

        assert all(
            isinstance(idx, int)
            for idx in self.dag_index.values()
        ), f"{self}: All index types should be int."

        for collection in [self._dag_codes, self._dag_index, self._dag_desc, self._code2dag, self._dag2code,
                           self._pt2ch, self._ch2pt]:
            assert all(
                type(c) == str
                for c in collection), f"{self}: All name types should be str."

        for collection in [self._dag_desc, self._code2dag, self._dag2code]:
            assert all(
                isinstance(v, str)
                for v in collection.values()
            ), f"{self}: All values should be str."

    def _check_sizes(self):
        super()._check_sizes()
        # TODO: note in the documentation that dag2code size can be less than the dag_codes since some dag_codes are internal nodes that themselves are not are not complete clinical concepts.
        for collection in [self.dag_codes, self.dag_index, self.dag_desc]:
            assert len(collection) == len(self.dag_codes), f"{self}: All collections should have the same size."

    def _check_index_integrity(self):
        super()._check_index_integrity()
        for code, idx in self.dag_index.items():
            assert idx == self.dag_codes.index(
                code), f"{self}: Index of {code} is not consistent with its position in the list."

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
                j = self._dag_index[ancestor_j]
                ancestors_mat[i, j] = 1

        return ancestors_mat

        # self._pt2ch = pt2ch or

    @cached_property
    def dag_index(self) -> Dict[str, int]:
        """
        Dict[str, int]: a dictionary mapping codes to their indices in the hierarchy.
        """
        return {c: i for i, c in enumerate(self.dag_codes)}

    @property
    def dag2code(self) -> Dict[str, str]:
        """
        Dict[str, str]: a dictionary mapping codes in the hierarchy to their corresponding codes.
        """
        return {d: c for c, d in self.code2dag.items()}

    @property
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
    def reverse_connection(connection: FrozenDict1N) -> FrozenDict1N:
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
            current_connections = connection.get(current_code, [])
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
            for conn in connection.get(_node, []):
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
            for conn in connection.get(_node, []):
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


class CodeMapConfig(Config):
    """
    Configuration class for code mapping.

    Attributes:
        source_scheme (str): the source coding scheme.
        target_scheme (str): the target coding scheme.
        mapped_to_dag_space (bool, optional): indicates if the codes are mapped to DAG space in the target scheme, if hierarchical. Defaults to False.
    """
    source_scheme: str
    target_scheme: str
    mapped_to_dag_space: bool = False


class UnsupportedMapping(ValueError):
    pass


class CodeMap(Module):
    """
    Represents a mapping between two coding schemes.

    Attributes:
        config (CodeMapConfig): the configuration of the CodeMap.
        _source_scheme (Union[CodingScheme, HierarchicalScheme]): the source coding scheme.
        _target_scheme (Union[CodingScheme, HierarchicalScheme]): the target coding scheme.
        _data (Dict[str, Set[str]]): the mapping data.

    Class Variables:
        _maps (Dict[Tuple[str, str], CodeMap]): a class-attribute dictionary of registered CodeMaps.
        _load_maps (Dict[Tuple[str, str], Callable]): a class-attribute dictionary of lazy-loaded CodeMaps.
        _maps_lock (Dict[Tuple[str, str], Lock]): a class-attribute dictionary of locks for thread safety.

    Methods:
        __init__(self, config: CodeMapConfig, data: Dict[str, Set[str]]): initializes a CodeMap instance.
        register_map(cls, source_scheme: str, target_scheme: str, mapper: CodeMap): registers a CodeMap.
        register_chained_map(cls, s_scheme: str, inter_scheme: str, t_scheme: str): registers a chained CodeMap.
        register_chained_map_loader(cls, s_scheme: str, inter_scheme: str, t_scheme: str): registers a chained CodeMap lazy-loading function.
        register_map_loader(cls, source_scheme: str, target_scheme: str, loader: Callable): registers a CodeMap lazy-loading function.
        __str__(self): returns a string representation of the CodeMap.
        __hash__(self): returns the hash value of the CodeMap.
        __bool__(self): returns True if the CodeMap is not empty, False otherwise.
        target_index(self): returns the target coding scheme index.
        target_desc(self): returns the target coding scheme description.
        source_index(self): returns the source coding scheme index.
        source_scheme(self): returns the source coding scheme.
        target_scheme(self): returns the target coding scheme.
        mapped_to_dag_space(self): returns True if the CodeMap is mapped to DAG space, False otherwise.
        has_mapper(cls, source_scheme: str, target_scheme: str): returns True if a mapper exists for the given source and target coding schemes, False otherwise.
        get_mapper(cls, source_scheme: str, target_scheme: str) -> CodeMap: returns the mapper for the given source and target coding schemes.
        __getitem__(self, item): returns the mapped codes for the given item.
        __contains__(self, item): returns True if the given item is mapped to the target coding scheme, False otherwise.
        keys(self): returns the supported codes in the source coding scheme that can be mapped to the target scheme.
        map_codeset(self, codeset: Set[str]): maps a codeset to the target coding scheme.
        target_code_ancestors(self, t_code: str, include_itself=True): returns the ancestors of a target code.
        codeset2vec(self, codeset: Set[str]): converts a codeset to a binary vector representation.
        codeset2dagset(self, codeset: Set[str]): converts a codeset to a DAG set representation.
        codeset2dagvec(self, codeset: Set[str]): converts a codeset to a DAG vector representation.
    """
    source_scheme: str
    target_scheme: str
    _data: Dict[str, Set[str]]

    _maps: ClassVar[Dict[Tuple[str, str], CodeMap]] = {}
    _load_maps: ClassVar[Dict[Tuple[str, str], Callable]] = {}
    _maps_lock: ClassVar[Dict[Tuple[str, str], Lock]] = defaultdict(Lock)

    def __init__(self, config: CodeMapConfig, data: Dict[str, Set[str]]):
        """
        Initializes a CodeMap instance.

        Args:
            config (CodeMapConfig): the configuration of the CodeMap.
            data (Dict[str, Set[str]]): the mapping data.
        """
        super().__init__(config=config)
        self._data = data

    @cached_property
    def source_scheme(self) -> Union[CodingScheme, HierarchicalScheme]:
        """
        Returns the source coding scheme.

        Returns:
            Union[CodingScheme, HierarchicalScheme]: the source coding scheme.
        """
        return CodingScheme.from_name(self.config.source_scheme)

    @cached_property
    def target_scheme(self) -> Union[CodingScheme, HierarchicalScheme]:
        """
        Returns the target coding scheme.

        Returns:
            Union[CodingScheme, HierarchicalScheme]: the target coding scheme.
        """
        return CodingScheme.from_name(self.config.target_scheme)

    @classmethod
    def register_map(cls, mapper: CodeMap):
        """
        Registers a CodeMap.

        Args:
            source_scheme (str): the source coding scheme.
            target_scheme (str): the target coding scheme.
            mapper (CodeMap): the CodeMap instance.
        """
        cls._maps[(mapper.config.source_scheme, mapper.config.target_scheme)] = mapper

    @classmethod
    def deregister_map(cls, source_scheme: str, target_scheme: str):
        """
        Deregisters a CodeMap.

        Args:
            source_scheme (str): the source coding scheme.
            target_scheme (str): the target coding scheme.
        """
        if (source_scheme, target_scheme) in cls._maps:
            del cls._maps[(source_scheme, target_scheme)]

    @classmethod
    def register_chained_map(cls, s_scheme: str, inter_scheme: str, t_scheme: str):
        """
        Registers a chained CodeMap. The source and target coding schemes are chained together if there is an intermediate scheme that can act as a bridge between the two.
        There must be registered two CodeMaps, one that maps between the source and intermediate coding schemes and one that maps between the intermediate and target coding schemes.
        Args:
            s_scheme (str): the source coding scheme.
            inter_scheme (str): the intermediate coding scheme.
            t_scheme (str): the target coding scheme.
        """
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
        """
        Registers a chained CodeMap lazy-loading function..

        Args:
            s_scheme (str): the source coding scheme.
            inter_scheme (str): the intermediate coding scheme.
            t_scheme (str): the target coding scheme.
        """
        cls._load_maps[(s_scheme, t_scheme)] = lambda: cls.register_chained_map(s_scheme, inter_scheme, t_scheme)

    @classmethod
    def register_map_loader(cls, source_scheme: str, target_scheme: str, loader: Callable):
        """
        Registers a CodeMap lazy-loading function.

        Args:
            source_scheme (str): the source coding scheme.
            target_scheme (str): the target coding scheme.
            loader (Callable): the loader function.
        """
        cls._load_maps[(source_scheme, target_scheme)] = loader

    def __str__(self):
        """
        Returns a string representation of the CodeMap.

        Returns:
            str: the string representation of the CodeMap.
        """
        return f'{self.source_scheme.name}->{self.target_scheme.name}'

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
        return len(self._data)

    @property
    def target_index(self):
        """
        Returns the target coding scheme index.

        Returns:
            dict: the target coding scheme index.
        """
        if self.config.mapped_to_dag_space and self.source_scheme.name != self.target_scheme.name:
            return self.target_scheme.dag_index
        return self.target_scheme.index

    @property
    def target_desc(self):
        """
        Returns the target coding scheme description.

        Returns:
            dict: the target coding scheme description.
        """
        if self.config.mapped_to_dag_space and self.source_scheme.name != self.target_scheme.name:
            return self.target_scheme.dag_desc
        return self.target_scheme.desc

    @property
    def source_index(self) -> dict:
        """
        Returns the source coding scheme index.

        Returns:
            dict: the source coding scheme index.
        """
        return self.source_scheme.index

    @property
    def source_to_target_index(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping source codes to their indices in the target coding scheme.

        Returns:
            Dict[str, int]: a dictionary mapping source codes to their indices in the target coding scheme.
        """
        return {c: self.target_index[c] for c in self.keys()}

    @property
    def mapped_to_dag_space(self) -> bool:
        """
        Returns True if the CodeMap is mapped to DAG space, False otherwise.

        Returns:
            bool: True if the CodeMap is mapped to DAG space, False otherwise.
        """
        return self.config.mapped_to_dag_space

    @classmethod
    def has_mapper(cls, source_scheme: str, target_scheme: str) -> bool:
        """
        Returns True if a mapper exists for the given source and target coding schemes, False otherwise.

        Args:
            source_scheme (str): the source coding scheme.
            target_scheme (str): the target coding scheme.

        Returns:
            bool: True if a mapper exists, False otherwise.
        """
        key = (source_scheme, target_scheme)
        return key in cls._maps or key[0] == key[1] or key in cls._load_maps

    @classmethod
    def get_mapper(cls, source_scheme: str, target_scheme: str) -> CodeMap:
        """
        Returns the mapper for the given source and target coding schemes.

        Args:
            source_scheme (str): the source coding scheme.
            target_scheme (str): the target coding scheme.

        Returns:
            CodeMap: The mapper for the given source and target coding schemes.
        """
        if not cls.has_mapper(source_scheme, target_scheme):
            raise UnsupportedMapping(f'Mapping {source_scheme}->{target_scheme} is not available')

        key = (source_scheme, target_scheme)
        with cls._maps_lock[key]:
            if key in cls._maps:
                return cls._maps[key]

            if key[0] == key[1]:
                m = IdentityCodeMap(source_scheme)
                cls.register_map(m)
                return m

            if key in cls._load_maps:
                cls._load_maps[key]()

            return cls._maps[key]

    def __getitem__(self, item):
        """
        Returns the mapped codes for the given item.

        Args:
            item: the item to retrieve the mapped codes for.

        Returns:
            Set[str]: the mapped codes for the given item.
        """
        return self._data[item]

    def __contains__(self, item):
        """
        Checks if an item is in the CodeMap.

        Args:
            item: the item to check.

        Returns:
            bool: True if the item is in the CodeMap, False otherwise.
        """
        return item in self._data

    def keys(self):
        """
        Returns the codes in the source coding scheme that have a mapping to the target coding scheme.

        Returns:
            List[str]: the codes in the source coding scheme that have a mapping to the target coding scheme.
        """
        return self._data.keys()

    def map_codeset(self, codeset: Set[str]):
        """
        Maps a codeset to the target coding scheme.

        Args:
            codeset (Set[str]): the codeset to map.

        Returns:
            Set[str]: The mapped codeset.
        """
        return set().union(*[self[c] for c in codeset])

    def target_code_ancestors(self, t_code: str, include_itself=True):
        """
        Returns the ancestors of a target code.

        Args:
            t_code (str): The target code.
            include_itself (bool): Whether to include the target code itself in the ancestors.

        Returns:
            List[str]: The ancestors of the target code.
        """
        if not self.config.mapped_to_dag_space:
            t_code = self.target_scheme.code2dag[t_code]
        return self.target_scheme.code_ancestors_bfs(t_code, include_itself=include_itself)

    def codeset2vec(self, codeset: Set[str]):
        """
        Converts a codeset to a multi-hot vector representation.

        Args:
            codeset (Set[str]): the codeset to convert.

        Returns:
            CodesVector: the binary vector representation of the codeset.
        """
        index = self.target_index
        vec = np.zeros(len(index), dtype=bool)
        try:
            for c in codeset:
                vec[index[c]] = True
        except KeyError as missing:
            logging.error(f'Code {missing} is missing. Accepted keys: {index.keys()}')

        return CodesVector(vec, self.config.target_scheme)

    def codeset2dagset(self, codeset: Set[str]):
        """
        Converts a codeset to a DAG set representation.

        Args:
            codeset (Set[str]): the codeset to convert.

        Returns:
            Set[str]: the DAG set representation of the codeset.
        """
        if not self.config.mapped_to_dag_space:
            return set(self.target_scheme.code2dag[c] for c in codeset)
        else:
            return codeset

    def codeset2dagvec(self, codeset: Set[str]):
        """
        Converts a codeset to a DAG vector representation.

        Args:
            codeset (Set[str]): the codeset to convert.

        Returns:
            np.ndarray: the DAG vector representation of the codeset.
        """
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
            logging.error(f'Code {missing} is missing. Accepted keys: {index.keys()}')

        return vec


class IdentityCodeMap(CodeMap):
    """
    A code mapping class that maps codes to themselves.

    This class inherits from the `CodeMap` base class and provides a simple
    implementation of the `map_codeset` method that returns the input codeset
    unchanged.

    """

    def __init__(self, scheme: str):
        """
        Initialize a Identity CodeMap object.

        Args:
            scheme (str): the name of the coding scheme.S
        """
        config = CodeMapConfig(source_scheme=scheme,
                               target_scheme=scheme,
                               mapped_to_dag_space=False)
        scheme = CodingScheme.from_name(scheme)
        data = {c: {c} for c in scheme.codes}
        super().__init__(config=config, data=data)

    def map_codeset(self, codeset):
        """
        Maps the input codeset to itself.

        This method takes a codeset as input and returns the same codeset
        unchanged.

        Args:
            codeset (list): the input codeset to be mapped.

        Returns:
            list: the mapped codeset, which is the same as the input codeset.

        """
        return codeset


class OutcomeExtractorConfig(CodingSchemeConfig):
    name: str


class OutcomeExtractor(CodingScheme, metaclass=ABCMeta):
    config: OutcomeExtractorConfig

    _supported_outcomes: ClassVar[Dict[str, Set[str]]] = defaultdict(set)

    @property
    @abstractmethod
    def base_scheme(self) -> CodingScheme:
        pass

    @staticmethod
    def from_name(name):
        """
        Creates an instance of the OutcomeExtractor class based on a given supported name.

        Args:
            name: the identifier name of the outcome extractor.

        Returns:
            OutcomeExtractor: an instance of the OutcomeExtractor class.

        Raises:
            AssertionError: if the created instance is not an OutcomeExtractor.
        """

        outcome_extractor = CodingScheme.from_name(name)
        assert isinstance(outcome_extractor,
                          OutcomeExtractor), f'OutcomeExtractor expected, got {type(outcome_extractor)}'
        return outcome_extractor

    @property
    def outcome_dim(self):
        """
        Gets the dimension of the outcome.

        Returns:
            int: the dimension of the outcome.

        """

        return len(self.index)

    def _map_codeset(self, codeset: Set[str], base_scheme: str) -> Set[str]:
        """
        Maps a codeset to the base coding scheme.

        Args:
            codeset (Set[str]): the codeset to map.
            base_scheme (str): the base coding scheme.

        Returns:
            Set[str]: the mapped codeset.

        """

        m = CodeMap.get_mapper(base_scheme, self.base_scheme.name)
        codeset = m.map_codeset(codeset)

        if m.config.mapped_to_dag_space:
            codeset &= set(m.target_scheme.dag2code)
            codeset = set(m.target_scheme.dag2code[c] for c in codeset)

        return codeset & set(self.codes)

    def map_vector(self, codes: CodesVector) -> CodesVector:
        """
        Extract outcomes from a codes vector into a new codes vector.

        Args:
            codes (CodesVector): the codes vector to extract.

        Returns:
            CodesVector: the extracted outcomes represented by a codes vector.

        """

        vec = np.zeros(len(self.index), dtype=bool)
        for c in self._map_codeset(codes.to_codeset(), codes.scheme):
            vec[self.index[c]] = True
        return CodesVector(np.array(vec), self.name)

    @staticmethod
    def supported_outcomes(base_scheme: str) -> Set[str]:
        """
        Gets the supported outcomes for a given base coding scheme.

        Args:
            base_scheme (str): the base coding scheme.

        Returns:
            Tuple[str]: the supported outcomes.
        """
        return OutcomeExtractor._supported_outcomes[base_scheme]


class ExcludingOutcomeExtractorConfig(OutcomeExtractorConfig):
    base_scheme: str
    exclude_codes: List[str]


class ExcludingOutcomeExtractor(OutcomeExtractor):
    config: ExcludingOutcomeExtractorConfig

    def __init__(self, config: ExcludingOutcomeExtractorConfig):
        base_scheme = CodingScheme.from_name(config.base_scheme)
        codes = [c for c in sorted(base_scheme.index) if c not in config.exclude_codes]
        desc = {c: base_scheme.desc[c] for c in codes}
        super().__init__(config=config,
                         codes=codes,
                         desc=desc)

        OutcomeExtractor._supported_outcomes[base_scheme.name].add(self.name)

    @property
    def base_scheme(self) -> CodingScheme:
        return CodingScheme.from_name(self.config.base_scheme)


class FileBasedOutcomeExtractorConfig(CodingSchemeConfig):
    spec_file: str


class FileBasedOutcomeExtractor(OutcomeExtractor):
    """
    Extracts outcomes from a coding scheme based on a given configuration/specs file.

    Args:
        config (OutcomeExtractorConfig): the configuration for the outcome extractor.

    Attributes:
        config (OutcomeExtractorConfig): the configuration for the outcome extractor.
        _spec_files (Dict[str, str]): a class-attribute dictionary of supported spec files.
    """

    config: FileBasedOutcomeExtractorConfig
    _spec_files: ClassVar[Dict[str, str]] = {}

    def __init__(self, config: FileBasedOutcomeExtractorConfig):
        """
        Initializes an instance of the OutcomeExtractor class.

        Args:
            config (FileBasedOutcomeExtractorConfig): The configuration for the outcome extractor.

        """
        self.config = config

        codes = [
            c for c in sorted(self.base_scheme.index)
            if c not in self.specs['exclude_codes']
        ]

        desc = {c: self.base_scheme.desc[c] for c in codes}
        super().__init__(config=config,
                         codes=codes,
                         desc=desc)

    @cached_property
    def specs(self) -> Dict[str, Any]:
        return self.spec_from_json(self.config.spec_file)

    @cached_property
    def base_scheme(self) -> CodingScheme:
        """
        Gets the base coding scheme used for outcome extraction.

        Returns:
            CodingScheme: the base coding scheme.

        """
        return CodingScheme.from_name(self.specs['code_scheme'])

    @staticmethod
    def register_outcome_extractor_loader(name: str, spec_file: str):
        """
        Registers an outcome extractor lazy loading routine.

        Args:
            name (str): the name of the outcome extractor.
            spec_file (str): the spec file for the outcome extractor.

        """

        def load():
            config = FileBasedOutcomeExtractorConfig(name=name, spec_file=spec_file)
            CodingScheme.register_scheme(FileBasedOutcomeExtractor(config))

        CodingScheme.register_scheme_loader(name, load)
        FileBasedOutcomeExtractor._spec_files[name] = spec_file
        base_scheme = load_config(spec_file, relative_to=resources_dir('outcome_filters'))['code_scheme']
        OutcomeExtractor._supported_outcomes[base_scheme].add(name)

    @staticmethod
    def spec_from_json(json_file: str):
        """
        Loads the spec from a JSON file.

        Args:
            json_file (str): the path to the JSON file.

        Returns:
            dict: the loaded spec.

        """

        conf = load_config(json_file, relative_to=resources_dir('outcome_filters'))

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
