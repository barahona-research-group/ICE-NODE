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
from typing import Set, Dict, Type, Optional, List, Union, ClassVar, Callable, Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..base import Config, Module, Data
from ..utils import load_config


def resources_dir(*subdir) -> str:
    return os.path.join(os.path.dirname(__file__), "resources", *subdir)


class CodesVector(Data):
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


class CodingSchemeConfig(Config):
    """The identifier name of the coding scheme."""
    name: str


class CodingScheme(Module):
    """
    CodingScheme defines the base class and utilities for working with medical
    coding schemes. It handles lazy loading of schemes, mapping between schemes,
    converting codesets to vector representations, and searching codes by regex.

    Key attributes and methods:

    - codes, index, desc: Access the codes, indexes, and descriptions
    - name: Get the scheme name
    - index2code, index2desc: Reverse mappings
    - search_regex: Search codes by a regex query
    - mapper_to: Get mapper between schemes
    - codeset2vec: Convert a codeset to a vector representation
    - as_dataframe: View scheme as a Pandas DataFrame
    """

    config: CodingSchemeConfig
    # Possible Schemes, Lazy-loaded schemes.
    _load_schemes: ClassVar[Dict[str, Callable]] = {}
    _schemes: ClassVar[Dict[str, "CodingScheme"]] = {}

    # vector representation class
    vector_cls: ClassVar[Type[CodesVector]] = CodesVector

    @classmethod
    def from_name(cls, name: str) -> "CodingScheme":
        if name in cls._schemes:
            return cls._schemes[name]

        if name in cls._load_schemes:
            cls._load_schemes[name]()

        return cls._schemes[name]

    @classmethod
    def register_scheme(cls, scheme: CodingScheme) -> None:
        """
        Register a scheme in order to be retrieved by its name in `scheme.config.name` using the function `from_name`.
        """
        assert scheme.name not in cls._schemes or scheme == cls._schemes[scheme.name], \
            f"Scheme {scheme.name} already registered with mismatched content. Make sure to unregister schemes before" \
            "loading new ones with the same name."

        if scheme.name in cls._schemes:
            logging.warning(f"Scheme {scheme.name} already registered and matching content. Overwriting.")

        cls._schemes[scheme.name] = scheme

    @classmethod
    def register_scheme_loader(cls, name: str, loader: Callable) -> None:
        """
        Register a scheme loader for easy-loading of schemes in order to be retrieved by its name in `name`.
        """

        if name in cls._load_schemes:
            logging.warning(f"Scheme {name} already registered. Overwriting.")

        cls._load_schemes[name] = loader

    @classmethod
    def unregister_schemes(cls):
        """
        Unregister all schemes.
        """
        cls._schemes = {}

    @classmethod
    def unregister_scheme_loaders(cls):
        """
        Unregister all scheme loaders.
        """
        cls._load_schemes = {}

    @classmethod
    def deregister_scheme(cls, name: str):
        """
        Deregister a scheme by its name.
        """
        if name in cls._schemes:
            del cls._schemes[name]

    def register_target_scheme(self,
                               target_name: Optional[str], map_table: pd.DataFrame,
                               c_code: str, c_target_code: str, c_target_desc: str) -> "FlatScheme":
        """
        Register a target scheme and its mapping.
        # TODO: test me.
        """
        target_scheme_conf = CodingSchemeConfig(target_name)
        target_codes = sorted(map_table[c_target_code].drop_duplicates().astype(str).tolist())
        # drop=False in case c_target_code == c_target_desc.
        target_desc = map_table.set_index(c_target_code, drop=False)[c_target_desc].to_dict()
        target_scheme = FlatScheme(target_scheme_conf, codes=target_codes, desc=target_desc)
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
                                       c_code: str, c_desc: str) -> FlatScheme:
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
        scheme = FlatScheme(CodingSchemeConfig(name),
                            codes=sorted(code_selection),
                            desc=desc)
        FlatScheme.register_scheme(scheme)
        return scheme

    @classmethod
    def available_schemes(cls) -> Set[str]:
        """
        Get a list of all registered schemes.
        """
        return set(cls._schemes.keys()) | set(cls._load_schemes.keys())

    @property
    @abstractmethod
    def codes(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def index(self) -> Dict[str, int]:
        pass

    @property
    @abstractmethod
    def desc(self) -> Dict[str, str]:
        pass

    @property
    def name(self) -> str:
        return self.config.name

    @property
    @abstractmethod
    def index2code(self) -> Dict[int, str]:
        pass

    @property
    @abstractmethod
    def index2desc(self) -> Dict[int, str]:
        pass

    def equals(self, other: "CodingScheme") -> bool:
        """
        Check if the current scheme is equal to another scheme.
        """
        return ((self.name == other.name) and (self.codes == other.codes) and (self.index == other.index) and (
                self.desc == other.desc))

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

    def empty_vector(self) -> "CodingScheme.vector_cls":
        """
        Returns an empty vector representation of the current scheme.
        Returns:
            CodingScheme.vector_cls: an empty vector representation of the current scheme.
        """
        return CodingScheme.vector_cls.empty(self.name)

    @property
    def supported_targets(self):
        return tuple(t for s, t in set(CodeMap._load_maps.keys()) | set(CodeMap._maps.keys()) if s == self.name)

    def as_dataframe(self):
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


class FlatScheme(CodingScheme):
    """FlatScheme is a subclass of CodingScheme that represents a flat coding scheme.

    It contains the following attributes to represent the coding scheme:

    - config: The CodingSchemeConfig for this scheme
    - _codes: List of code strings
    - _index: Dict mapping codes to integer indices
    - _desc: Dict mapping codes to descriptions
    - _index2code: Reverse mapping of _index
    - _index2desc: Reverse mapping of _desc

    The __init__ method constructs the scheme from the provided components after validating the types.

    Additional methods:

    - codes: Property exposing _codes
    - index: Property exposing _index
    - desc: Property exposing _desc
    - index2code: Property exposing _index2code
    - index2desc: Property exposing _index2desc

    Subclasses would inherit from FlatScheme to represent specific flat coding scheme implementations.
    """

    config: CodingSchemeConfig
    _codes: List[str]
    _index: Dict[str, int]
    _desc: Dict[str, str]
    _index2code: Dict[int, str]
    _index2desc: Dict[int, str]

    def __init__(
            self,
            config: CodingSchemeConfig,
            codes: List[str],
            desc: Dict[str, str],
    ):
        super().__init__(config=config)

        logging.debug(f"Constructing {config.name} ({type(self)}) scheme")

        self._codes = list(codes)
        self._index = dict(zip(self._codes, range(len(codes))))
        self._desc = desc

        self._index2code = {idx: code for code, idx in self._index.items()}
        self._index2desc = {self._index[code]: _desc for code, _desc in desc.items()}

        self._check_uniqueness()
        self._check_types()
        self._check_sizes()
        self._check_index_integrity()

    def _check_uniqueness(self):
        assert len(self.codes) == len(set(self.codes)), f"{self}: Codes should be unique."

    def _check_types(self):
        assert isinstance(self.config, CodingSchemeConfig), f"{self}: config should be CodingSchemeConfig."
        assert isinstance(self.codes, list), f"{self}: codes should be a list."
        assert isinstance(self.index, dict), f"{self}: index should be a dict."
        assert isinstance(self.desc, dict), f"{self}: desc should be a dict."
        for collection in [self.codes, self.index, self.desc]:
            assert all(
                isinstance(c, str) for c in collection
            ), f"{self}: All name types should be str."

        assert all(
            isinstance(idx, int)
            for idx in self.index.values()
        ), f"{self}: All index types should be int."

        assert all(
            isinstance(desc, str)
            for desc in self.desc.values()

        ), f"{self}: All desc types should be str."

    def _check_sizes(self):
        for collection in [self.codes, self.index, self.desc, self.index2code, self.index2desc]:
            assert len(collection) == len(self.codes)

    def _check_index_integrity(self):
        for code, idx in self.index.items():
            assert code == self.codes[idx], f"{self}: Index of {code} is not consistent with its position in the list."

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


class BinaryCodesVector(CodesVector):
    """
    Represents a single-element binary code vector. It is used to represent a single code in a binary coding scheme, where the two codes are mutually exclusive.

    Attributes:
        vec (numpy.ndarray): the binary codes vector, which is a 1-element vector.
        scheme (str): the binary coding scheme associated with the vector.
    """

    @classmethod
    def empty(cls, scheme: str):
        """
        Creates a default binary codes vector.

        Args:
            scheme (str): the coding scheme associated with the vector.

        Returns:
            BinaryCodesVector: a vector representing the first code in the binary coding scheme.
        """
        return cls(np.zeros(1, dtype=bool), scheme)

    def to_codeset(self):
        """
        Converts the binary codes vector to a set of one code.

        Returns:
            set: a set containing one code.
        """
        return {self.scheme_object.index2code[self.vec[0]]}

    def __len__(self):
        """
        Returns 1, the length of the binary codes vector.

        Returns:
            int: the length of the binary codes vector.
        """
        return 1


class BinaryScheme(FlatScheme):
    """
    A class representing a special case of a coding scheme with two, mutually exclusive codes.

    Inherits from the FlatScheme class.

    Attributes:
        codes (List[str]): List containing the two codes in the scheme.
        index (Dict[str, int]): Dictionary mapping the two codes to their indices.
        desc (Dict[str, str]): Dictionary mapping the two codes to their descriptions.
        name (str): Name of the coding scheme.

    Methods:
        codeset2vec(code: str) -> BinaryCodesVector:
            Converts a code to a BinaryCodesVector object.

        __len__() -> int:
            Returns the length of the coding scheme vectorized representation, i.e. one code can be represented at a time.
    """

    vector_cls: ClassVar[Type[BinaryCodesVector]] = BinaryCodesVector

    def __init__(self, codes: List[str], index: Dict[str, int], desc: Dict[str, str], name: str) -> object:
        assert all(len(c) == 2 for c in (codes, index, desc)), \
            f"{self}: Codes should be of length 2."
        super().__init__(codes, index, desc, name)

    def codeset2vec(self, code: str) -> "BinaryCodesVector":
        """
        Converts a code to a BinaryCodesVector object.
        Args:
            code (str): The code to be converted to a BinaryCodesVector object.

        Returns:
            BinaryCodesVector: A BinaryCodesVector object representing the code.
        """
        return BinaryCodesVector(np.array(self.index[code], dtype=bool), self)

    def __len__(self) -> int:
        """
        Returns the length of the coding scheme vectorized representation, i.e. one code can be represented at a time.
        Returns:
            int: The length of the coding scheme vectorized representation, i.e. one code can be represented at a time.
        """
        return 1


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


class SchemeWithMissing(FlatScheme):
    """
    A coding scheme that represents categorical schemes and supports missing/unkown values.

    This class extends the `FlatScheme` class and adds support for a missing code.
    It provides methods to convert a set of codes to a multi-hot vector representation,
    where each element in the vector represents the presence or absence of a code.

    Attributes:
        _missing_code (str): the code that represents a missing value in the coding scheme.
    """

    _missing_code: str
    vector_cls: ClassVar[Type[CodesVectorWithMissing]] = CodesVectorWithMissing

    def __init__(self, config: CodingSchemeConfig,
                 codes: List[str], desc: Dict[str, str], missing_code: str):
        self._missing_code = missing_code
        super().__init__(config, codes, desc)
        assert missing_code not in codes, f"{self}: Missing code should not be in the list of codes."
        self._codes = codes + [missing_code]
        self._desc = desc.copy()
        self._desc[self._missing_code] = "Missing"
        self._index[self._missing_code] = -1

        self._check_index_integrity()

    def __len__(self) -> int:
        return len(self.codes) - 1

    @property
    def missing_code(self):
        return self._missing_code

    def _check_index_integrity(self):
        for code, idx in self.index.items():
            if code == self.missing_code:
                continue
            assert idx == self.codes.index(
                code), f"{self}: Index of {code} is not consistent with its position in the list."


class NullScheme(FlatScheme):
    """
    A coding scheme that represents a null scheme.

    This scheme does not contain any codes or mappings. Only used as a typed placeholder alternative to None.

    Attributes:
        None

    Methods:
        __init__: Initializes the NullScheme object.
    """

    def __init__(self):
        super().__init__(CodingSchemeConfig('null'), [], {})


class HierarchicalScheme(FlatScheme):
    """
    A class representing a hierarchical coding scheme.

    This class extends the functionality of the FlatScheme class and provides
    additional methods for working with hierarchical coding schemes.

    Attributes:
        dag_index (Dict[str, int]): A dictionary mapping codes to their indices in the hierarchy.
        dag_codes (List[str]): A list of codes in the hierarchy.
        dag_desc (Dict[str, str]): A dictionary mapping codes to their descriptions in the hierarchy.
        code2dag (Dict[str, str]): A dictionary mapping codes to their corresponding codes in the hierarchy.
        dag2code (Dict[str, str]): A dictionary mapping codes in the hierarchy to their corresponding codes.
    """

    _dag_codes: List[str]
    _dag_index: Dict[str, int]
    _dag_desc: Dict[str, str]
    _code2dag: Dict[str, str]
    _dag2code: Dict[str, str]
    _pt2ch: Dict[str, Set[str]]
    _ch2pt: Dict[str, Set[str]]

    def __init__(self,
                 config: CodingSchemeConfig,
                 codes: Optional[List[str]] = None,
                 desc: Optional[Dict[str, str]] = None,
                 dag_codes: Optional[List[str]] = None,
                 dag_desc: Optional[Dict[str, str]] = None,
                 code2dag: Optional[Dict[str, str]] = None,
                 pt2ch: Optional[Dict[str, Set[str]]] = None,
                 ch2pt: Optional[Dict[str, Set[str]]] = None):
        """
        Initializes a HierarchicalScheme object.

        Args:
            config (CodingSchemeConfig): the configuration for the coding scheme.
            codes (Optional[List[str]]): a list of codes in the flattened version of the scheme. Defaults to None.
            desc (Optional[Dict[str, str]]): a dictionary mapping codes to their descriptions in the flattened version of the scheme. Defaults to None.
            dag_codes (Optional[List[str]]): a list of codes in the hierarchical scheme. Defaults to None.
            dag_desc (Optional[Dict[str, str]]): a dictionary mapping codes to their descriptions in the hierarchical scheme. Defaults to None.
            code2dag (Optional[Dict[str, str]]): a dictionary mapping codes in the flat scheme to their corresponding codes in the hierarchical scheme. Defaults to None.
            pt2ch (Optional[Dict[str, Set[str]]]): a dictionary mapping parent codes to their child codes in the hierarchical scheme. Defaults to None.
            ch2pt (Optional[Dict[str, Set[str]]]): a dictionary mapping child codes to their parent codes in the hierarchical scheme. Defaults to None.
        """

        self._dag_codes = dag_codes or codes
        self._dag_index = {c: i for i, c in enumerate(self._dag_codes)}
        self._dag_desc = dag_desc or desc
        self._code2dag = code2dag or {c: c for c in codes}
        self._dag2code = {d: c for c, d in self._code2dag.items()}

        assert pt2ch or ch2pt, (
            "Should provide ch2pt or pt2ch connection dictionary")
        self._pt2ch = pt2ch or self.reverse_connection(ch2pt)
        self._ch2pt = ch2pt or self.reverse_connection(pt2ch)

        super().__init__(config, codes, desc)

    def equals(self, other: "HierarchicalScheme") -> bool:
        return (super().equals(other) and (self._dag_codes == other._dag_codes) and (
                self._dag_index == other._dag_index)
                and (self._dag_desc == other._dag_desc)
                and (self._code2dag == other._code2dag)
                and (self._dag2code == other._dag2code)
                and (self._pt2ch == other._pt2ch)
                and (self._ch2pt == other._ch2pt))

    def _check_types(self):
        """
        Checks the types of the collections in the hierarchical scheme.

        Raises:
            AssertionError: If any of the collections contains a non-string element.
        """
        super()._check_types()

        assert isinstance(self.config, CodingSchemeConfig), f"{self}: config should be CodingSchemeConfig."

        assert isinstance(self.dag_codes, list), f"{self}: codes should be a list."
        assert isinstance(self.dag_index, dict), f"{self}: index should be a dict."
        assert isinstance(self.dag_desc, dict), f"{self}: desc should be a dict."
        assert isinstance(self.code2dag, dict), f"{self}: code2dag should be a dict."
        assert isinstance(self.dag2code, dict), f"{self}: dag2code should be a dict."
        assert isinstance(self.pt2ch, dict), f"{self}: pt2ch should be a dict."
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

    @property
    def dag_index(self) -> Dict[str, int]:
        """
        Dict[str, int]: a dictionary mapping codes to their indices in the hierarchy.
        """
        return self._dag_index

    @property
    def dag_codes(self) -> List[str]:
        """
        List[str]: a list of codes in the hierarchy.
        """
        return self._dag_codes

    @property
    def dag_desc(self) -> Dict[str, str]:
        """
        Dict[str, str]: a dictionary mapping codes to their descriptions in the hierarchy.
        """
        return self._dag_desc

    @property
    def code2dag(self) -> Dict[str, str]:
        """
        Dict[str, str]: a dictionary mapping codes to their corresponding codes in the hierarchy.
        """
        return self._code2dag

    @property
    def dag2code(self) -> Dict[str, str]:
        """
        Dict[str, str]: a dictionary mapping codes in the hierarchy to their corresponding codes.
        """
        return self._dag2code

    @property
    def pt2ch(self):
        return self._pt2ch

    @property
    def ch2pt(self):
        return self._ch2pt

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
    def reverse_connection(connection: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
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
        return rev_connection

    @staticmethod
    def _bfs_traversal(connection: Dict[str, Set[str]], code: str, include_itself: bool) -> List[str]:
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
    def _dfs_traversal(connection: Dict[str, Set[str]], code: str, include_itself: bool) -> List[str]:
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
    def _dfs_edges(connection: Dict[str, Set[str]], code: str) -> Set[Tuple[str, str]]:
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
        return self._bfs_traversal(self._ch2pt, code, include_itself)

    def code_ancestors_dfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the ancestors of a code in the hierarchy using depth-first traversal.

        Args:
            code (str): the code for which to find the ancestors.
            include_itself (bool): whether to include the code itself as its own ancestor. Defaults to True.

        Returns:
            List[str]: a list of ancestor codes.
        """
        return self._dfs_traversal(self._ch2pt, code, include_itself)

    def code_successors_bfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the successors of a code in the hierarchy using breadth-first traversal.

        Args:
            code (str): the code for which to find the successors.
            include_itself (bool): whether to include the code itself as its own successor. Defaults to True.

        Returns:
            List[str]: A list of successor codes.
        """
        return self._bfs_traversal(self._pt2ch, code, include_itself)

    def code_successors_dfs(self, code: str, include_itself: bool) -> List[str]:
        """
        Returns the successors of a code in the hierarchy using depth-first traversal.

        Args:
            code (str): the code for which to find the successors.
            include_itself (bool): whether to include the code itself as its own successor. Defaults to True.

        Returns:
            List[str]: a list of successor codes.
        """
        return self._dfs_traversal(self._pt2ch, code, include_itself)

    def ancestors_edges_dfs(self, code: str) -> Set[Tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal of ancestors.

        Args:
            code (str): the code for which to find the ancestor edges.

        Returns:
            Set[Tuple[str, str]]: a set of edges in the hierarchy.
        """
        return self._dfs_edges(self._ch2pt, code)

    def successors_edges_dfs(self, code: str) -> Set[Tuple[str, str]]:
        """
        Returns the edges of the hierarchy obtained through a depth-first traversal of successors.

        Args:
            code (str): the code for which to find the successor edges.

        Returns:
            Set[Tuple[str, str]]: a set of edges in the hierarchy.
        """
        return self._dfs_edges(self._pt2ch, code)

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
            lambda c: re.findall(query, self._desc[c], flags=regex_flags),
            self.codes)

        dag_codes = filter(
            lambda c: re.findall(query, self._dag_desc[c], flags=regex_flags),
            self.dag_codes)

        all_codes = set(map(self._code2dag.get, codes)) | set(dag_codes)

        for c in list(all_codes):
            all_codes.update(self.code_successors_dfs(c))

        return all_codes


class Ethnicity(FlatScheme):
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
    config: CodeMapConfig
    _source_scheme: Union[CodingScheme, HierarchicalScheme]
    _target_scheme: Union[CodingScheme, HierarchicalScheme]
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
        self._source_scheme = CodingScheme.from_name(config.source_scheme)
        self._target_scheme = CodingScheme.from_name(config.target_scheme)

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
        if self.config.mapped_to_dag_space and self._source_scheme.name != self._target_scheme.name:
            return self._target_scheme.dag_index
        return self._target_scheme.index

    @property
    def target_desc(self):
        """
        Returns the target coding scheme description.

        Returns:
            dict: the target coding scheme description.
        """
        if self.config.mapped_to_dag_space and self._source_scheme.name != self._target_scheme.name:
            return self._target_scheme.dag_desc
        return self._target_scheme.desc

    @property
    def source_index(self) -> dict:
        """
        Returns the source coding scheme index.

        Returns:
            dict: the source coding scheme index.
        """
        return self._source_scheme.index

    @property
    def source_scheme(self) -> Union[CodingScheme, HierarchicalScheme]:
        """
        Returns the source coding scheme.

        Returns:
            Union[CodingScheme, HierarchicalScheme]: the source coding scheme.
        """
        return self._source_scheme

    @property
    def target_scheme(self) -> Union[CodingScheme, HierarchicalScheme]:
        """
        Returns the target coding scheme.

        Returns:
            Union[CodingScheme, HierarchicalScheme]: the target coding scheme.
        """
        return self._target_scheme

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
            logging.warning(f'Mapping {source_scheme}->{target_scheme} is not available')
            return NullCodeMap()

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
        if self.config.mapped_to_dag_space == False:
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

        return CodesVector(vec, self.target_scheme)

    def codeset2dagset(self, codeset: Set[str]):
        """
        Converts a codeset to a DAG set representation.

        Args:
            codeset (Set[str]): the codeset to convert.

        Returns:
            Set[str]: the DAG set representation of the codeset.
        """
        if self.config.mapped_to_dag_space == False:
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


class NullCodeMap(CodeMap):
    """
    A code map implementation that represents a null mapping.

    This class provides a null implementation of the CodeMap interface.
    It does not perform any mapping and always returns None.

    Attributes:
        config (CodeMapConfig): the configuration for the code map.
        data (dict): the data for the code map.

    Methods:
        map_codeset(codeset): teturns None.
        codeset2vec(codeset): teturns None.
        __bool__(): Returns False.
    """

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

    @classmethod
    def empty(cls, scheme: str) -> CodesVector:
        """
        Creates an empty CodesVector with the specified coding scheme.

        Args:
            scheme (str): the coding scheme.

        Returns:
            CodesVector: the empty CodesVector.
        """
        return cls(np.zeros(len(CodingScheme.from_name(scheme)), dtype=bool), scheme)

    def to_codeset(self):
        """
        Converts the CodesVector to a set of codes using the associated coding scheme.

        Returns:
            set: the set of codes.
        """
        index = self.vec.nonzero()[0]
        scheme = self.scheme_object
        return set(scheme.index2code[i] for i in index)

    def union(self, other):
        """
        Performs a union operation with another CodesVector.

        Args:
            other (CodesVector): the CodesVector to perform the union with.

        Returns:
            CodesVector: the resulting CodesVector after the union operation.
        """
        return CodesVector(self.vec | other.vec, self.scheme)

    def __len__(self):
        """
        Returns the length of the CodesVector.

        Returns:
            int: the length of the CodesVector.
        """
        return len(self.vec)


def register_gender_scheme():
    """
    Register a binary gender coding scheme.

    This function registers a binary coding scheme for gender, with codes 'M' for male and 'F' for female.
    The index maps the codes to their corresponding positions, and the desc provides descriptions for each code.
    """
    CodingScheme.register_scheme(BinaryScheme(CodingSchemeConfig('gender'),
                                              codes=['M', 'F'],
                                              desc={'M': 'male', 'F': 'female'}))


# _OUTCOME_DIR = os.path.join(resources_dir(), 'outcome_filters')


class OutcomeExtractorConfig(CodingSchemeConfig):
    name: str


class OutcomeExtractor(FlatScheme, metaclass=ABCMeta):
    config: OutcomeExtractorConfig
    # Possible Schemes, Lazy-loaded schemes.
    _load_schemes: ClassVar[Dict[str, Callable]] = {}
    _schemes: ClassVar[Dict[str, "CodingScheme"]] = {}

    @property
    @abstractmethod
    def base_scheme(self) -> CodingScheme:
        pass

    @classmethod
    def from_name(cls, name):
        """
        Creates an instance of the OutcomeExtractor class based on a given supported name.

        Args:
            name: the identifier name of the outcome extractor.

        Returns:
            OutcomeExtractor: an instance of the OutcomeExtractor class.

        Raises:
            AssertionError: if the created instance is not an OutcomeExtractor.
        """

        outcome_extractor = super().from_name(name)
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

    def mapcodevector(self, codes: CodesVector) -> CodesVector:
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

    @classmethod
    def register_outcome_extractor_loader(cls, name: str, spec_file: str):
        """
        Registers an outcome extractor lazy loading routine.

        Args:
            name (str): the name of the outcome extractor.
            spec_file (str): the spec file for the outcome extractor.

        """

        def load():
            config = FileBasedOutcomeExtractorConfig(name=name, spec_file=spec_file)
            cls.register_scheme(FileBasedOutcomeExtractor(config))

        cls._spec_files[name] = spec_file
        cls.register_scheme_loader(name, load)

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

    @staticmethod
    def supported_outcomes(base_scheme: str):
        """
        Gets the supported outcomes for a given base coding scheme.

        Args:
            base_scheme (str): the base coding scheme.

        Returns:
            Tuple[str]: the supported outcomes.
        """

        outcome_base = {
            k: load_config(v, relative_to=resources_dir('outcome_filters'))['code_scheme']
            for k, v in FileBasedOutcomeExtractor._spec_files.items()
        }
        return tuple(k for k, v in outcome_base.items()
                     if v == base_scheme or v in CodingScheme.from_name(base_scheme).supported_targets)
