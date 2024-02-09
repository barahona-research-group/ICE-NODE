import os
from copy import deepcopy
from typing import Dict, List
from unittest import mock

import pytest

from lib.ehr import (CodingScheme, FlatScheme, CodingSchemeConfig, setup_icd, setup_cprd, OutcomeExtractor)

# from unittest import TestCase, main as main_test
# from unittest.mock import patch
# from parameterized import parameterized

_DIR = os.path.dirname(__file__)


# TODO: Add tests for the following:
# [  ] test CodeMaps:
#       [  ] register code map
#       [  ] test mapper_to
#       [  ] test chained maps
#       [  ] test code2index
#       [  ] test code2vec
# [  ] test hierarchical schemes:
#       [  ] test 1
#       [  ] test 2
#       [  ] test 3
# [  ] test OutcomeExtractor:
#       [  ] test 1
#       [  ] test 2
#       [  ] test 3

@pytest.fixture(scope='class', params=[dict(name='one', codes=['1'], desc={'1': 'one'}),
                                       dict(name='zero', codes=[], desc=dict()),
                                       dict(name='100', codes=list(f'code_{i}' for i in range(100)))])
def primitive_flat_scheme_kwarg(request):
    config = CodingSchemeConfig(request.param['name'])
    if 'desc' in request.param:
        desc = request.param['desc']
    elif len(request.param['codes']) > 0:
        desc = dict(zip(request.param['codes'], request.param['codes']))
    else:
        desc = dict()
    return dict(config=config, codes=request.param['codes'], desc=desc)


@pytest.fixture(scope='class')
def primitive_flat_scheme(primitive_flat_scheme_kwarg):
    return FlatScheme(**primitive_flat_scheme_kwarg)


@pytest.fixture
def clean_schemes():
    CodingScheme.unregister_schemes()
    CodingScheme.unregister_scheme_loaders()
    OutcomeExtractor.unregister_schemes()
    OutcomeExtractor.unregister_scheme_loaders()


class TestFlatScheme:

    def test_from_name(self, primitive_flat_scheme):
        """
        Test the `from_name` method of the FlatScheme class.

        This method registers the scheme and asserts that
        calling `from_name` with the scheme's name returns the same scheme object.

        If a KeyError is raised, it means that the scheme is not registered and the test passes.
        """
        FlatScheme.register_scheme(primitive_flat_scheme)
        assert FlatScheme.from_name(primitive_flat_scheme.config.name) is primitive_flat_scheme

        with pytest.raises(KeyError):
            # Unregistered scheme
            FlatScheme.from_name('42')

    @pytest.mark.parametrize("codes", [('A', 'B', 'C', 'C'),
                                       ('A', 'B', 'C', 'B'),
                                       ('A', 'A', 'A', 'A')])
    def test_codes_uniqueness(self, codes):
        with pytest.raises(AssertionError) as excinfo:
            FlatScheme(CodingSchemeConfig('test'), codes=codes, desc={c: c for c in codes})
        assert 'should be unique' in str(excinfo.value)

    def test_register_scheme(self, primitive_flat_scheme):
        """
        Test the register_scheme method.

        This method tests two scenarios:
        1. It tests that the register_scheme method works by registering a scheme and then
           asserting that the registered scheme can be retrieved using its name.
        2. It tests that the register_scheme method logs a warning when trying to
           register a scheme that is already registered with the same name and content.
        """
        # First, test that the register_scheme method works.
        FlatScheme.register_scheme(primitive_flat_scheme)
        assert FlatScheme.from_name(primitive_flat_scheme.config.name) is primitive_flat_scheme

        # Second, test that the register_scheme method raises an error when
        # the scheme is already registered.
        with mock.patch('logging.warning') as mocker:
            FlatScheme.register_scheme(primitive_flat_scheme)
            mocker.assert_called_once()

    def test_scheme_equality(self, primitive_flat_scheme):
        """
        Test the equality of schemes.

        This test asserts that a scheme equal to its deepcopy.
        It then mutates the description and index of one of the schemes and asserts that the two
        schemes are not equal.
        """
        assert primitive_flat_scheme == deepcopy(primitive_flat_scheme)

        if len(primitive_flat_scheme) > 0:
            desc_mutated = {code: f'{desc} muted' for code, desc in primitive_flat_scheme.desc.items()}
            mutated_scheme = FlatScheme(config=primitive_flat_scheme.config,
                                        codes=primitive_flat_scheme.codes,
                                        desc=desc_mutated)
            assert primitive_flat_scheme != mutated_scheme

    def test_register_scheme_loader(self, primitive_flat_scheme):
        """
        Test case for registering a scheme loader and verifying the scheme registration.

        This test performs the following steps:
        1. Registers a scheme loader for the scheme's name using a lambda function.
        2. Asserts that the scheme can be retrieved using the scheme's name.
        3. Asserts that attempting to register the same scheme loader again logs a warning if it has the same name with
        matching content.
        4. Asserts that attempting to register the same scheme again logs a warning.
        """

        FlatScheme.register_scheme_loader(primitive_flat_scheme.config.name,
                                          lambda: FlatScheme.register_scheme(primitive_flat_scheme))
        assert FlatScheme.from_name(primitive_flat_scheme.config.name) is primitive_flat_scheme

        with mock.patch('logging.warning') as mocker:
            FlatScheme.register_scheme_loader(primitive_flat_scheme.config.name,
                                              lambda: FlatScheme.register_scheme(primitive_flat_scheme))
            mocker.assert_called_once()
        with mock.patch('logging.warning') as mocker:
            FlatScheme.register_scheme(primitive_flat_scheme)
            mocker.assert_called_once()

        # If the scheme is registered with the same name but different content, an AssertionError should be raised.
        if len(primitive_flat_scheme) > 0:
            desc_mutated = {code: f'{desc} muted' for code, desc in primitive_flat_scheme.desc.items()}
            mutated_scheme = FlatScheme(config=primitive_flat_scheme.config,
                                        codes=primitive_flat_scheme.codes,
                                        desc=desc_mutated)

            with pytest.raises(AssertionError):
                FlatScheme.register_scheme(mutated_scheme)

    @pytest.mark.expensive_test
    @pytest.mark.usefixtures('clean_schemes')
    def test_registered_schemes(self):
        """
        Test case to verify the behavior of registered coding schemes.

        This test case checks the following:
        - The initial set of available coding schemes is empty.
        - After each setup (setup_cprd, setup_mimic, setup_icd), the number of available coding schemes increases.
        - Each registered coding scheme is an instance of FlatScheme object.
        - The name of the instantiated FlatScheme object matches the registered coding scheme.
        """

        assert CodingScheme.available_schemes() == set()
        count = 0
        for setup in (setup_cprd, setup_icd):
            setup()
            assert len(CodingScheme.available_schemes()) > count
            count = len(CodingScheme.available_schemes())

        for registered_scheme in CodingScheme.available_schemes():
            scheme = FlatScheme.from_name(registered_scheme)
            assert isinstance(scheme, FlatScheme)
            assert scheme.name == registered_scheme

    @pytest.mark.parametrize("config, codes, desc",
                             [(CodingSchemeConfig('problematic_codes'), [1], {'1': 'one'}),
                              (CodingSchemeConfig('problematic_desc'), ['1'], {1: 'one'}),
                              (CodingSchemeConfig('problematic_desc'), ['1'], {'1': 5})])
    def test_type_error(self, config: CodingSchemeConfig, codes: List[str], desc: Dict[str, str]):
        """
        Test for type error handling in the FlatScheme constructor.

        This test adds a problematic scheme to test the error handling of the constructor.
        The code and description types should be strings, not integers. The test expects an AssertionError
        or KeyError to be raised.
        """
        with pytest.raises((AssertionError, KeyError)):
            FlatScheme(config=config, codes=codes, desc=desc)

    @pytest.mark.parametrize("config, codes, desc", [
        (CodingSchemeConfig('problematic_desc'), ['1', '3'], {'1': 'one'}),
        (CodingSchemeConfig('problematic_desc'), ['3'], {'3': 'three', '1': 'one'}),
        (CodingSchemeConfig('duplicate_codes'), ['1', '2', '2'], {'1': 'one', '2': 'two'})
    ])
    def test_sizes(self, config: CodingSchemeConfig, codes: List[str], desc: Dict[str, str]):
        """
        Test the consistency between scheme components, in their size, and mapping correctness.

        This method adds a problematic scheme to test error handling.
        The codes, description, and index collections should all have the same sizes,
        codes should be unique, and mapping should be correct and 1-to-1.
        FlatScheme constructor raises either an AssertionError or KeyError when provided with invalid input.
        """
        with pytest.raises((AssertionError, KeyError)):
            FlatScheme(config=config, codes=codes, desc=desc)

    def test_codes(self, primitive_flat_scheme_kwarg):
        """
        Test the initialization and retrieval of codes in the FlatScheme class.

        This method performs the following steps:
        1. Registers the scheme using the `register_scheme` method of the FlatScheme class.
        2. Retrieves the scheme using the `from_name` method of the FlatScheme class.
        3. Asserts that the retrieved scheme's codes match the provided keyword arguments' codes.

        This test ensures that the codes are properly initialized and can be retrieved correctly.

        """
        scheme = FlatScheme(**primitive_flat_scheme_kwarg)
        FlatScheme.register_scheme(scheme)
        assert FlatScheme.from_name(primitive_flat_scheme_kwarg['config'].name).codes == primitive_flat_scheme_kwarg[
            'codes']

    def test_desc(self, primitive_flat_scheme_kwarg):
        """
        Test the initialization and description of the FlatScheme.

        Same as the test_codes method, but for the description.
        """
        scheme = FlatScheme(**primitive_flat_scheme_kwarg)
        FlatScheme.register_scheme(scheme)
        assert FlatScheme.from_name(primitive_flat_scheme_kwarg['config'].name).desc == primitive_flat_scheme_kwarg[
            'desc']

    def test_name(self, primitive_flat_scheme_kwarg):
        """
        Test case to verify if the name attribute of the FlatScheme instance
        matches the name attribute specified in the scheme configuration.
        """
        scheme = FlatScheme(**primitive_flat_scheme_kwarg)
        FlatScheme.register_scheme(scheme)
        assert scheme.name == primitive_flat_scheme_kwarg['config'].name

    def test_index2code(self, primitive_flat_scheme):
        """
        Test the index to code mapping in the coding scheme.

        It tests if the index to code mapping is correct and respects the codes order.
        """

        assert all(c == primitive_flat_scheme.codes[i] for i, c in primitive_flat_scheme.index2code.items())

    def test_index2code(self, primitive_flat_scheme):
        """
        Test the mapping of index to description in the coding scheme.

        Iterates over the scheme_kwargs and creates a FlatScheme object using each set of keyword arguments.
        Then, it checks if the description for each code in the scheme matches the description obtained from the index.
        """

        assert all(desc == primitive_flat_scheme.desc[primitive_flat_scheme.codes[i]] for i, desc in
                   primitive_flat_scheme.index2desc.items())

    def test_search_regex(self):
        """
        Test the search_regex method of the FlatScheme class.

        This method tests the search_regex method of the FlatScheme class by creating a FlatScheme object
        with a specific coding scheme configuration and performing various search operations using the
        search_regex method. It asserts the expected results for each search operation.

        """
        # Arrange
        scheme = FlatScheme(CodingSchemeConfig('simple_searchable'),
                            codes=['1', '3'],
                            desc={'1': 'one', '3': 'pancreatic cAnCeR'})
        # Act & Assert
        assert scheme.search_regex('cancer') == {'3'}
        assert scheme.search_regex('one') == {'1'}
        assert scheme.search_regex('pancreatic') == {'3'}
        assert scheme.search_regex('cAnCeR') == {'3'}
        assert scheme.search_regex('x') == set()

    # def test_mapper_to(self):
    #     self.fail()

    # def test_codeset2vec(self):
    #     self.fail()

    # def test_empty_vector(self):
    #     self.fail()

    # def test_supported_targets(self):
    #     self.fail()

    def test_as_dataframe(self, primitive_flat_scheme):
        """
        Test the `as_dataframe` method of the FlatScheme class.

        This method tests whether the `as_dataframe` method returns a DataFrame with the expected structure and values.
        It checks if the index of the DataFrame matches the index values of the scheme, and if the columns of the DataFrame
        are 'code' and 'desc'. It also verifies if the scheme object is equal to a new FlatScheme object created with the
        same configuration, codes, descriptions, and index.
        """
        df = primitive_flat_scheme.as_dataframe()
        assert set(df.index) == set(primitive_flat_scheme.index.values())
        assert set(df.columns) == {'code', 'desc'}
        codes = df.code.tolist()
        desc = df.set_index('code')['desc'].to_dict()
        assert primitive_flat_scheme == FlatScheme(config=primitive_flat_scheme.config, codes=codes, desc=desc)

# class TestCommonCodingSchemes:
#     def setUp(self):
#         pass
#
#     def test_codes_uniqueness(self):
#         for scheme in self.scheme_set:
#             codes = scheme.codes
#             self.assertCountEqual(codes,
#                                   set(codes),
#                                   msg=f'Problematic Scheme: {scheme.name}')
#
#     def test_code_index_desc_support(self: TestCase):
#         for scheme in self.scheme_set:
#             index = scheme.index
#             codes = scheme.codes
#             desc = scheme.desc
#             msg = f'Problematic Scheme: {scheme.name}'
#
#             self.assertTrue(len(codes) > 0, msg=msg)
#             self.assertCountEqual(codes, index.keys(), msg=msg)
#             self.assertCountEqual(codes, desc.keys(), msg=msg)
#             self.assertCountEqual(index.values(),
#                                   list(range(len(index))),
#                                   msg=msg)
#
#     def test_codes_subsets_dag(self):
#         for s in self.scheme_set:
#             if not isinstance(s, HierarchicalScheme):
#                 continue
#
#             self.assertTrue(len(s.codes) <= len(s.dag_codes))
#             self.assertTrue(
#                 set(map(s.code2dag.get, s.codes)).issubset(set(s.dag_codes)))
#             self.assertEqual(list(map(s.code2dag.get, s.codes)),
#                              s.dag_codes[:len(s.codes)])
#
#     def test_bfs_vs_dfs(self):
#         for s in self.scheme_set:
#
#             if not isinstance(s, HierarchicalScheme):
#                 continue
#
#             rng = random.Random(42)
#             some_codes = rng.sample(s.dag_codes, 15)
#
#             for code in some_codes:
#                 ancestors_bfs = s.code_ancestors_bfs(code)
#                 ancestors_dfs = s.code_ancestors_dfs(code)
#
#                 self.assertCountEqual(ancestors_bfs, ancestors_dfs)
#
#                 successors_bfs = s.code_successors_bfs(code)
#                 successors_dfs = s.code_successors_dfs(code)
#
#                 self.assertCountEqual(successors_bfs, successors_dfs)
#
#     def test_code_mappers(self):
#         log_dir = os.path.join(_DIR, 'logs')
#         Path(log_dir).mkdir(parents=True, exist_ok=True)
#         for s1 in self.scheme_set:
#             for s2 in self.scheme_set:
#
#                 m = s1.mapper_to(s2)
#
#                 if (type(s1),
#                     type(s2)) not in load_maps and type(s1) != type(s2):
#                     self.assertTrue(
#                         m is None,
#                         msg=
#                         f"Mapping {s1.name}->{s2.name} actually not included in load_maps dictionary then mapper should be None"
#                     )
#
#                 if m is None: continue
#
#                 with self.subTest(f"M: {m}"):
#                     self.assertTrue(all(type(c) == str for c in m))
#                     m_range = set().union(*m.values())
#                     self.assertTrue(all(type(c) == str for c in m_range))
#                     m.log_unrecognised_domain(
#                         f'{log_dir}/{m}_unrecognised_domain.json')
#                     m.log_unrecognised_range(
#                         f'{log_dir}/{m}_unrecognised_range.json')
#                     m.log_uncovered_source_codes(
#                         f'{log_dir}/{m}_uncovered_source.json')
#
