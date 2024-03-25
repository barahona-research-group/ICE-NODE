import os
from typing import Dict, List, Tuple
from unittest import mock

import pytest
import tables as tb

from lib.ehr import (CodingScheme)
from lib.ehr.coding_scheme import CodingSchemesManager, FrozenDict11, OutcomeExtractor
from lib.ehr.example_schemes.icd import CCSICDSchemeSelection, setup_standard_icd_ccs, CCSICDOutcomeSelection, \
    setup_icd_schemes, setup_icd_outcomes

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
#
# @pytest.fixture(scope='session')
# def icd_schemes_manager():
#     manager = CodingSchemesManager()
#     return setup_standard_icd_ccs(manager)


class TestFlatScheme:
    @pytest.fixture(scope='class', params=[dict(name='one', codes=['1'], desc={'1': 'one'}),
                                           dict(name='zero', codes=[], desc=dict()),
                                           dict(name='100', codes=list(f'code_{i}' for i in range(100)))])
    def primitive_flat_scheme_kwarg(self, request):
        if 'desc' in request.param:
            desc = request.param['desc']
        elif len(request.param['codes']) > 0:
            desc = dict(zip(request.param['codes'], request.param['codes']))
        else:
            desc = dict()
        return dict(name=request.param['name'], codes=tuple(sorted(request.param['codes'])),
                    desc=FrozenDict11.from_dict(desc))

    @pytest.fixture(scope='class')
    def primitive_flat_scheme(self, primitive_flat_scheme_kwarg) -> CodingScheme:
        return CodingScheme(**primitive_flat_scheme_kwarg)

    @pytest.fixture(scope='class')
    def scheme_manager(self, primitive_flat_scheme: CodingScheme) -> CodingSchemesManager:
        return CodingSchemesManager().add_scheme(primitive_flat_scheme)

    def test_from_name(self, primitive_flat_scheme: CodingScheme, scheme_manager: CodingSchemesManager):
        assert scheme_manager.scheme[primitive_flat_scheme.name].equals(primitive_flat_scheme)

        with pytest.raises(KeyError):
            # Unregistered scheme
            scheme_manager.scheme['42']

    @pytest.mark.parametrize("codes", [('A', 'B', 'C', 'C'),
                                       ('A', 'B', 'C', 'B'),
                                       ('A', 'A', 'A', 'A')])
    def test_codes_uniqueness(self, codes):
        with pytest.raises(AssertionError) as excinfo:
            CodingScheme(name='test', codes=tuple(sorted(codes)), desc=FrozenDict11.from_dict({c: c for c in codes}))
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
        manager = CodingSchemesManager().add_scheme(primitive_flat_scheme)
        assert manager.scheme[primitive_flat_scheme.name].equals(primitive_flat_scheme)

        # Second, test that the register_scheme method raises an error when
        # the scheme is already registered.
        with mock.patch('logging.warning') as mocker:
            manager.add_scheme(primitive_flat_scheme)
            mocker.assert_called_once()

    def test_scheme_equality(self, primitive_flat_scheme):
        """
        Test the equality of schemes.

        This test asserts that a scheme equal to its deepcopy.
        It then mutates the description and index of one of the schemes and asserts that the two
        schemes are not equal.
        """
        assert primitive_flat_scheme.equals(primitive_flat_scheme)

        if len(primitive_flat_scheme) > 0:
            desc_mutated = FrozenDict11.from_dict(
                {code: f'{desc} muted' for code, desc in primitive_flat_scheme.desc.items()})
            mutated_scheme = CodingScheme(name=primitive_flat_scheme.name,
                                          codes=primitive_flat_scheme.codes,
                                          desc=desc_mutated)
            assert not primitive_flat_scheme.equals(mutated_scheme)

    # def test_register_scheme_loader(self, primitive_flat_scheme):
    #     """
    #     Test case for registering a scheme loader and verifying the scheme registration.
    #
    #     This test performs the following steps:
    #     1. Registers a scheme loader for the scheme's name using a lambda function.
    #     2. Asserts that the scheme can be retrieved using the scheme's name.
    #     3. Asserts that attempting to register the same scheme loader again logs a warning if it has the same name with
    #     matching content.
    #     4. Asserts that attempting to register the same scheme again logs a warning.
    #     """
    #
    #     CodingScheme.register_scheme_loader(primitive_flat_scheme.name,
    #                                         lambda: CodingScheme.register_scheme(primitive_flat_scheme))
    #     assert CodingScheme.from_name(primitive_flat_scheme.name) is primitive_flat_scheme
    #
    #     with mock.patch('logging.warning') as mocker:
    #         CodingScheme.register_scheme_loader(primitive_flat_scheme.name,
    #                                             lambda: CodingScheme.register_scheme(primitive_flat_scheme))
    #         mocker.assert_called_once()
    #     with mock.patch('logging.warning') as mocker:
    #         CodingScheme.register_scheme(primitive_flat_scheme)
    #         mocker.assert_called_once()
    #
    #     # If the scheme is registered with the same name but different content, an AssertionError should be raised.
    #     if len(primitive_flat_scheme) > 0:
    #         desc_mutated = {code: f'{desc} muted' for code, desc in primitive_flat_scheme.desc.items()}
    #         mutated_scheme = CodingScheme(name=primitive_flat_scheme.name,
    #                                       codes=primitive_flat_scheme.codes,
    #                                       desc=desc_mutated)
    #
    #         with pytest.raises(AssertionError):
    #             CodingScheme.register_scheme(mutated_scheme)

    @pytest.fixture(params=[('dx_icd10',), ('dx_icd9',), ('pr_icd9',), ('dx_flat_icd10',), ('pr_flat_icd10',),
                            ('dx_ccs', 'dx_icd9'), ('pr_ccs', 'pr_icd9'), ('dx_flat_ccs', 'dx_icd9'),
                            ('pr_flat_ccs', 'pr_icd9')],
                    ids=lambda x: '_'.join(x), scope='class')
    def scheme_selection(self, request):
        return CCSICDSchemeSelection(**{k: True for k in request.param})

    @pytest.fixture(params=[('dx_icd10', 'dx_icd9'), ('dx_icd9', 'dx_icd10'), ('dx_icd9', 'dx_flat_icd10'),
                            ('dx_flat_icd10', 'dx_icd9'),
                            ('pr_flat_icd10', 'pr_icd9'), ('pr_icd9', 'pr_flat_icd10'), ('dx_ccs', 'dx_icd9'),
                            ('pr_ccs', 'pr_icd9'), ('dx_flat_ccs', 'dx_icd9'), ('pr_flat_ccs', 'pr_icd9')],
                    scope='class',
                    ids=lambda x: '_'.join(x))
    def scheme_pair_selection(self, request):
        return CCSICDSchemeSelection(**{k: True for k in request.param})

    @pytest.fixture(params=['dx_icd9_v1', 'dx_icd9_v2_groups', 'dx_icd9_v3_groups', 'dx_flat_ccs_mlhc_groups',
                            'dx_flat_ccs_v1'], scope='class')
    def outcome_selection(self, request):
        return CCSICDOutcomeSelection(**{request.param: True})

    @pytest.fixture(scope="class")
    def outcome_selection_name(self, outcome_selection: CCSICDOutcomeSelection) -> str:
        (name,) = outcome_selection.flag_set
        return name

    @pytest.fixture(scope="class")
    def selection_names(self, scheme_selection: CCSICDSchemeSelection) -> Tuple[str, ...]:
        return tuple(scheme_selection.flag_set)

    @pytest.fixture(scope="class")
    def icd_ccs_scheme_manager(self, scheme_selection: CCSICDSchemeSelection) -> CodingSchemesManager:
        return setup_icd_schemes(CodingSchemesManager(), scheme_selection)

    @pytest.fixture(scope="class")
    def icd_ccs_outcome_manager_prerequisite(self, outcome_selection: CCSICDOutcomeSelection) -> CodingSchemesManager:
        return setup_icd_schemes(CodingSchemesManager(), CCSICDSchemeSelection(dx_icd9=True, dx_flat_ccs=True))

    @pytest.fixture(scope="class")
    def icd_ccs_map_manager(self, scheme_pair_selection: CCSICDSchemeSelection) -> CodingSchemesManager:
        return setup_standard_icd_ccs(CodingSchemesManager(), scheme_pair_selection, CCSICDOutcomeSelection())

    @pytest.fixture
    def icd_ccs_outcome_manager(self, icd_ccs_outcome_manager_prerequisite: CodingSchemesManager,
                                outcome_selection: CCSICDOutcomeSelection) -> CodingSchemesManager:
        return setup_icd_outcomes(icd_ccs_outcome_manager_prerequisite, outcome_selection)

    def test_icd_ccs_schemes(self, icd_ccs_scheme_manager, selection_names):
        assert len(icd_ccs_scheme_manager.schemes) == len(selection_names)
        assert len(icd_ccs_scheme_manager.scheme) == len(selection_names)
        for name in selection_names:
            assert name in icd_ccs_scheme_manager.scheme
            assert icd_ccs_scheme_manager.scheme[name].name == name
            assert isinstance(icd_ccs_scheme_manager.scheme[name], CodingScheme)

    def test_icd_ccs_outcomes(self, icd_ccs_outcome_manager, outcome_selection_name):
        assert len(icd_ccs_outcome_manager.outcomes) == 1
        assert len(icd_ccs_outcome_manager.outcome) == 1

        assert outcome_selection_name in icd_ccs_outcome_manager.outcome
        assert icd_ccs_outcome_manager.outcome[outcome_selection_name].name == outcome_selection_name
        assert isinstance(icd_ccs_outcome_manager.outcome[outcome_selection_name], OutcomeExtractor)

    def test_icd_ccs_maps(self, icd_ccs_map_manager: CodingSchemesManager,
                          scheme_pair_selection: CCSICDSchemeSelection):
        assert len(scheme_pair_selection.flag_set) == 2
        assert len(icd_ccs_map_manager.maps) == 2
        assert len(icd_ccs_map_manager.map) == 2 + 2  # the two identity maps
        (a, b) = scheme_pair_selection.flag_set
        assert (a, b) in icd_ccs_map_manager.map
        assert (b, a) in icd_ccs_map_manager.map
        m1 = icd_ccs_map_manager.map[(a, b)]
        m2 = icd_ccs_map_manager.map[(b, a)]

        assert m1.source_name == a
        assert m1.target_name == b
        assert m2.source_name == b
        assert m2.target_name == a
        assert m1.source_scheme.name == a
        assert m1.target_scheme.name == b
        assert m2.source_scheme.name == b
        assert m2.target_scheme.name == a
        assert m1.support_ratio > 0.5
        assert m2.support_ratio > 0.5
        assert m1.range_ratio > 0.5
        assert m2.range_ratio > 0.5


    def test_primitive_scheme_serialization(self, primitive_flat_scheme: CodingScheme, tmpdir: str):
        path = f'{tmpdir}/coding_scheme.h5'
        with tb.open_file(path, 'w') as f:
            primitive_flat_scheme.to_hdf_group(f.create_group('/', 'scheme_data'))

        with tb.open_file(path, 'r') as f:
            reloaded = CodingScheme.from_hdf_group(f.root.scheme_data)

        assert primitive_flat_scheme.equals(reloaded)

    def test_icd_ccs_scheme_serialization(self, icd_ccs_scheme_manager: CodingSchemesManager, hf5_write_group: tb.Group,
                                          tmpdir: str):
        path = f'{tmpdir}/coding_schemes.h5'
        with tb.open_file(path, 'w') as f:
            icd_ccs_scheme_manager.to_hdf_group(f.create_group('/', 'context_view'))

        with tb.open_file(path, 'r') as f:
            reloaded = CodingSchemesManager.from_hdf_group(f.root.context_view)

        assert icd_ccs_scheme_manager.equals(reloaded)

    def test_icd_ccs_outcome_serialization(self, icd_ccs_outcome_manager: CodingSchemesManager,
                                           hf5_write_group: tb.Group, tmpdir: str):
        path = f'{tmpdir}/coding_schemes.h5'
        with tb.open_file(path, 'w') as f:
            icd_ccs_outcome_manager.to_hdf_group(f.create_group('/', 'context_view'))

        with tb.open_file(path, 'r') as f:
            reloaded = CodingSchemesManager.from_hdf_group(f.root.context_view).sync()

        assert icd_ccs_outcome_manager.equals(reloaded)

    def test_icd_ccs_map_serialization(self, icd_ccs_map_manager: CodingSchemesManager, hf5_group: tb.Group,
                                       tmpdir: str):
        path = f'{tmpdir}/coding_schemes.h5'
        with tb.open_file(path, 'w') as f:
            icd_ccs_map_manager.to_hdf_group(f.create_group('/', 'context_view'))

        with tb.open_file(path, 'r') as f:
            reloaded = CodingSchemesManager.from_hdf_group(f.root.context_view)

        assert icd_ccs_map_manager.equals(reloaded)


    @pytest.mark.parametrize("name, codes, desc",
                             [('problematic_codes', [1], {'1': 'one'}),
                              ('problematic_desc', ['1'], {1: 'one'}),
                              ('problematic_desc', ['1'], {'1': 5})])
    def test_type_error(self, name: str, codes: List[str], desc: Dict[str, str]):
        """
        Test for type error handling in the FlatScheme constructor.

        This test adds a problematic scheme to test the error handling of the constructor.
        The code and description types should be strings, not integers. The test expects an AssertionError
        or KeyError to be raised.
        """
        with pytest.raises((AssertionError, KeyError)):
            CodingScheme(name=name, codes=tuple(sorted(codes)), desc=FrozenDict11.from_dict(desc))

    @pytest.mark.parametrize("name, codes, desc", [
        ('problematic_desc', ['1', '3'], {'1': 'one'}),
        ('problematic_desc', ['3'], {'3': 'three', '1': 'one'}),
        ('duplicate_codes', ['1', '2', '2'], {'1': 'one', '2': 'two'})
    ])
    def test_sizes(self, name: str, codes: List[str], desc: Dict[str, str]):
        """
        Test the consistency between scheme components, in their size, and mapping correctness.

        This method adds a problematic scheme to test error handling.
        The codes, description, and index collections should all have the same sizes,
        codes should be unique, and mapping should be correct and 1-to-1.
        FlatScheme constructor raises either an AssertionError or KeyError when provided with invalid input.
        """
        with pytest.raises((AssertionError, KeyError)):
            CodingScheme(name=name, codes=tuple(sorted(codes)), desc=FrozenDict11.from_dict(desc))




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
        scheme = CodingScheme(name='simple_searchable',
                              codes=('1', '3'),
                              desc=FrozenDict11.from_dict({'1': 'one', '3': 'pancreatic cAnCeR'}))
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
        assert primitive_flat_scheme == CodingScheme(name=primitive_flat_scheme.name, codes=tuple(sorted(codes)),
                                                     desc=FrozenDict11.from_dict(desc))


class TestSchemeManager:
    pass

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
