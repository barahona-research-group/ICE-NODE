import os
from typing import Dict, List
from unittest import TestCase, main as main_test
from unittest.mock import patch

from parameterized import parameterized

from lib.ehr import (CodingScheme, FlatScheme, CodingSchemeConfig, setup_icd, setup_mimic, setup_cprd, OutcomeExtractor)

_DIR = os.path.dirname(__file__)


class TestFlatScheme(TestCase):
    @classmethod
    def setUpClass(cls):
        codes100 = list(f'code_{i}' for i in range(100))
        desc100 = dict(zip(codes100, codes100))
        index100 = dict(zip(codes100, range(100)))
        CodingScheme.unregister_schemes()
        CodingScheme.unregister_scheme_loaders()
        OutcomeExtractor.unregister_schemes()
        OutcomeExtractor.unregister_scheme_loaders()

        schemes_kwargs = [{'config': CodingSchemeConfig('one'),
                           'codes': ['1'],
                           'desc': {'1': 'one'},
                           'index': {'1': 0}},
                          {'config': CodingSchemeConfig('zero'),
                           'codes': [],
                           'desc': dict(),
                           'index': dict()},
                          {'config': CodingSchemeConfig('100'),
                           'codes': codes100,
                           'desc': desc100,
                           'index': index100}]
        cls.schemes_kwargs = schemes_kwargs

    def test_from_name(self):
        """
        Test the `from_name` method of the FlatScheme class.

        This method iterates over the `schemes_kwargs` list and creates a FlatScheme object
        using the provided keyword arguments. It then registers the scheme and asserts that
        calling `from_name` with the scheme's name returns the same scheme object.

        If a KeyError is raised, it means that the scheme is not registered and the test passes.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name), scheme)

        with self.assertRaises(KeyError):
            # Unregistered scheme
            FlatScheme.from_name('42')

    @patch('logging.warning')
    def test_register_scheme(self, mock_warning):
        """
        Test the register_scheme method.

        This method tests two scenarios:
        1. It tests that the register_scheme method works by registering a scheme and then
           asserting that the registered scheme can be retrieved using its name.
        2. It tests that the register_scheme method logs a warning when trying to
           register a scheme that is already registered with the same name and content.
        """
        # First, test that the register_scheme method works.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name), scheme)

        # Second, test that the register_scheme method raises an error when
        # the scheme is already registered.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            mock_warning.reset_mock()
            FlatScheme.register_scheme(scheme)
            mock_warning.assert_called_once()

    def test_scheme_equality(self):
        """
        Test the equality of schemes.

        This test iterates over the `schemes_kwargs` list and creates two FlatScheme objects
        using the provided keyword arguments. It then asserts that the two schemes are equal.
        It then mutates the description and index of one of the schemes and asserts that the two
        schemes are not equal.
        """

        for scheme_kwargs in self.schemes_kwargs:
            scheme1 = FlatScheme(**scheme_kwargs)
            scheme2 = FlatScheme(**scheme_kwargs)
            self.assertEqual(scheme1, scheme2)

        for scheme_kwargs in self.schemes_kwargs:
            scheme1 = FlatScheme(**scheme_kwargs)

            if len(scheme1) > 0:
                scheme_kwargs_desc_mutated = scheme_kwargs.copy()

                scheme_kwargs_desc_mutated['desc'] = {code: f'{desc} muted' for code, desc in
                                                      scheme_kwargs['desc'].items()}
                scheme2 = FlatScheme(**scheme_kwargs_desc_mutated)
                self.assertNotEqual(scheme1, scheme2)

    @patch('logging.warning')
    def test_register_scheme_loader(self, mock_warning):
        """
        Test case for registering a scheme loader and verifying the scheme registration.

        This test iterates over a list of scheme keyword arguments and performs the following steps:
        1. Creates a FlatScheme instance using the scheme keyword arguments.
        2. Registers a scheme loader for the scheme's name using a lambda function.
        3. Asserts that the scheme can be retrieved using the scheme's name.
        4. Asserts that attempting to register the same scheme loader again logs a warning if it has the same name with
        matching content.
        5. Asserts that attempting to register the same scheme again logs a warning.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme_loader(scheme.config.name, lambda: FlatScheme.register_scheme(scheme))
            self.assertEqual(FlatScheme.from_name(scheme.config.name), scheme)

            mock_warning.reset_mock()
            FlatScheme.register_scheme_loader(scheme.config.name, lambda: FlatScheme.register_scheme(scheme))
            mock_warning.assert_called_once()

            mock_warning.reset_mock()
            FlatScheme.register_scheme(scheme)
            mock_warning.assert_called_once()

            # If the scheme is registered with the same name but different content, an AssertionError should be raised.
            if len(scheme) > 0:
                scheme_kwargs_desc_mutated = scheme_kwargs.copy()
                scheme_kwargs_desc_mutated['desc'] = {code: f'{desc} muted' for code, desc in
                                                      scheme_kwargs['desc'].items()}

                with self.assertRaises(AssertionError):
                    FlatScheme.register_scheme(FlatScheme(**scheme_kwargs_desc_mutated))

    def test_registered_schemes(self):
        """
        Test case to verify the behavior of registered coding schemes.

        This test case checks the following:
        - The initial set of available coding schemes is empty.
        - After each setup (setup_cprd, setup_mimic, setup_icd), the number of available coding schemes increases.
        - Each registered coding scheme is an instance of FlatScheme object.
        - The name of the instantiated FlatScheme object matches the registered coding scheme.
        """

        self.assertSetEqual(CodingScheme.available_schemes(), set())
        count = 0
        for setup in (setup_cprd, setup_mimic, setup_icd):
            setup()
            self.assertGreater(len(CodingScheme.available_schemes()), count)
            count = len(CodingScheme.available_schemes())

        for registered_scheme in CodingScheme.available_schemes():
            scheme = FlatScheme.from_name(registered_scheme)
            self.assertIsInstance(scheme, FlatScheme)
            self.assertEqual(scheme.name, registered_scheme)

    @parameterized.expand(
        [(CodingSchemeConfig('p_codes'), [1], {'1': 'one'}, {'1': 0}),
         (CodingSchemeConfig('p_desc'), ['1'], {1: 'one'}, {'1': 0}),
         (CodingSchemeConfig('p_idx'), ['1'], {'1': 'one'}, {1: 0}),
         (CodingSchemeConfig('p_desc'), ['1'], {'1': 5}, {'1': 0})])
    def test_type_error(self, config: CodingSchemeConfig, codes: List[str], desc: Dict[str, str],
                        index: Dict[str, int]):
        """
        Test for type error handling in the FlatScheme constructor.

        This test adds a problematic scheme to test the error handling of the constructor.
        The code and description types should be strings, not integers. The test expects an AssertionError
        or KeyError to be raised.
        """
        with self.assertRaises((AssertionError, KeyError)):
            FlatScheme(config=config, codes=codes, desc=desc, index=index)

    @parameterized.expand([
        (CodingSchemeConfig('p_codes'), ['1', '3'], {'1': 'one'}, {'1': 0}),
        (CodingSchemeConfig('p_desc'), ['3'], {'3': 'three', '1': 'one'}, {'1': 0}),
        (CodingSchemeConfig('p_idx'), ['1'], {'1': 'one'}, {'1': 0, '2': 0}),
        (CodingSchemeConfig('p_duplicate_codes'), ['1', '2', '2'], {'1': 'one', '2': 'two'}, {'1': 0, '2': 1})
    ])
    def test_sizes(self, config: CodingSchemeConfig, codes: List[str], desc: Dict[str, str], index: Dict[str, int]):
        """
        Test the consistency between scheme components, in their size, and mapping correctness.

        This method adds a problematic scheme to test error handling.
        The codes, description, and index collections should all have the same sizes,
        codes should be unique, and mapping should be correct and 1-to-1.
        FlatScheme constructor raises either an AssertionError or KeyError when provided with invalid input.
        """
        with self.assertRaises((AssertionError, KeyError)):
            FlatScheme(config=config, codes=codes, desc=desc, index=index)



    def test_codes(self):
        """
        Test the initialization and retrieval of codes in the FlatScheme class.

        This method iterates over a list of scheme keyword arguments and performs the following steps:
        1. Initializes a FlatScheme object with the provided keyword arguments.
        2. Registers the scheme using the `register_scheme` method of the FlatScheme class.
        3. Retrieves the scheme using the `from_name` method of the FlatScheme class.
        4. Asserts that the retrieved scheme's codes match the provided keyword arguments' codes.

        This test ensures that the codes are properly initialized and can be retrieved correctly.

        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name).codes, scheme_kwargs['codes'])

    def test_index(self):
        """
        Test the initialization and retrieval of index in FlatScheme.

        Same as the test_codes method, but for the index.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name).codes, scheme_kwargs['codes'])

    def test_desc(self):
        """
        Test the initialization and description of the FlatScheme.

        Same as the test_codes method, but for the description.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name).desc, scheme_kwargs['desc'])

    def test_name(self):
        """
        Test case to verify if the name attribute of the FlatScheme instance
        matches the name attribute specified in the scheme configuration.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            self.assertEqual(scheme.name, scheme_kwargs['config'].name)

    def test_index2code(self):
        """
        Test the index to code mapping in the coding scheme.

        It tests if the index to code mapping is correct and respects the codes order.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            self.assertTrue(all(c == scheme.codes[i] for i, c in scheme.index2code.items()))

    def test_index2desc(self):
        """
        Test the mapping of index to description in the coding scheme.

        Iterates over the scheme_kwargs and creates a FlatScheme object using each set of keyword arguments.
        Then, it checks if the description for each code in the scheme matches the description obtained from the index.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            self.assertTrue(all(desc == scheme.desc[scheme.codes[i]] for i, desc in scheme.index2desc.items()))

    def test_search_regex(self):
        """
        Test the search_regex method of the FlatScheme class.

        This method tests the search_regex method of the FlatScheme class by creating a FlatScheme object
        with a specific coding scheme configuration and performing various search operations using the
        search_regex method. It asserts the expected results for each search operation.

        """
        scheme = FlatScheme(CodingSchemeConfig('p_codes'),
                            codes=['1', '3'],
                            desc={'1': 'one', '3': 'pancreatic cAnCeR'},
                            index={'1': 0, '3': 1})
        self.assertSetEqual(scheme.search_regex('cancer'), {'3'})
        self.assertSetEqual(scheme.search_regex('one'), {'1'})
        self.assertSetEqual(scheme.search_regex('pancreatic'), {'3'})
        self.assertSetEqual(scheme.search_regex('cAnCeR'), {'3'})
        self.assertSetEqual(scheme.search_regex('x'), set())

    # def test_mapper_to(self):
    #     self.fail()

    # def test_codeset2vec(self):
    #     self.fail()

    # def test_empty_vector(self):
    #     self.fail()

    # def test_supported_targets(self):
    #     self.fail()

    def test_as_dataframe(self):
        """
        Test the `as_dataframe` method of the FlatScheme class.

        This method tests whether the `as_dataframe` method returns a DataFrame with the expected structure and values.
        It checks if the index of the DataFrame matches the index values of the scheme, and if the columns of the DataFrame
        are 'code' and 'desc'. It also verifies if the scheme object is equal to a new FlatScheme object created with the
        same configuration, codes, descriptions, and index.
        """
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            df = scheme.as_dataframe()
            self.assertSetEqual(set(df.index), set(scheme.index.values()))
            self.assertSetEqual(set(df.columns), {'code', 'desc'})
            codes = df.code.tolist()
            desc = df.set_index('code')['desc'].to_dict()
            index = dict(zip(df.code, df.index))
            self.assertEqual(scheme, FlatScheme(config=scheme.config, codes=codes, desc=desc, index=index))

    def tearDown(self):
        CodingScheme.unregister_schemes()
        CodingScheme.unregister_scheme_loaders()
        OutcomeExtractor.unregister_schemes()
        OutcomeExtractor.unregister_scheme_loaders()


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

if __name__ == '__main__':
    main_test()
