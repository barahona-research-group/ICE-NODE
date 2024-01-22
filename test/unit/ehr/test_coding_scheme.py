import os
from unittest import TestCase, main as main_test

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
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name), scheme)

        with self.assertRaises(KeyError):
            # Unregistered scheme
            FlatScheme.from_name('42')

    def test_register_scheme(self):
        # First, test that the register_scheme method works.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name), scheme)

        # Second, test that the register_scheme method raises an error when
        # the scheme is already registered.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            with self.assertRaises(AssertionError):
                FlatScheme.register_scheme(scheme)

    def test_register_scheme_loader(self):
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme_loader(scheme.config.name, lambda: FlatScheme.register_scheme(scheme))
            self.assertEqual(FlatScheme.from_name(scheme.config.name), scheme)

            with self.assertRaises(AssertionError):  # Scheme already registered
                FlatScheme.register_scheme_loader(scheme.config.name, lambda: FlatScheme.register_scheme(scheme))

            with self.assertRaises(AssertionError):
                FlatScheme.register_scheme(scheme)

    def test_registered_schemes(self):

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
    def test_type_error(self, config, codes, desc, index):
        # Add a problematic scheme to test error handling
        # code type should be str, not int.

        with self.assertRaises((AssertionError, KeyError)):
            FlatScheme(config=config, codes=codes, desc=desc, index=index)

    @parameterized.expand([
        (CodingSchemeConfig('p_codes'), ['1', '3'], {'1': 'one'}, {'1': 0}),
        (CodingSchemeConfig('p_desc'), ['3'], {'3': 'three', '1': 'one'}, {'1': 0}),
        (CodingSchemeConfig('p_idx'), ['1'], {'1': 'one'}, {'1': 0, '2': 0}),
    ])
    def test_sizes(self, config, codes, desc, index):
        # Add a problematic scheme to test error handling
        # code type should be str, not int.
        with self.assertRaises((AssertionError, KeyError)):
            FlatScheme(config=config, codes=codes, desc=desc, index=index)

    def test_codes(self):
        # First, test that the codes is properly initialized.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name).codes, scheme_kwargs['codes'])

    def test_index(self):
        # First, test that the index is properly initialized.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name).codes, scheme_kwargs['codes'])

    def test_desc(self):
        # First, test that the desc is properly initialized.
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            FlatScheme.register_scheme(scheme)
            self.assertEqual(FlatScheme.from_name(scheme.config.name).desc, scheme_kwargs['desc'])

    def test_name(self):
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            self.assertEqual(scheme.name, scheme_kwargs['config'].name)

    def test_index2code(self):
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            self.assertTrue(all(c == scheme.codes[i] for i, c in scheme.index2code.items()))

    def test_index2desc(self):
        for scheme_kwargs in self.schemes_kwargs:
            scheme = FlatScheme(**scheme_kwargs)
            self.assertTrue(all(desc == scheme.desc[scheme.codes[i]] for i, desc in scheme.index2desc.items()))

    def test_search_regex(self):
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
