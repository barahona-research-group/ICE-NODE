import unittest
import random

from icenode.ehr.coding_scheme import (DxCCS, DxFlatCCS, DxICD10, DxICD9,
                                       PrCCS, PrFlatCCS, PrICD10, PrICD9,
                                       code_scheme, HierarchicalScheme)


class AbstractSchemeTests(object):

    @classmethod
    def setUpClass(cls):
        raise RuntimeError('Unreachable')

    def setUp(self):
        pass

    def test_codes_uniqueness(self):
        codes = self.scheme.codes
        self.assertCountEqual(codes, set(codes))

    def test_codes_sorted(self: unittest.TestCase):
        codes = self.scheme.codes
        self.assertEqual(codes, sorted(codes))

    def test_code_index_desc_support(self: unittest.TestCase):
        self.maxDiff = None
        index = self.scheme.index
        codes = self.scheme.codes
        desc = self.scheme.desc

        self.assertTrue(len(codes) > 0)
        self.assertCountEqual(codes, index.keys())
        self.assertCountEqual(codes, desc.keys())
        self.assertCountEqual(index.values(), list(range(len(index))))


class HierarchicalSchemeTests(AbstractSchemeTests):

    @classmethod
    def setUpClass(cls):
        # Unreachable, should be overriden.
        raise RuntimeError('Unreachable')

    def setUp(self):
        pass

    def test_codes_subsets_dag(self):
        s = self.scheme

        self.assertTrue(len(s.codes) <= len(s.dag_codes))
        self.assertTrue(
            set(map(s.code2dag.get, s.codes)).issubset(set(s.dag_codes)))
        self.assertEqual(list(map(s.code2dag.get, s.codes)),
                         s.dag_codes[:len(s.codes)])

    def test_bfs_vs_dfs(self):
        s = self.scheme
        rng = random.Random(42)
        some_codes = rng.sample(s.dag_codes, 15)

        for code in some_codes:

            ancestors_bfs = s.code_ancestors_bfs(code)
            ancestors_dfs = s.code_ancestors_dfs(code)

            self.assertCountEqual(ancestors_bfs, ancestors_dfs)

            successors_bfs = s.code_successors_bfs(code)
            successors_dfs = s.code_successors_dfs(code)

            self.assertCountEqual(successors_bfs, successors_dfs)


class TestDxFlatCCS(AbstractSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['dx_flatccs']


class TestPrFlatCCS(AbstractSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['pr_flatccs']


class TestDxCCS(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['dx_ccs']


class TestPrCCS(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['pr_ccs']


class TestDxICD9(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['dx_icd9']


class TestPrICD9(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['pr_icd9']


class TestDxICD10(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['dx_icd10']


class TestPrICD10(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = code_scheme['pr_icd10']


class TestConversionCoverage(unittest.TestCase):

    def test_conversion_validity_and_95_coverage(self):

        for s1_k in code_scheme.keys():
            for s2_k in code_scheme.keys():
                s1 = code_scheme[s1_k]
                s2 = code_scheme[s2_k]
                m = s1.maps.get((type(s1), type(s2)))
                if m is not None:
                    with self.subTest(
                            msg=f"{s1_k} to {s2_k}: Target scheme coverage"):
                        trgt_codes = set().union(*m.values())
                        s2_codes = s2.codes
                        if issubclass(type(s2), HierarchicalScheme):
                            s2_codes = list(map(s2.code2dag.get, s2_codes))
                        mappedto_count = len(trgt_codes)
                        mappedto_coverage = len(
                            [c for c in trgt_codes if c in s2_codes])

                        self.assertTrue(
                            mappedto_coverage == mappedto_count,
                            msg=
                            f'coverage {mappedto_coverage / mappedto_count} unmatched: {trgt_codes - set(s2_codes)}'
                        )

                    with self.subTest(
                            msg=f"{s1_k} to {s2_k}: Source scheme coverage"):
                        mappedfrom_count = len(s1.codes)
                        mappedfrom_cover = len(
                            [c for c in s1.codes if c in m and len(m[c]) > 0])
                        self.assertTrue(
                            mappedfrom_cover == mappedfrom_count,
                            msg=
                            f'coverage: {mappedfrom_cover / mappedfrom_count} unmatched: {set(s1.codes) - set(m)}'
                        )
