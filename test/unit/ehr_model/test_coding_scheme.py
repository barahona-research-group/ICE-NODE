import unittest
import random

from icenode.ehr_model.coding_scheme import (DxCCS, DxFlatCCS, DxICD10, DxICD9,
                                             PrCCS, PrFlatCCS, PrICD10, PrICD9)


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
        cls.scheme = DxFlatCCS()


class TestPrFlatCCS(AbstractSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = PrFlatCCS()


class TestDxCCS(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = DxCCS()


class TestPrCCS(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = PrCCS()


class TestDxICD9(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = DxICD9()


class TestPrICD9(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = PrICD9()


class TestDxICD10(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = DxICD10()


class TestPrICD10(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = PrICD10()
