import unittest
import random
import os
from pathlib import Path

from icenode.ehr.coding_scheme import (DxCCS, DxFlatCCS, DxICD10, DxICD9,
                                       PrCCS, PrFlatCCS, PrICD10, PrICD9,
                                       HierarchicalScheme, CodeMapper,
                                       code_scheme as C)

_DIR = os.path.dirname(__file__)


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
        cls.scheme = C['dx_flatccs']


class TestPrFlatCCS(AbstractSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['pr_flatccs']


class TestDxCCS(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['dx_ccs']


class TestPrCCS(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['pr_ccs']


class TestDxICD9(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['dx_icd9']


class TestPrICD9(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['pr_icd9']


class TestDxICD10(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['dx_icd10']


class TestPrICD10(HierarchicalSchemeTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scheme = C['pr_icd10']


class TestCodeMappers(unittest.TestCase):

    def test_code_mappers(self):
        log_dir = os.path.join(_DIR, 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for s1 in C:
            for s2 in C:
                m = CodeMapper.get_mapper(s1, s2)
                if not m: continue
                with self.subTest(f"M: {m}"):
                    self.assertTrue(all(type(c) == str for c in m))
                    m_range = set().union(*m.values())
                    self.assertTrue(all(type(c) == str for c in m_range))
                    m.log_unrecognised_domain(
                        f'{log_dir}/{m}_unrecognised_domain.json')
                    m.log_unrecognised_range(
                        f'{log_dir}/{m}_unrecognised_range.json')
                    m.log_uncovered_source_codes(
                        f'{log_dir}/{m}_uncovered_source.json')


if __name__ == '__main__':
    unittest.main()
