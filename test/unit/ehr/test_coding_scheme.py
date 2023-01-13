import unittest
import random
import os
from pathlib import Path

from lib.ehr.coding_scheme import (load_maps, HierarchicalScheme)

_DIR = os.path.dirname(__file__)


class SchemeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        scheme_cls_set = set(s for pair in load_maps for s in pair)
        cls.scheme_set = set(s() for s in scheme_cls_set)

    def setUp(self):
        pass

    def test_codes_uniqueness(self):
        for scheme in self.scheme_set:
            codes = scheme.codes
            self.assertCountEqual(codes,
                                  set(codes),
                                  msg=f'Problematic Scheme: {scheme.name}')

    def test_codes_sorted(self: unittest.TestCase):
        for scheme in self.scheme_set:
            codes = scheme.codes
            self.assertEqual(codes,
                             sorted(codes),
                             msg=f'Problematic Scheme: {scheme.name}')

    def test_code_index_desc_support(self: unittest.TestCase):
        for scheme in self.scheme_set:

            index = scheme.index
            codes = scheme.codes
            desc = scheme.desc
            msg = f'Problematic Scheme: {scheme.name}'

            self.assertTrue(len(codes) > 0, msg=msg)
            self.assertCountEqual(codes, index.keys(), msg=msg)
            self.assertCountEqual(codes, desc.keys(), msg=msg)
            self.assertCountEqual(index.values(),
                                  list(range(len(index))),
                                  msg=msg)

    def test_codes_subsets_dag(self):
        for s in self.scheme_set:
            if not isinstance(s, HierarchicalScheme):
                continue

            self.assertTrue(len(s.codes) <= len(s.dag_codes))
            self.assertTrue(
                set(map(s.code2dag.get, s.codes)).issubset(set(s.dag_codes)))
            self.assertEqual(list(map(s.code2dag.get, s.codes)),
                             s.dag_codes[:len(s.codes)])

    def test_bfs_vs_dfs(self):
        for s in self.scheme_set:

            if not isinstance(s, HierarchicalScheme):
                continue

            rng = random.Random(42)
            some_codes = rng.sample(s.dag_codes, 15)

            for code in some_codes:

                ancestors_bfs = s.code_ancestors_bfs(code)
                ancestors_dfs = s.code_ancestors_dfs(code)

                self.assertCountEqual(ancestors_bfs, ancestors_dfs)

                successors_bfs = s.code_successors_bfs(code)
                successors_dfs = s.code_successors_dfs(code)

                self.assertCountEqual(successors_bfs, successors_dfs)

    def test_code_mappers(self):
        log_dir = os.path.join(_DIR, 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        for s1 in self.scheme_set:
            for s2 in self.scheme_set:

                m = s1.mapper_to(s2)

                if (type(s1),
                        type(s2)) not in load_maps and type(s1) != type(s2):

                    self.assertTrue(
                        m is None,
                        msg=
                        f"Mapping {s1.name}->{s2.name} actually not included in load_maps dictionary then mapper should be None"
                    )

                if m is None: continue

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
