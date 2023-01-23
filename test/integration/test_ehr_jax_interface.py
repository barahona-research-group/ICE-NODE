"""Integration Test for EHR data model"""

import unittest
import logging

from lib.ehr import (Subject_JAX, MIMIC3EHRDataset, MIMIC4EHRDataset)
from lib.ehr.coding_scheme import (DxICD9, DxICD10, DxCCS, DxFlatCCS,
                                   NullScheme)
from lib.ehr.outcome import OutcomeExtractor


class TestSubject_JAX(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        m3_dataset = MIMIC3EHRDataset.from_meta_json(
            'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')
        m4_dataset = MIMIC4EHRDataset.from_meta_json(
            'test/integration/fixtures/synthetic_mimic/mimic4_syn_meta.json')

        cls.interfaces = []
        for dx_scheme in [DxICD9, DxICD10, DxCCS, DxFlatCCS]:
            code_scheme = {
                'dx': dx_scheme(),
                'outcome': OutcomeExtractor('dx_flatccs_filter_v1'),
                'pr': NullScheme()
            }
            interface = Subject_JAX.from_dataset(m3_dataset, code_scheme)
            cls.interfaces.append(interface)

        for dx_scheme in [DxICD9, DxICD10]:
            code_scheme = {
                'dx': dx_scheme(),
                'outcome': OutcomeExtractor('dx_icd9_filter_v1'),
                'pr': NullScheme()
            }
            interface = Subject_JAX.from_dataset(m4_dataset, code_scheme)
            cls.interfaces.append(interface)

    def test_split(self):
        IF = self.interfaces[0]
        train_ids, valid_ids, test_ids = IF.random_splits(split1=0.7,
                                                          split2=0.85,
                                                          random_seed=42)
        self.assertCountEqual(
            set(train_ids) | set(valid_ids) | set(test_ids),
            IF.subjects.keys())

    def test_code_frequency_paritions(self):
        IFs = self.interfaces

        for IF in IFs:
            splits = IF.random_splits(split1=0.7, split2=0.85, random_seed=42)
            train_ids, valid_ids, test_ids = splits
            with self.subTest(msg=f"{IF.dx_mappers}"):
                for percentile_range in [2, 5, 10, 20, 25, 33, 50, 100]:
                    code_partitions = IF.outcome_by_percentiles(
                        percentile_range, train_ids)
                    # Assert that union of all partitions recovers all the codes.
                    self.assertEqual(
                        set(IF.outcome_extractor.index.values()),
                        set.union(*code_partitions))

                    # Assert that no intersection between the partitions
                    for i in range(len(code_partitions)):
                        for j in range(i + 1, len(code_partitions)):
                            self.assertCountEqual(
                                code_partitions[i] & code_partitions[j], set())

    def test_tabular(self):
        subjects_ids = list(self.interfaces[0].keys())
        X, y = self.interfaces[0].tabular_features(subjects_ids)
        self.assertTrue(set(X.flatten()) & set(y.flatten()) == {0, 1})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
