"""Integration Test for EHR data model"""

import unittest
import logging

from icenode import ehr


class TestSubject_JAX(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        m3_dataset = ehr.ConsistentSchemeEHRDataset.from_meta_json(
            'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')
        m4_dataset = ehr.AbstractEHRDataset.from_meta_json(
            'test/integration/fixtures/synthetic_mimic/mimic4_syn_meta.json')

        cls.interfaces = []
        for dx_scheme in [s for s in ehr.code_scheme if 'dx' in s]:
            code_scheme = {
                'dx': dx_scheme,
                'dx_outcome': 'dx_flatccs_filter_v1',
                'pr': 'none'
            }
            interface = ehr.Subject_JAX.from_dataset(m3_dataset, code_scheme)
            cls.interfaces.append(interface)

        for dx_scheme in [s for s in ehr.code_scheme if 'dx' in s]:
            code_scheme = {
                'dx': dx_scheme,
                'dx_outcome': 'dx_icd9_filter_v1',
                'pr': 'none'
            }
            interface = ehr.Subject_JAX.from_dataset(m4_dataset, code_scheme)
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
        train_ids, valid_ids, test_ids = IFs[0].random_splits(split1=0.7,
                                                              split2=0.85,
                                                              random_seed=42)

        for IF in IFs:
            with self.subTest(msg=f"{IF.dx_mappers}"):
                for percentile_range in [2, 5, 10, 20, 25, 33, 50, 100]:
                    code_partitions = IF.dx_outcome_by_percentiles(
                        percentile_range, train_ids)
                    # Assert that union of all partitions recovers all the codes.
                    self.assertEqual(
                        set(IF.dx_outcome_extractor.index.values()),
                        set.union(*code_partitions))

                    # Assert that no intersection between the partitions
                    for i in range(len(code_partitions)):
                        for j in range(i + 1, len(code_partitions)):
                            self.assertCountEqual(
                                code_partitions[i] & code_partitions[j], set())


class CommonWindowedInterfaceTests(object):

    @classmethod
    def setUpClass(cls):
        dataset = ehr.mimicdataset.from_meta_json(
            'test/integration/fixtures/synthetic_mimic/mimic_syn_meta.json')
        interface = ehr.Subject_JAX.from_dataset(dataset, {
            'dx': 'dx_flatccs',
            'dx_outcome': 'dx_flatccs_v1'
        })
        cls.win_features = ehr.WindowedInterface_JAX(interface)

    def test_tabular(self):
        X, y = self.win_features.tabular_features()
        self.assertTrue(set(X.flatten()) & set(y.flatten()) == {0, 1})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
