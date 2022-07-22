"""Integration Test for EHR data model"""

import unittest
import logging

from icenode.ehr_model.jax_interface import (DxInterface_JAX,
                                             DxWindowedInterface_JAX,
                                             create_patient_interface)
from icenode.ehr_model.ccs_dag import ccs_dag


class TestDxInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interface = create_patient_interface(
            'test/integration/fixtures/synthetic_mimic')

    def test_split(self):

        train_ids, valid_ids, test_ids = self.interface.random_splits(
            split1=0.7, split2=0.85, random_seed=42)

        self.assertCountEqual(
            set(train_ids) | set(valid_ids) | set(test_ids),
            self.interface.subjects.keys())

    def test_code_frequency_paritions(self):

        train_ids, valid_ids, test_ids = self.interface.random_splits(
            split1=0.7, split2=0.85, random_seed=42)

        for percentile_range in [2, 5, 10, 20, 25, 33, 50, 100]:
            partitioners = [
                self.interface.dx_flatccs_by_percentiles,
                self.interface.dx_ccs_by_percentiles
            ]
            index_sets = [
                set(ccs_dag.dx_flatccs_idx.values()),
                set(ccs_dag.dx_ccs_idx.values())
            ]
            for partitioner, index_set in zip(partitioners, index_sets):
                code_partitions = partitioner(percentile_range, train_ids)
                # Assert that union of all partitions recovers all the codes.
                self.assertEqual(index_set, set.union(*code_partitions))

                # Assert that no intersection between the partitions
                for i in range(len(code_partitions)):
                    for j in range(i + 1, len(code_partitions)):
                        self.assertCountEqual(
                            code_partitions[i] & code_partitions[j], set())


class TestDxWindowedInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.interface = create_patient_interface(
            'test/integration/fixtures/synthetic_mimic')
        cls.win_features = DxWindowedInterface_JAX(cls.interface)

    def test_tabular(self):
        X, y = self.win_features.tabular_features()
        self.assertTrue(set(X.flatten()) & set(y.flatten()) == {0, 1})


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
