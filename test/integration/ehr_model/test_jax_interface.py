"""Integration Test for EHR data model"""

import unittest

from icenode.ehr_model.jax_interface import (DxInterface_JAX,
                                             create_patient_interface)


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


if __name__ == '__main__':
    unittest.main()
