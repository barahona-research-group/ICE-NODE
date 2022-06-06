"""Integration Test for EHR data model"""

import unittest

from test.integration.common import load_mimic_files
from icenode.ehr_model.ccs_dag import CCSDAG
from icenode.ehr_model.jax_interface import (SubjectDiagSequenceJAXInterface,
                                             DiagnosisJAXInterface)


def setUpModule():
    global mimic3_interface


class TestDxGRAM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_subjects = load_mimic_files(
            'test/integration/fixtures/synthetic_mimic/adm_df.csv.gz',
            'test/integration/fixtures/synthetic_mimic/diag_df.csv.gz')
        cls.interface_seq = SubjectDiagSequenceJAXInterface(
            test_subjects, CCSDAG())
        cls.interface_ts = DiagnosisJAXInterface(test_subjects, CCSDAG())

    def test_split(self):

        def _test_split(interface):
            train_ids, valid_ids, test_ids = interface.random_splits(
                split1=0.7, split2=0.85, random_seed=42)

            self.assertCountEqual(
                set(train_ids) | set(valid_ids) | set(test_ids),
                interface.subjects.keys())

        _test_split(self.interface_seq)
        _test_split(self.interface_ts)


if __name__ == '__main__':
    unittest.main()
