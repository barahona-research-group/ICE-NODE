"""Integration Test for EHR data model"""

import unittest

from test.integration.common import load_mimic_files
from icenode.ehr_model.ccs_dag import CCSDAG
from icenode.ehr_model.jax_interface import SubjectDiagSequenceJAXInterface
from icenode.ehr_model.jax_interface import DiagSubject


class TestJAXInterface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mimic3_subjects = load_mimic_files(
            'test/integration/fixtures/mimic3/adm_df.csv.gz',
            'test/integration/fixtures/mimic3/diag_df.csv.gz')
        cls.mimic3_interface_seq = SubjectDiagSequenceJAXInterface(
            mimic3_subjects, CCSDAG())
        cls.mimic3_interface_ts = DiagSubject(mimic3_subjects, CCSDAG())

    def _test_split(self, interface):
        train_ids, valid_ids, test_ids = interface.random_splits(
            split1=0.7, split2=0.85, random_seed=42)

        self.assertCountEqual(
            set(train_ids) | set(valid_ids) | set(test_ids),
            interface.subjects.keys())

    def test_split_seq(self):
        self._test_split(self.mimic3_interface_seq)

    def test_split_ts(self):
        self._test_split(self.mimic3_interface_ts)


if __name__ == '__main__':
    unittest.main()
