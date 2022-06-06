"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ehr_predictive -v -t .
"""

import unittest

import pandas as pd

from test.integration.common import load_mimic_files
from icenode.ehr_predictive.dx_gram import GRAM
from icenode.metric.common_metrics import evaluation_table
from icenode.utils import load_config, load_params
from icenode.ehr_model.ccs_dag import CCSDAG
from icenode.ehr_model.jax_interface import (SubjectDiagSequenceJAXInterface,
                                             DiagnosisJAXInterface)


def setUpModule():
    global subject_interface_ts, subject_interface_seq

    test_subjects = load_mimic_files(
        'test/integration/fixtures/synthetic_mimic/adm_df.csv.gz',
        'test/integration/fixtures/synthetic_mimic/diag_df.csv.gz')
    subject_interface_seq = SubjectDiagSequenceJAXInterface(
        test_subjects, CCSDAG())
    subject_interface_ts = DiagnosisJAXInterface(test_subjects, CCSDAG())


def tearDownModule():
    pass


class TestDxGRAM(unittest.TestCase):

    def setUp(self):
        self.dx_gram_m_config = load_config(
            'test/integration/fixtures/model_configs/dx_gram_m.json')

    def test_first(self):
        self.assertTrue(len(self.dx_gram_m_config) > 0)


if __name__ == '__main__':
    unittest.main()
