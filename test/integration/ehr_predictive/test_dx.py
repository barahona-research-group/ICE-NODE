"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ehr_predictive -v -t .
"""

import unittest
import random

import pandas as pd
from tqdm import tqdm

from test.integration.common import load_mimic_files

from icenode.ehr_predictive.abstract import minibatch_trainer, MinibatchLogger
from icenode.ehr_predictive.dx_gram import GRAM
from icenode.ehr_predictive.dx_retain import RETAIN
from icenode.ehr_predictive.dx_icenode_2lr import ICENODE
from icenode.ehr_predictive.dx_icenode_uniform2lr import ICENODE as ICENODE_UNIFORM

from icenode.metric.common_metrics import evaluation_table
from icenode.utils import load_config, load_params
from icenode.ehr_model.ccs_dag import CCSDAG
from icenode.ehr_model.jax_interface import (SubjectDiagSequenceJAXInterface,
                                             DiagnosisJAXInterface)


def setUpModule():
    global subject_interface_ts, subject_interface_seq, splits, code_groups

    test_subjects = load_mimic_files(
        'test/integration/fixtures/synthetic_mimic/adm_df.csv.gz',
        'test/integration/fixtures/synthetic_mimic/diag_df.csv.gz')
    subject_interface_seq = SubjectDiagSequenceJAXInterface(
        test_subjects, CCSDAG())
    subject_interface_ts = DiagnosisJAXInterface(test_subjects, CCSDAG())
    splits = subject_interface_ts.random_splits(split1=0.7,
                                                split2=0.85,
                                                random_seed=42)
    code_groups = subject_interface_ts.diag_flatccs_by_percentiles(20)


def tearDownModule():
    pass


class DxCommonTests(object):

    @classmethod
    def setUpClass(cls):
        # Unreachable, should be overriden.
        raise RuntimeError('Unreachable')

    def test_config(self):
        self.assertTrue(len(self.config) > 0)

    def test_create(self):
        model = self.model_cls.create_model(self.config, self.interface, [],
                                            None)
        state = model.init(self.config)

        self.assertTrue(callable(model))
        self.assertTrue(state is not None)

    def test_train(self):
        model = self.model_cls.create_model(self.config, self.interface, [],
                                            None)
        state = model.init(self.config)
        train_ids, val_ids, tst_ids = splits

        minibatch_trainer(model,
                          state,
                          self.config,
                          *splits,
                          rng=random.Random(42),
                          reporters=[MinibatchLogger()])


class TestDxGRAM(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_gram_m.json')
        cls.model_cls = GRAM
        cls.interface = subject_interface_seq


class TestDxRETAIN(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_retain_m.json')
        cls.model_cls = RETAIN
        cls.interface = subject_interface_seq


class TestDxICENODE(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json')
        cls.model_cls = ICENODE
        cls.interface = subject_interface_ts


class TestDxICENODE_UNIFORM(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json')
        cls.model_cls = ICENODE_UNIFORM
        cls.interface = subject_interface_ts


if __name__ == '__main__':
    unittest.main()
