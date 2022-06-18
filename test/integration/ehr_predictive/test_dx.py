"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ehr_predictive -v -t .
"""

import unittest
import random
from sklearn.exceptions import ConvergenceWarning

from icenode.ehr_predictive.trainer import MinibatchLogger
from icenode.ehr_predictive.dx_window_logreg import (WindowLogReg,
                                                     WindowLogReg_Sklearn)
from icenode.ehr_predictive.dx_gram import GRAM
from icenode.ehr_predictive.dx_retain import RETAIN
from icenode.ehr_predictive.dx_icenode_2lr import ICENODE
from icenode.ehr_predictive.dx_icenode_uniform2lr import ICENODE as ICENODE_UNIFORM

from icenode.utils import load_config, load_params
from icenode.ehr_model.jax_interface import (create_patient_interface)


def setUpModule():
    global dx_interface, splits, code_groups

    dx_interface = create_patient_interface(
        'test/integration/fixtures/synthetic_mimic')
    splits = dx_interface.random_splits(split1=0.7,
                                        split2=0.85,
                                        random_seed=42)
    code_groups = dx_interface.dx_flatccs_by_percentiles(20)


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
        model = self.model_cls.create_model(self.config, dx_interface, [])
        state = model.init(self.config)

        self.assertTrue(callable(model))
        self.assertTrue(state is not None)

    def test_train(self):
        model = self.model_cls.create_model(self.config, dx_interface, [])
        state = model.init(self.config)
        model.get_trainer()(model=model,
                            m_state=state,
                            config=self.config,
                            splits=splits,
                            rng_seed=42,
                            reporters=[MinibatchLogger()])


class TestDxWindowLogReg(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_winlogreg.json')
        cls.model_cls = WindowLogReg


class TestDxWindowLogReg_Sklearn(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_winlogreg.json')
        cls.model_cls = WindowLogReg_Sklearn


class TestDxGRU_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_gru_m.json')
        cls.model_cls = GRAM


class TestDxGRU_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_gru_g.json')
        cls.model_cls = GRAM


class TestDxRETAIN(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_retain_m.json')
        cls.model_cls = RETAIN


class TestDxICENODE_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json')
        cls.model_cls = ICENODE


class TestDxICENODE_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_icenode_2lr_g.json')
        cls.model_cls = ICENODE


class TestDxICENODE_M_UNIFORM(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(
            'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json')
        cls.model_cls = ICENODE_UNIFORM


if __name__ == '__main__':
    unittest.main()
