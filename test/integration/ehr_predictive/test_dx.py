"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ehr_predictive -v -t .
"""

import unittest
import os

from icenode.ehr_predictive.trainer import MinibatchLogger
from icenode.ehr_predictive.dx_window_logreg import (
    WindowLogReg, logreg_loss_multinomial_mode, WindowLogReg_Sklearn)
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

    def setUp(self):
        self.models = []
        for config in self.configs:
            model = self.model_cls.create_model(config, dx_interface, [])
            state = model.init(config)

            self.assertTrue(callable(model))
            self.assertTrue(state is not None)
            self.models.append((model, state))

    def test_train_read_write_params(self):
        for (model, state), config in zip(self.models, self.configs):
            results = model.get_trainer()(model=model,
                                          m_state=state,
                                          config=config,
                                          splits=splits,
                                          rng_seed=42,
                                          reporters=[MinibatchLogger()])
            model_, state_ = results['model']
            test_out1 = model_(model_.get_params(state_),
                               splits[2])['risk_prediction']
            param_fname = f'test_{hash(str(config))}.pickle'
            model_.write_params(state_, param_fname)
            params = load_params(param_fname)
            os.remove(param_fname)

            state_ = model_.init_with_params(config, params)
            test_out2 = model_(model_.get_params(state_),
                               splits[2])['risk_prediction']

            self.assertEqual(test_out1, test_out2)


class TestDxWindowLogReg(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = load_config(
            'test/integration/fixtures/model_configs/dx_winlogreg.json')

        cls.configs = []
        for class_weight in logreg_loss_multinomial_mode.keys():
            config = c.copy()
            config['class_weight'] = class_weight
            cls.configs.append(config)

        cls.model_cls = WindowLogReg


class TestDxWindowLogReg_Sklearn(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_winlogreg.json')
        ]
        cls.model_cls = WindowLogReg_Sklearn


class TestDxGRU_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_m.json')
        ]
        cls.model_cls = GRAM


class TestDxGRU_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_g.json')
        ]
        cls.model_cls = GRAM


class TestDxRETAIN(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_retain_m.json')
        ]
        cls.model_cls = RETAIN


class TestDxICENODE_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ICENODE


class TestDxICENODE_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_g.json'
            )
        ]
        cls.model_cls = ICENODE


class TestDxICENODE_M_UNIFORM(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ICENODE_UNIFORM


if __name__ == '__main__':
    unittest.main()
