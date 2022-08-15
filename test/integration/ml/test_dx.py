"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ml -v -t .
"""

import unittest
import os
from icenode import ml
from icenode import ehr
from icenode import embeddings as E
from icenode.utils import load_config, load_params


def setUpModule():
    global interface, interface_icd10, interface_icd9, splits, code_groups, dataset
    dataset = ehr.MIMICDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic_syn_meta.json')
    interface = ehr.Subject_JAX.from_dataset(
        dataset, {
            'dx': 'dx_ccs',
            'dx_outcome': 'dx_flatccs_filter_v1'
        })

    interface_icd10 = ehr.Subject_JAX.from_dataset(
        dataset, {
            'dx': 'dx_icd10',
            'dx_outcome': 'dx_flatccs_filter_v1'
        })

    interface_icd9 = ehr.Subject_JAX.from_dataset(
        dataset, {
            'dx': 'dx_icd9',
            'dx_outcome': 'dx_flatccs_filter_v1'
        })
    splits = interface.random_splits(split1=0.7, split2=0.85, random_seed=42)
    code_groups = interface.dx_outcome_by_percentiles(20)


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
            model = self.model_cls.create_model(config, self.interface,
                                                splits[0])
            state = model.init(config)

            self.assertTrue(callable(model), msg=f"config: {config}")
            self.assertTrue(state is not None, msg=f"config: {config}")
            self.models.append((model, state))

    def test_train_read_write_params(self):
        for (model, state), config in zip(self.models, self.configs):
            results = model.get_trainer()(model=model,
                                          m_state=state,
                                          config=config,
                                          splits=splits,
                                          rng_seed=42,
                                          reporters=[ml.MinibatchLogger()])
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

            self.assertEqual(test_out1, test_out2, msg=f"config: {config}")


class TestDxWindowLogReg(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = load_config(
            'test/integration/fixtures/model_configs/dx_winlogreg.json')

        cls.configs = []
        for class_weight in ml.logreg_loss_multinomial_mode.keys():
            config = c.copy()
            config['class_weight'] = class_weight
            cls.configs.append(config)

        cls.model_cls = ml.WLR
        cls.interface = interface


class TestDxWindowLogReg_Sklearn(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_winlogreg.json')
        ]
        cls.model_cls = ml.WLR_SK
        cls.interface = interface


class TestDxGRU_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_m.json')
        ]
        cls.model_cls = ml.GRU
        cls.interface = interface


class TestDxGRU_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_g.json')
        ]
        cls.model_cls = ml.GRU
        cls.interface = interface


class TestDxRETAIN(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_retain_m.json')
        ]
        cls.model_cls = ml.RETAIN
        cls.interface = interface


class TestDxICENODE_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = interface


class TestDxICENODE_M_LazyLoad(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env_patcher = unittest.mock.patch.dict(
            os.environ, {"ICENODE_INTERFACE_MAX_SIZE_GB": "0.0"})
        cls.env_patcher.start()

        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = ehr.Subject_JAX.from_dataset(
            dataset, {
                'dx': 'dx_ccs',
                'dx_outcome': 'dx_flatccs_filter_v1'
            })

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        cls.env_patcher.stop()

    def test_env_effect(self):
        self.assertEqual(self.interface.data_max_size_gb, 0.0)


class TestDxICENODE_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_g.json'
            )
        ]
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = interface


class TestDxICENODE_M_UNIFORM(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_UNIFORM_2LR
        cls.interface = interface


class TestGlove(unittest.TestCase):

    def test_glove(self):
        glove_E = E.glove_representation(category='dx',
                                         subject_interface=interface,
                                         train_ids=splits[0],
                                         vector_size=150,
                                         iterations=30,
                                         window_size_days=730)

        glove_E = E.glove_representation(category='dx',
                                         subject_interface=interface_icd10,
                                         train_ids=splits[0],
                                         vector_size=150,
                                         iterations=30,
                                         window_size_days=730)

        glove_E = E.glove_representation(category='dx',
                                         subject_interface=interface_icd9,
                                         train_ids=splits[0],
                                         vector_size=150,
                                         iterations=30,
                                         window_size_days=730)


if __name__ == '__main__':
    unittest.main()
