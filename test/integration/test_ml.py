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
    global m3_interface, m3_interface_icd10, m3_interface_icd9, m3_interface_icd9_dagvec, m3_splits, m3_code_groups, m3_dataset, m4_interface, m4_interface_ccs, m4_splits, m4_code_groups

    m3_dataset = ehr.MIMIC3EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')
    m4_dataset = ehr.MIMIC4EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic4_syn_meta.json')
    m3_interface = ehr.Subject_JAX.from_dataset(
        m3_dataset, {
            'dx': 'dx_ccs',
            'dx_outcome': 'dx_flatccs_filter_v1'
        })

    m3_interface_icd10 = ehr.Subject_JAX.from_dataset(
        m3_dataset, {
            'dx': 'dx_icd10',
            'dx_outcome': 'dx_icd9_filter_v1'
        })

    m3_interface_icd9 = ehr.Subject_JAX.from_dataset(
        m3_dataset, {
            'dx': 'dx_icd9',
            'dx_outcome': 'dx_icd9_filter_v1'
        })

    m3_interface_icd9_dagvec = ehr.Subject_JAX.from_dataset(
        m3_dataset, {
            'dx': 'dx_icd9',
            'dx_dagvec': True,
            'dx_outcome': 'dx_icd9_filter_v1'
        })

    m3_splits = m3_interface.random_splits(split1=0.7,
                                           split2=0.85,
                                           random_seed=42)
    m3_code_groups = m3_interface.dx_outcome_by_percentiles(20)

    m4_interface = ehr.Subject_JAX.from_dataset(
        m4_dataset, {
            'dx': 'dx_icd9',
            'dx_dagvec': True,
            'pr': 'pr_icd9',
            'pr_dagvec': True,
            'dx_outcome': 'dx_icd9_filter_v1'
        })

    m4_interface_ccs = ehr.Subject_JAX.from_dataset(
        m4_dataset, {
            'dx': 'dx_ccs',
            'dx_outcome': 'dx_flatccs_filter_v1'
        })

    m4_splits = m4_interface.random_splits(split1=0.7,
                                           split2=0.85,
                                           random_seed=42)
    m4_code_groups = m4_interface.dx_outcome_by_percentiles(20)


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
                                                self.splits[0])
            state = model.init(config)

            self.assertTrue(callable(model), msg=f"config: {config}")
            self.assertTrue(state is not None, msg=f"config: {config}")
            self.models.append((model, state))

    def test_train_read_write_params(self):
        for (model, state), config in zip(self.models, self.configs):
            results = model.get_trainer()(model=model,
                                          m_state=state,
                                          config=config,
                                          splits=self.splits,
                                          rng_seed=42,
                                          reporters=[ml.MinibatchLogger()])
            model_, state_ = results['model']
            test_out1 = model_(model_.get_params(state_),
                               self.splits[2])['risk_prediction']
            param_fname = f'test_{hash(str(config))}.pickle'
            model_.write_params(state_, param_fname)
            params = load_params(param_fname)
            os.remove(param_fname)

            state_ = model_.init_with_params(config, params)
            test_out2 = model_(model_.get_params(state_),
                               self.splits[2])['risk_prediction']

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
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxWindowLogReg_Sklearn(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_winlogreg.json')
        ]
        cls.model_cls = ml.WLR_SK
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxGRU_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_m.json')
        ]
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxGRU_M4CCS_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_m.json')
        ]
        cls.model_cls = ml.GRU
        cls.interface = m4_interface_ccs
        cls.splits = m4_splits


class TestDxGRU_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_gru_g.json')
        ]
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestICD9DxGRU_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/icd9_dx_gru_g.json')
        ]
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxRETAIN(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_retain_m.json')
        ]
        cls.model_cls = ml.RETAIN
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


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
            m3_dataset, {
                'dx': 'dx_ccs',
                'dx_outcome': 'dx_flatccs_filter_v1'
            })
        cls.splits = m3_splits

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
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_UNIFORM_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM1(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_UNIFORM1D_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM0(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dx_icenode_2lr_m.json'
            )
        ]
        cls.model_cls = ml.ICENODE_UNIFORM0D_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxPrICENODE_G(DxCommonTests, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.configs = [
            load_config(
                'test/integration/fixtures/model_configs/dxpr_icenode_2lr_g.json'
            )
        ]
        cls.model_cls = ml.PR_ICENODE
        cls.interface = m4_interface
        cls.splits = m4_splits


class TestGlove(unittest.TestCase):

    def test_glove(self):
        glove_E = E.glove_representation(category='dx',
                                         subject_interface=m3_interface,
                                         train_ids=m3_splits[0],
                                         vector_size=150,
                                         iterations=30,
                                         window_size_days=730)

        glove_E = E.glove_representation(category='dx',
                                         subject_interface=m3_interface_icd10,
                                         train_ids=m3_splits[0],
                                         vector_size=150,
                                         iterations=30,
                                         window_size_days=730)

        glove_E = E.glove_representation(category='dx',
                                         subject_interface=m3_interface_icd9,
                                         train_ids=m3_splits[0],
                                         vector_size=150,
                                         iterations=30,
                                         window_size_days=730)


if __name__ == '__main__':
    unittest.main()
