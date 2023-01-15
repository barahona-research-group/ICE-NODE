"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ml -v -t .
"""

import unittest
import os
import jax.random as jrandom
from typing import List, Tuple
from lib import ml
from lib.ehr.coding_scheme import DxICD10, DxICD9, DxCCS, PrICD9
from lib.ehr.outcome import OutcomeExtractor
from lib.ehr.jax_interface import Subject_JAX
from lib.ehr.dataset import MIMIC3EHRDataset, MIMIC4EHRDataset
from lib.utils import load_config, load_params
from lib.embeddings import embeddings_from_conf


def setUpModule():
    global m3_interface, m4_interface, model_configs

    m3_dataset = MIMIC3EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')
    m4_dataset = MIMIC4EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic4_syn_meta.json')
    icd9_outcome = OutcomeExtractor('dx_icd9_filter_v1')
    flatccs_outcome = OutcomeExtractor('dx_flatccs_filter_v1')

    if3_f = lambda scheme: Subject_JAX.from_dataset(m3_dataset, scheme)
    if4_f = lambda scheme: Subject_JAX.from_dataset(m4_dataset, scheme)

    m3_interface = {
        'ccs':
        if3_f(dict(dx=DxCCS(), dx_outcome=flatccs_outcome)),
        'icd10':
        if3_f(dict(dx=DxICD10(), dx_outcome=icd9_outcome)),
        'icd9':
        if3_f(dict(dx=DxICD9(), dx_outcome=icd9_outcome)),
        'icd9dag':
        if3_f(dict(dx=DxICD9(), dx_outcome=icd9_outcome, dx_dagvec=True))
    }

    m4_interface = {
        'ccs':
        if4_f(dict(dx=DxCCS(), dx_outcome=flatccs_outcome)),
        'icd9dag':
        if4_f(
            dict(dx=DxICD9(),
                 dx_dagvec=True,
                 pr=PrICD9(),
                 pr_dagvec=True,
                 dx_outcome=icd9_outcome))
    }
    _loadconfig = lambda fname: load_config(
        f'test/integration/fixtures/model_configs/{fname}.json')
    fnames = [
        'dx_winlogreg', 'dx_gru_m', 'dx_gru_g', 'icd9_dx_gru_g', 'dx_retain_m',
        'dx_icenode_2lr_m'
    ]
    model_configs = {fname: _loadconfig(fname) for fname in fnames}


def tearDownModule():
    pass


class DxCommonTests(object):

    @classmethod
    def setUpClass(cls):
        cls.model_if_pairs: List[Tuple[Type[ml.AbstractModel], Subject_JAX]] = []
        cls.config = dict()

    def test_train_read_write_params(self):
        for (model_cls, IF) in self.model_if_pairs:
            splits = IF.random_splits(0.7, 0.85, random_seed=42)

            emb_models = embeddings_from_conf(self.config["emb"], IF, splits[0])
            model =
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

    def setUp(self):
        self.config = model_configs['dx_winlogreg']
        cls.configs = []
        for class_weight in ml.logreg_loss_multinomial_mode.keys():
            config = c.copy()
            config['class_weight'] = class_weight
            cls.configs.append(config)

        cls.model_cls = ml.WLR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxWindowLogReg_Sklearn(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_winlogreg']
        cls.model_cls = ml.WLR_SK
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxGRU_M(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_gru_m']
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxGRU_M4CCS_M(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_gru_m']
        cls.model_cls = ml.GRU
        cls.interface = m4_interface_ccs
        cls.splits = m4_splits


class TestDxGRU_G(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_gru_g']
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestICD9DxGRU_G(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['icd9_dx_gru_g']
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxRETAIN(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_retain_m']
        cls.model_cls = ml.RETAIN
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_LazyLoad(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.env_patcher = unittest.mock.patch.dict(
            os.environ, {"ICENODE_INTERFACE_MAX_SIZE_GB": "0.0"})
        cls.env_patcher.start()

        cls.model_cls = ICENODE_2LR
        cls.interface = Subject_JAX.from_dataset(
            m3_dataset, {
                'dx': DxCCS(),
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

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_g']
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_UNIFORM_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM1(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_UNIFORM1D_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM0(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_UNIFORM0D_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxPrICENODE_G(DxCommonTests, unittest.TestCase):

    def setUp(self):
        self.config = self.configs['dxpr_icenode_2lr_g']
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
