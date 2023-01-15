"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ml -v -t .
"""

import unittest
import string
import os
import uuid
import random
import jax.random as jrandom
from typing import List, Tuple, Type
from collections import namedtuple
from lib import ml
from lib.ehr.coding_scheme import DxICD10, DxICD9, DxCCS, PrICD9
from lib.ehr.outcome import OutcomeExtractor
from lib.ehr.jax_interface import Subject_JAX
from lib.ehr.dataset import MIMIC3EHRDataset, MIMIC4EHRDataset
from lib.utils import load_config, load_params, write_params
from lib.embeddings import embeddings_from_conf


def setUpModule():
    global m3_interface, m4_interface, model_configs, key

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
    key = jrandom.PRNGKey(443)


def tearDownModule():
    pass


TestActors = namedtuple('TestActors',
                        ['model', 'conf', 'trainer', 'interface', 'splits'])


class DxCommonTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.actors: List[TestActors] = []

    def test_train_read_write_params(self):
        for i, actors in enumerate(self.actors):
            res = actors.trainer(model=actors.model,
                                 subject_interface=actors.interface,
                                 splits=actors.splits,
                                 rng_seed=42,
                                 reporters=[ml.MinibatchLogger()])
            model = res['model']
            preds1 = model(actors.interface, actors.splits[2])['predictions']

            param_fname = f'test_{str(uuid.uuid4())}.eqx'
            model.write_params(param_fname)

            model = model.from_config(actors.config, actors.IF,
                                      actors.splits[0], key)
            model = model.load_params(param_fname)
            os.remove(param_fname)

            preds2 = model(actors.interface, actors.splits[2])['predictions']

            self.assertEqual(preds1, preds2, msg=f"config: {actors.config}")


class TestDxWindowLogReg(DxCommonTests):

    def setUp(self):
        config = model_configs['dx_winlogreg']
        self.models = []
        for IF in list(m3_interface.values()) + list(m4_interface.values()):
            splits = IF.random_splits(0.7, 0.85, 42)
            model = ml.WindowLogReg.from_config(config, IF, splits[0], key)
            trainer_cls = getattr(ml, config["training"]["classname"])
            self.assertEqual(trainer_cls, ml.LassoNetTrainer)
            trainer = trainer_cls(**config["training"])
            self.models.append(TestActors(model, config, trainer, IF, splits))


class TestDxGRU_M(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_gru_m']
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits

    def setUp(self):
        config = model_configs['dx_gru_m']
        self.models = []
        for IF in list(m3_interface.values()) + list(m4_interface.values()):
            splits = IF.random_splits(0.7, 0.85, 42)
            model = ml.WindowLogReg.from_config(config, IF, splits[0], key)
            trainer_cls = getattr(ml, config["training"]["classname"])
            self.assertEqual(trainer_cls, ml.LassoNetTrainer)
            trainer = trainer_cls(**config["training"])
            self.models.append(TestActors(model, config, trainer, IF, splits))



class TestDxGRU_M4CCS_M(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_gru_m']
        cls.model_cls = ml.GRU
        cls.interface = m4_interface_ccs
        cls.splits = m4_splits


class TestDxGRU_G(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_gru_g']
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestICD9DxGRU_G(DxCommonTests):

    def setUp(self):
        self.config = self.configs['icd9_dx_gru_g']
        cls.model_cls = ml.GRU
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxRETAIN(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_retain_m']
        cls.model_cls = ml.RETAIN
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_LazyLoad(DxCommonTests):

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


class TestDxICENODE_G(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_g']
        cls.model_cls = ml.ICENODE_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_UNIFORM_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM1(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_UNIFORM1D_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxICENODE_M_UNIFORM0(DxCommonTests):

    def setUp(self):
        self.config = self.configs['dx_icenode_2lr_m']
        cls.model_cls = ml.ICENODE_UNIFORM0D_2LR
        cls.interface = m3_interface
        cls.splits = m3_splits


class TestDxPrICENODE_G(DxCommonTests):

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
