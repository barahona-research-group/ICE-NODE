"""Integration Test for EHR Dx predictive model
To execute this test from the root directory:
    python -m unittest discover -s test/integration/ml -v -t .
"""

import unittest
import string
import os
import uuid
import random
from typing import List, Tuple, Type
from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx

from lib import ml
from lib.ehr.coding_scheme import DxICD10, DxICD9, DxCCS, PrICD9
from lib.ehr.outcome import OutcomeExtractor
from lib.ehr.jax_interface import Subject_JAX
from lib.ehr.dataset import MIMIC3EHRDataset, MIMIC4EHRDataset
from lib.utils import load_config
from lib.embeddings import embeddings_from_conf
from lib.metric import (CodeAUC, UntilFirstCodeAUC, AdmissionAUC,
                        MetricsCollection)


def setUpModule():
    global m3_dataset, m3_interface, m4_interface, model_configs, key

    m3_dataset = MIMIC3EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')
    m4_dataset = MIMIC4EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic4_syn_meta.json')
    icd9_outcome = OutcomeExtractor('dx_icd9_filter_v1')
    flatccs_outcome = OutcomeExtractor('dx_flatccs_filter_v1')

    if3_f = lambda scheme: Subject_JAX.from_dataset(m3_dataset, scheme)
    if4_f = lambda scheme: Subject_JAX.from_dataset(m4_dataset, scheme)

    m3_interface = {
        'ccs': if3_f(dict(dx=DxCCS(), dx_outcome=flatccs_outcome)),
        # 'icd10':
        # if3_f(dict(dx=DxICD10(), dx_outcome=icd9_outcome)),
        # 'icd9':
        # if3_f(dict(dx=DxICD9(), dx_outcome=icd9_outcome)),
        # 'icd9dag':
        # if3_f(dict(dx=DxICD9(), dx_outcome=icd9_outcome, dx_dagvec=True))
    }

    m4_interface = {
        # 'ccs': if4_f(dict(dx=DxCCS(), dx_outcome=flatccs_outcome))
    }
    _loadconfig = lambda fname: load_config(
        f'test/integration/fixtures/model_configs/{fname}.json')
    fnames = [
        'dx_winlogreg', 'dx_gru_m', 'dx_gru_g', 'dx_retain_m',
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

        def assert_equal_models(model1, model2):
            m1 = jtu.tree_leaves(eqx.filter(model1, eqx.is_inexact_array))
            m2 = jtu.tree_leaves(eqx.filter(model2, eqx.is_inexact_array))
            self.assertTrue(all((a1 == a2).all() for a1, a2 in zip(m1, m2)))
            _, def1 = jtu.tree_flatten(model1)
            _, def2 = jtu.tree_flatten(model2)
            try:
                self.assertEqual(def1, def2)
            except:
                for module in ('dx_emb', 'dx_dec'):
                    if not hasattr(model1, module):
                        continue
                    model1_m = getattr(model1, module)
                    model2_m = getattr(model2, module)

                    try:
                        _, mm1 = jtu.tree_flatten(model1_m)
                        _, mm2 = jtu.tree_flatten(model2_m)
                        self.assertEqual(mm1, mm2)
                    except:
                        print(f"Issue regarding module {module}")

                self.assertTrue(False, msg=f"\ndef1: {def1}\ndef2: {def2}")

        for i, actors in enumerate(self.actors):
            print(actors.conf)
            metrics = [
                CodeAUC(actors.interface),
                AdmissionAUC(actors.interface),
                UntilFirstCodeAUC(actors.interface)
            ]
            objective = metrics[2].aggregate_extractor({
                'field': 'auc',
                'aggregate': 'mean'
            })
            history = ml.MetricsHistory(MetricsCollection(metrics))
            res = actors.trainer(model=actors.model,
                                 subject_interface=actors.interface,
                                 splits=actors.splits,
                                 history=history,
                                 prng_seed=42,
                                 reporters=[ml.MinibatchLogger(actors.conf)])

            history = res['history']
            self.assertTrue(objective(history.train_df()) > 0.4)

            model1 = res['model']
            preds1 = model1(actors.interface,
                            actors.splits[2],
                            args=dict(eval_only=True))['predictions']

            param_fname = f'test_{str(uuid.uuid4())}.eqx'
            model1.write_params(param_fname)
            # jax.debug.print("model1: {}", model1)

            model2 = type(actors.model).from_config(actors.conf,
                                                    actors.interface,
                                                    actors.splits[0], key)
            model2 = model2.load_params(param_fname)
            # jax.debug.print("model2: {}", model2)
            os.remove(param_fname)
            jax.clear_backends()
            assert_equal_models(model1, model2)

            preds2 = model2(actors.interface,
                            actors.splits[2],
                            args=dict(eval_only=True))['predictions']

            self.assertTrue(preds1.equals(preds2),
                            msg=f"config: {actors.conf}")


# class TestDxWindowLogReg(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_winlogreg']
#         self.actors = []
#         for IF in list(m3_interface.values()) + list(m4_interface.values()):
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.WindowLogReg.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.LassoNetTrainer)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))


# class TestDxGRU_M(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_gru_m']
#         self.actors = []
#         for IF in list(m3_interface.values()) + list(m4_interface.values()):
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.GRU.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.Trainer)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))


# class TestDxRETAIN(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_retain_m']
#         self.actors = []
#         IFs = [m3_interface['ccs']]
#         for IF in IFs:
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.RETAIN.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.Trainer)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))


class TestDxICENODE_M(DxCommonTests):

    def setUp(self):
        config = model_configs['dx_icenode_2lr_m']
        self.actors = []
        IFs = [m3_interface['ccs']]
        for IF in IFs:
            splits = IF.random_splits(0.7, 0.85, 42)
            model = ml.ICENODE.from_config(config, IF, splits[0], key)
            trainer_cls = getattr(ml, config["training"]["classname"])
            self.assertEqual(trainer_cls, ml.ODETrainer2LR)
            trainer = trainer_cls(**config["training"])
            self.actors.append(TestActors(model, config, trainer, IF, splits))


# class TestDxICENODE_M_LazyLoad(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_icenode_2lr_m']
#         self.actors = []
#         self.env_patcher = unittest.mock.patch.dict(
#             os.environ, {"ICENODE_INTERFACE_MAX_SIZE_GB": "0.0"})
#         self.interface = Subject_JAX.from_dataset(
#             m3_dataset, {
#                 'dx': DxCCS(),
#                 'dx_outcome': OutcomeExtractor('dx_flatccs_filter_v1')
#             })
#         IFs = [self.interface]
#         for IF in IFs:
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.ICENODE.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.ODETrainer2LR)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))

#     def tearDown(self):
#         super().tearDownClass()
#         self.env_patcher.stop()

#     def test_env_effect(self):
#         self.assertEqual(self.interface.data_max_size_gb, 0.0)


# class TestDxICENODE_M_UNIFORM(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_icenode_2lr_m']
#         self.actors = []
#         IFs = [m3_interface['ccs']]
#         for IF in IFs:
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.ICENODE_UNIFORM.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.ODETrainer2LR)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))


# class TestDxICENODE_M_UNIFORM0(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_icenode_2lr_m']
#         self.actors = []
#         IFs = [m3_interface['ccs']]
#         for IF in IFs:
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.ICENODE_ZERO.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.ODETrainer2LR)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))


# class TestDxICENODE_G(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_icenode_2lr_g']
#         self.actors = []
#         IFs = [m3_interface['icd9dag']]
#         for IF in IFs:
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.ICENODE.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.ODETrainer2LR)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))

# class TestDxGRU_G(DxCommonTests):

#     def setUp(self):
#         config = model_configs['dx_gru_g']
#         self.actors = []
#         IFs = [m3_interface['icd9dag']]
#         for IF in IFs:
#             splits = IF.random_splits(0.7, 0.85, 42)
#             model = ml.GRU.from_config(config, IF, splits[0], key)
#             trainer_cls = getattr(ml, config["training"]["classname"])
#             self.assertEqual(trainer_cls, ml.Trainer)
#             trainer = trainer_cls(**config["training"])
#             self.actors.append(TestActors(model, config, trainer, IF, splits))

# class TestDxPrICENODE_G(DxCommonTests):

#     def setUp(self):
#         self.config = self.configs['dxpr_icenode_2lr_g']
#         cls.model_cls = ml.PR_ICENODE
#         cls.interface = m4_interface
#         cls.splits = m4_splits

if __name__ == '__main__':
    unittest.main()
