import unittest
import random

import numpy as np
import jax.tree_util as jtu
import jax.random as jrandom
import equinox as eqx

from lib.ehr.coding_scheme import DxICD9
from lib.ehr.outcome import OutcomeExtractor
from lib.ehr.jax_interface import Subject_JAX
from lib.ehr.dataset import MIMIC3EHRDataset
from lib.embeddings import GRAM, CachedGRAM, train_glove


def setUpModule():
    global m3_interface, gram_config, subjects_sample

    m3_dataset = MIMIC3EHRDataset.from_meta_json(
        'test/integration/fixtures/synthetic_mimic/mimic3_syn_meta.json')

    m3_interface = Subject_JAX.from_dataset(
        m3_dataset, {
            'dx': DxICD9(),
            'dx_dagvec': True,
            'dx_outcome': OutcomeExtractor('dx_icd9_filter_v1')
        })

    m3_subjects = list(m3_interface.keys())
    m3_coocurrence = m3_interface.dx_augmented_coocurrence(m3_subjects,
                                                           window_size_days=60)
    m3_glove = train_glove(m3_coocurrence, embeddings_size=30, iterations=5)
    m3_ancestors_mat = m3_interface.dx_make_ancestors_mat()
    subjects_sample = random.sample(m3_subjects, 5)

    gram_config = {
        "classname": "CachedGRAM", 
            "attention_size": 20,
        "attention_method": "tanh",
        "ancestors_mat": m3_ancestors_mat,
        "basic_embeddings": m3_glove,
        "key": jrandom.PRNGKey(0)
    }


def tearDownModule():
    pass


class CachedGRAMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gram0 = GRAM(**gram_config)
        cls.gramC = CachedGRAM(**gram_config)

    def test_encoding_difference(self):

        # Compare paramters themselves
        p0 = eqx.filter(self.gram0, eqx.is_inexact_array)
        pC = eqx.filter(self.gramC, eqx.is_inexact_array)

        for p0, pC in zip(jtu.tree_leaves(p0), jtu.tree_leaves(pC)):
            self.assertSequenceEqual(p0.tolist(), pC.tolist())

        G0 = self.gram0.compute_embeddings_mat()
        GC = self.gramC.compute_embeddings_mat()

        for i in subjects_sample:
            for adm in m3_interface[i]:
                v = adm.dx_vec

                v0 = self.gram0.encode(G0, v)
                vC = self.gramC.encode(GC, v)
                np.testing.assert_allclose(v0, vC, atol=0, rtol=1e-5)
